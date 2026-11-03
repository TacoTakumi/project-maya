// src/core/glm_resplit.cu - the room check of a whole-card lend: may a card of the split be lent WHOLE (its part
// destroyed, the card free for another program), with its layers taken in by a neighbour part?  Two rules, on the
// figures the load already knows (resplit_room, glm_model.hpp):
//   room    a neighbour part - the previous or the next in the chain, the one with the most pool room above its own
//           floor (pool_keep slots a layer), the previous on a tie - has room in its pool for the card's dense
//           weights + KV + pool floor: the pool's slots give way to the layers that move in, down to the floor the
//           lend rule keeps anyway (glm_lend.cu);
//   RAM     the merged RAM tier - the neighbour's as pinned plus the lent part's, which the neighbour pins anew before
//           the lent part's own is released - stays within the neighbour's tier plus what the host can pin now under
//           the headroom the tier sizing keeps free (fast_setup: MemAvailable less STRATA_GLM_RAM_HEADROOM_GB, 6 GB
//           on a discrete card, 16 on unified memory).
// The footprint a pass reports is everything the part holds on its card as the VRAM line counts it - dense weights,
// sequence state, activations, a drafter, the verify window, the expert pool as mapped - in whole MiB rounded up.  The
// CUDA context is not in it: the card's total - free also holds other programs' memory, so it is not the part's.
//
// lend_whole_check runs the check for --lend-whole's cards at the load, once the split and the pools are known: a
// pass records the footprint on the part (whole_mib_, read by whole_mib(dev) - the VRAM line's whole_mib= field and
// the lender's whole_mib), a fail prints one line and the card keeps pool lending only.  The load never fails for it.
// Two rules of the lend itself come on top of resplit_room's: the first part is the model object the engine holds
// (it cannot be destroyed and loaded again, so it is neither lent nor the taker), and the last part holds the head,
// the sampler's buffers and the drafter (not lent whole; it may be the taker).
//
// THE LEND (lend_whole_plan, lend_whole_run; the lender in glm_lend.cu calls them from the engine's command loop with
// nothing in flight), in phases, each timed in the log line:
//   quiet    the head's window crew stopped and its commits waited for; on every part the background tier moves held
//            (bg_hold), the copy stream synchronized, fast_boundary run (every move landed, every route taken), no
//            move or drain left, then cudaDeviceSynchronize - every stream of every card (compute, copy, prefetch,
//            the prompt path's, the window lanes, the drafter's);
//   save     the conversation state to RAM: Glm5Model::slot_save's bytes (the snapshot's recurrent states, the open
//            indexer pools and the DSA caches up to the snapshot's position, one section a part).  No snapshot (no
//            request yet): nothing to carry, the chain is reset after the rebuild;
//   destroy  the lent part and the neighbour, unlinked from the chain first (a part's destructor takes the chain
//            after it along), the later one first: each part frees its device and pinned memory under its own
//            device, as at exit;
//   load     one part with the merged layer range on the neighbour's device, through load_pack as load_pack_split
//            loads a part - the same pack, context, drafter path and expert profile; its RAM tier budget is the two
//            tiers' sum (ram_budget_: the tier is not derived again, the RAM rule above was checked against this
//            figure), the pool carved from the card's free VRAM as at the load, the warm-up as at the load.  The part
//            count it loads with is the OLD one: the CPU lane's shared pool is sized from it, and the survivors keep
//            theirs (a new count would open a second pool beside it);
//   link     split_next_ re-linked around it; every part's part index, count and device list set to the new split;
//            fast_setup_finish on the new part (the CPU lane and the service thread); the prompt chunk capped at the
//            split's smallest; dflash_bind again (the capture rows and the drafter's layer map on every part); the
//            prompt path's pairings forgotten (calc_pf_state_ / dual_pf_state_, decided again at the next prompt);
//            each survivor's window skip word dropped (win.from, set again by the next pipelined window); the new
//            part's window slot bank set to the survivors' (every part flips its own at a window's begin and reads
//            the previous part's rows at the slot its bank names: they must agree);
//   restore  slot_load of the saved bytes onto the new split (Glm5SlotLayout maps each saved section onto whichever
//            live part holds its layers): the snapshot and the live state as before the lend;
//   release  cudaDeviceReset on the lent card: its primary context goes, and with it the process's presence there -
//            every allocation made under it was freed with the part, so nothing else of this process lives there.
// What is NOT carried: the host-RAM checkpoints (ckpt_save) of the two destroyed parts - a restore of one answers
// "holds none" and the engine reads the prompt again; and an EXL3 head split whose helper or last part died - the
// pick reads the whole head row on the last part from then on.
#include "glm_fast_state.hpp"

#include "strata/core/glm_model.hpp"
#include "strata/kernels/glm_fast.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <set>
#include <string>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace strata::core {

namespace {

std::string mib(uint64_t bytes) { return std::to_string(bytes >> 20); }

uint64_t room_of(const ResplitPart& p) { return p.pool > p.floor ? p.pool - p.floor : 0; }

// the RAM this process can still get, as the tier sizing reads it (glm_fast_path.cu's avail_ram_now); 0: unknown
uint64_t ram_avail_now() {
#ifdef _WIN32
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    return GlobalMemoryStatusEx(&ms) ? (uint64_t) std::min(ms.ullAvailPhys, ms.ullAvailPageFile) : 0;
#else
    long long kb = 0;
    if (FILE* f = std::fopen("/proc/meminfo", "r")) {
        char line[256];
        while (std::fgets(line, sizeof line, f))
            if (std::sscanf(line, "MemAvailable: %lld kB", &kb) == 1) break;
        std::fclose(f);
    }
    return kb > 0 ? (uint64_t) kb * 1024 + (uint64_t) std::max<int64_t>(0, glmfast::zfs_arc_reclaimable()) : 0;
#endif
}

}  // namespace

ResplitRoom resplit_room(const std::vector<ResplitPart>& parts, int dev, uint64_t ram_avail, uint64_t ram_headroom) {
    ResplitRoom r;
    int at = -1, n_at = 0;
    for (size_t i = 0; i < parts.size(); ++i)
        if (parts[i].dev == dev) {
            if (at < 0) at = (int) i;
            ++n_at;
        }
    if (at < 0) {
        r.reason = "not in the split";
        return r;
    }
    const std::string card = "CUDA" + std::to_string(dev);
    if (n_at > 1) {
        r.reason = card + " holds " + std::to_string(n_at) + " parts of the split: only a card with one part is lent whole";
        return r;
    }
    const ResplitPart& m = parts[(size_t) at];
    // the neighbours: the previous part first, so a tie on room keeps it
    struct Nb {
        int idx;
        uint64_t room;
    };
    std::vector<Nb> nbs;
    if (at > 0) nbs.push_back({at - 1, room_of(parts[(size_t) at - 1])});
    if ((size_t) at + 1 < parts.size()) nbs.push_back({at + 1, room_of(parts[(size_t) at + 1])});
    if (nbs.empty()) {
        r.reason = "no neighbour part: the split has one part";
        return r;
    }
    Nb best = nbs[0];
    for (const Nb& n : nbs)
        if (n.room > best.room) best = n;
    const uint64_t need = m.weights + m.kv + m.floor;
    if (best.room < need) {
        r.reason = "no room: needs " + mib(need) + " MiB (dense " + mib(m.weights) + " + KV " + mib(m.kv) +
                   " + pool floor " + mib(m.floor) + ") in a neighbour's pool above its floor; ";
        for (size_t i = 0; i < nbs.size(); ++i) {
            const ResplitPart& n = parts[(size_t) nbs[i].idx];
            r.reason += (i ? ", " : "") + ("CUDA" + std::to_string(n.dev)) + " has " + mib(nbs[i].room) +
                        " MiB (short " + mib(need - nbs[i].room) + " MiB)";
        }
        return r;
    }
    const ResplitPart& nb = parts[(size_t) best.idx];
    if (ram_avail > 0) {
        const uint64_t merged = nb.ram + m.ram;
        const uint64_t free_room = ram_avail > ram_headroom ? ram_avail - ram_headroom : 0;
        const uint64_t limit = nb.ram + free_room;
        if (merged > limit) {
            const std::string nbc = "CUDA" + std::to_string(nb.dev);
            r.reason = "RAM: the merged tier " + mib(merged) + " MiB (" + nbc + " " + mib(nb.ram) + " + " + card + " " +
                       mib(m.ram) + " pinned anew) is over the " + mib(limit) + " MiB the host can pin (" + nbc +
                       "'s tier + " + mib(free_room) + " MiB free under the headroom " + mib(ram_headroom) +
                       " MiB): short " + mib(merged - limit) + " MiB";
            return r;
        }
    }
    r.ok = true;
    r.into = best.idx;
    const uint64_t all = m.weights + m.kv + m.act + m.drafter + m.window + m.pool;
    r.whole_mib = (all + ((uint64_t) 1 << 20) - 1) >> 20;
    return r;
}


namespace {

// the headroom the RAM tier was sized against (fast_setup's head_gb)
uint64_t ram_headroom_bytes(const Glm5Model::FastState* F) {
    double head_gb = F != nullptr && F->unified_memory ? 16.0 : 6.0;
    if (const char* h = getenv("STRATA_GLM_RAM_HEADROOM_GB")) head_gb = std::max(0.0, std::atof(h));
    return (uint64_t) (head_gb * 1073741824.0);
}

// the lend's own rules over a passed room check (see the top): "" when the card may be lent, else why not
std::string lend_rules(const std::vector<ResplitPart>& parts, int dev, const ResplitRoom& r) {
    size_t at = 0;
    while (at < parts.size() && parts[at].dev != dev) ++at;
    if (at == 0) return "the first part is the model object the engine holds: it cannot be destroyed and loaded again";
    if (r.into == 0)
        return "its layers would go to the first part, the model object the engine holds, which cannot be loaded again";
    if (at + 1 == parts.size()) return "the last part holds the head, the sampler's buffers and the drafter: not lent whole";
    return "";
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

void Glm5Model::resplit_figures(std::vector<ResplitPart>& parts, std::vector<Glm5Model*>& chain) {
    parts.clear();
    chain.clear();
    // the split's figures, part by part (the VRAM line's, and the pool floor and RAM tier the rules add)
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        const FastState* F = m->fast_;
        ResplitPart p;
        p.dev = m->dev_;
        p.l0 = m->l0_;
        p.l1 = m->l1_;
        p.weights = m->w_arena_bytes_;
        p.kv = m->state_bytes_;
        p.act = m->sc_bytes_ + (F != nullptr ? F->arena_bytes + F->scratch_bytes : 0);
        p.drafter = m->mtp_arena_bytes_ + (m->dflash_ ? m->dflash_->bytes() : 0);
        if (F != nullptr) {
            p.window = F->win_bytes;
            p.pool = F->pool_bytes;
            p.ram = F->ram_bytes;
            // the floor: pool_keep slots a layer (one route's experts, the spares and 2 - what a lend keeps), in the
            // slot bytes of each MoE layer of the tier range (fast_setup's floor_b)
            size_t stride_sum = 0;
            for (int il = m->l0_; il < m->lt_ && (size_t) il < F->L.size(); ++il) {
                const auto& Ly = F->L[(size_t) il];
                if (Ly.moe) stride_sum += glmfast::expert_stride(Ly.blob, Ly.gu_type, Ly.d_type);
            }
            p.floor = (uint64_t) pool_keep(m->g_.n_exp_used) * stride_sum;
        }
        parts.push_back(p);
        chain.push_back(m);
    }
}

void Glm5Model::lend_whole_check(const std::vector<int>& devs) {
    if (devs.empty()) return;
    std::vector<ResplitPart> parts;
    std::vector<Glm5Model*> chain;
    resplit_figures(parts, chain);
    // the host's room: MemAvailable now and the headroom the RAM tier was sized against
    const uint64_t head_b = ram_headroom_bytes(fast_);
    const uint64_t avail = ram_avail_now();
    std::set<int> seen;
    for (int dev : devs) {
        if (!seen.insert(dev).second) continue;   // (named twice: checked once)
        if (fast_ == nullptr) {
            std::fprintf(stderr, "glm lend: CUDA%d is not lent whole: no expert pool (the slow path)\n", dev);
            continue;
        }
        const ResplitRoom r = resplit_room(parts, dev, avail, head_b);
        const std::string why = r.ok ? lend_rules(parts, dev, r) : r.reason;
        if (!why.empty()) {
            std::fprintf(stderr, "glm lend: CUDA%d is not lent whole: %s\n", dev, why.c_str());
            continue;
        }
        size_t at = 0;
        while (parts[at].dev != dev) ++at;   // (a pass: the device is in the split, once)
        chain[at]->whole_mib_ = r.whole_mib;
        const ResplitPart& m = parts[at];
        const ResplitPart& nb = parts[(size_t) r.into];
        std::fprintf(stderr, "glm lend: CUDA%d may be lent whole: %llu MiB (dense %llu, KV %llu, pool %llu MiB); its "
                             "layers [%d,%d) would move into CUDA%d's pool (%llu MiB above its floor), the merged RAM "
                             "tier %llu MiB\n", dev, (unsigned long long) r.whole_mib,
                     (unsigned long long) (m.weights >> 20), (unsigned long long) (m.kv >> 20),
                     (unsigned long long) (m.pool >> 20), m.l0, m.l1, nb.dev, (unsigned long long) (room_of(nb) >> 20),
                     (unsigned long long) ((nb.ram + m.ram) >> 20));
    }
}

uint64_t Glm5Model::whole_mib(int dev) const {
    for (const Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (m->dev_ == dev && m->whole_mib_ > 0) return m->whole_mib_;
    return 0;
}

bool Glm5Model::lend_whole_plan(int dev, WholePlan& plan, std::string& err) {
    plan = WholePlan{};
    const std::string card = "CUDA" + std::to_string(dev);
    if (whole_mib(dev) == 0) {
        err = card + " is not whole-lendable: the load's room check did not pass it";
        return false;
    }
    if (seq_count() > 1) {
        err = card + " is not lent whole: several conversations at once (STRATA_GLM_SEQS) are set up";
        return false;
    }
    std::vector<ResplitPart> parts;
    std::vector<Glm5Model*> chain;
    resplit_figures(parts, chain);
    const ResplitRoom r = resplit_room(parts, dev, ram_avail_now(), ram_headroom_bytes(fast_));
    const std::string why = r.ok ? lend_rules(parts, dev, r) : r.reason;
    if (!why.empty()) {
        err = card + " is not lent whole now: " + why;
        return false;
    }
    size_t at = 0;
    while (parts[at].dev != dev) ++at;
    plan.lent = chain[at];
    plan.into = chain[(size_t) r.into];
    plan.mib = chain[at]->whole_mib_;   // (the figure the LENDABLE lines announced)
    return true;
}

bool Glm5Model::lend_whole_run(const WholePlan& plan, std::string& err) {
    using clock = std::chrono::steady_clock;
    const auto t_all = clock::now();
    std::vector<Glm5Model*> chain;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) chain.push_back(m);
    const int old_n = (int) chain.size();
    int at = -1, into = -1;
    for (int i = 0; i < old_n; ++i) {
        if (chain[(size_t) i] == plan.lent) at = i;
        if (chain[(size_t) i] == plan.into) into = i;
    }
    if (at < 1 || into < 1 || at + 1 >= old_n || (into != at - 1 && into != at + 1)) {
        err = "glm lend: the plan does not fit the split";
        return false;
    }
    for (Glm5Model* m : chain)
        if (m->fast_ == nullptr) {
            err = "glm lend: CUDA" + std::to_string(m->dev_) + " has no expert pool (the slow path)";
            return false;
        }
    Glm5Model* L = plan.lent;
    Glm5Model* N = plan.into;
    const int ldev = L->dev_, ndev = N->dev_;
    const int lo = std::min(at, into);
    Glm5Model* prev = chain[(size_t) lo - 1];
    Glm5Model* tail = chain.back();
    int prev_dev = 0;
    cudaGetDevice(&prev_dev);
    struct DevBack {   // the caller's device again on every return
        int dev;
        ~DevBack() { cudaSetDevice(dev); }
    } dev_back{prev_dev};

    // ---- quiet: nothing runs, nothing moves, every stream of every card done
    ckpt_join();
    win_commit_wait();
    win_pipe_drain();
    win_crew_stop();
    struct Holds {   // the background hold off again on every survivor, whatever happens
        std::vector<Glm5Model*> parts;
        ~Holds() {
            for (Glm5Model* m : parts) m->fast_->bg_hold = false;
        }
    } holds;
    for (Glm5Model* m : chain) {
        FastState* F = m->fast_;
        cudaSetDevice(m->dev_);
        F->bg_hold = true;
        if (m != L && m != N) holds.parts.push_back(m);
        if (!glmfast::cuda_ok(cudaStreamSynchronize(F->copy), "glm lend copies", err) || !m->fast_boundary(err)) {
            L->fast_->bg_hold = N->fast_->bg_hold = false;
            return false;
        }
        if (!F->bg.empty() || !F->draining.empty()) {
            err = "glm lend: CUDA" + std::to_string(m->dev_) + " tier moves were not retired";
            L->fast_->bg_hold = N->fast_->bg_hold = false;
            return false;
        }
        if (!glmfast::cuda_ok(cudaDeviceSynchronize(), "glm lend: the card's streams", err)) {
            L->fast_->bg_hold = N->fast_->bg_hold = false;
            return false;
        }
    }
    const double t_quiet = ms_since(t_all);

    // ---- save: the conversation state to RAM (nothing to carry without a snapshot)
    const auto t1 = clock::now();
    std::vector<uint8_t> slot;
    int64_t n_pos = -1;
    bool have_state = snap_pos_ >= 0;
    for (Glm5Model* m : chain) have_state = have_state && m->snap_pos_ == snap_pos_;
    if (have_state) {
        std::string e;
        if (slot_save(slot, e) == 0) {
            err = "glm lend: the conversation state did not save: " + e;
            L->fast_->bg_hold = N->fast_->bg_hold = false;
            return false;
        }
        n_pos = snap_pos_;
    }
    const double t_save = ms_since(t1);

    // ---- what the reload needs, read before the parts go
    struct Spec {
        int dev, l0, l1, idx;
        int64_t ram;
    };
    const Spec spec_lo{chain[(size_t) lo]->dev_, chain[(size_t) lo]->l0_, chain[(size_t) lo]->l1_, lo,
                       (int64_t) chain[(size_t) lo]->fast_->ram_bytes};
    const Spec spec_hi{chain[(size_t) lo + 1]->dev_, chain[(size_t) lo + 1]->l0_, chain[(size_t) lo + 1]->l1_, lo + 1,
                       (int64_t) chain[(size_t) lo + 1]->fast_->ram_bytes};
    const Spec spec_merged{ndev, std::min(L->l0_, N->l0_), std::max(L->l1_, N->l1_), lo, spec_lo.ram + spec_hi.ram};
    std::vector<int> devs_old, devs_new;
    for (Glm5Model* m : chain) {
        devs_old.push_back(m->dev_);
        if (m != L) devs_new.push_back(m->dev_);
    }
    const std::string pack = pack_dir_, dpath = dflash_path_, ppath = profile_path_;
    const int64_t ctx = max_ctx_;
    const int n_layers = g_.n_layers;
    // an EXL3 head split whose helper dies: the last part's pick reads the whole head row from now on
    if (tail->head_helper_ == L || tail->head_helper_ == N) tail->head_helper_ = nullptr;

    // ---- destroy: the two parts out of the chain, the later one first
    const auto t2 = clock::now();
    std::unique_ptr<Glm5Model> first = std::move(prev->split_next_);
    std::unique_ptr<Glm5Model> second = std::move(first->split_next_);
    std::unique_ptr<Glm5Model> rest = std::move(second->split_next_);
    second.reset();
    first.reset();
    L = N = nullptr;
    const double t_destroy = ms_since(t2);

    // ---- load: one part through load_pack, as load_pack_split loads a part
    const auto make = [&](const Spec& sp, const std::vector<int>& devs) -> std::unique_ptr<Glm5Model> {
        std::unique_ptr<Glm5Model> p(new Glm5Model());
        p->n_parts_ = old_n;   // (the CPU lane's shared pool is sized from the count: the survivors' pool is joined)
        p->part_ = sp.idx;
        p->split_devs_ = devs;
        p->dflash_path_ = dpath;
        p->profile_path_ = ppath;
        p->ram_budget_ = sp.ram;
        p->defer_lane_ = true;   // (the lane and the service thread once the chain is linked, as load_pack_split)
        p->head_half_ = false;
        std::string e;
        if (!p->load_pack(pack, ctx, e, sp.dev, sp.l0, sp.l1 >= n_layers ? 0 : sp.l1)) {
            std::fprintf(stderr, "glm lend: CUDA%d layers [%d, %d) did not load: %s\n", sp.dev, sp.l0, sp.l1, e.c_str());
            return nullptr;
        }
        return p;
    };
    // the chain finished as load_pack_split finishes it, the fresh parts' lanes and service threads up
    const auto finish = [&](const std::vector<Glm5Model*>& fresh, std::string& e) -> bool {
        // the fresh parts' CPU lane and service thread first, still under the old part count: the lane's thread count
        // follows it, and the shared pool is the survivors' only while the count agrees
        for (Glm5Model* f : fresh) {
            f->defer_lane_ = false;
            if (!f->fast_setup_finish(e)) return false;
        }
        std::vector<Glm5Model*> now;
        std::vector<int> devs;
        for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
            now.push_back(m);
            devs.push_back(m->dev_);
        }
        for (size_t i = 0; i < now.size(); ++i) {
            Glm5Model* m = now[i];
            m->n_parts_ = (int) now.size();
            m->part_ = (int) i;
            m->split_devs_ = devs;
            m->calc_pf_state_ = 0;   // (the prompt path's pairings: decided again at the next prompt)
            m->dual_pf_state_ = 0;
            if (m->fast_ != nullptr) m->fast_->win.from = nullptr;   // (set again by the next pipelined window)
        }
        // the pipelined window's slot bank: every part flips its own at each window's begin and reads the previous
        // part's rows at the slot ITS bank names, so the parts must agree - a fresh part starts at 0 while the
        // survivors have flipped theirs as many times as windows ran (the first lend after an odd number of windows
        // read the other bank's stale rows: garbage from the merged part on)
        for (Glm5Model* f : fresh)
            if (f->fast_ != nullptr && fast_ != nullptr) f->fast_->win.bank = fast_->win.bank;
        int tmin = 0;
        for (Glm5Model* m : now)
            if (m->prefill_chunk() > 0) tmin = tmin == 0 ? m->prefill_chunk() : std::min(tmin, m->prefill_chunk());
        if (tmin > 0)
            for (Glm5Model* m : now) m->prefill_cap(tmin, "the split's smallest");
        if (!dflash_bind(e)) return false;
        for (Glm5Model* f : fresh)
            if (!f->fast_helper_check(e)) return false;
        cudaSetDevice(dev_);
        return true;
    };
    const auto fatal = [&](const std::string& what) {
        std::fprintf(stderr, "glm lend: FATAL: %s - layers [%d, %d) are gone and could not be loaded again: the engine "
                             "ends\n", what.c_str(), spec_merged.l0, spec_merged.l1);
        std::fflush(stderr);
        std::_Exit(1);
    };
    const auto t3 = clock::now();
    std::unique_ptr<Glm5Model> np = make(spec_merged, devs_new);
    if (np == nullptr) {
        // the split as it was: both parts loaded again (the lent card still has its context)
        std::fprintf(stderr, "glm lend: CUDA%d the merged part did not load: the split is rebuilt as it was\n", ndev);
        std::unique_ptr<Glm5Model> a = make(spec_lo, devs_old);
        std::unique_ptr<Glm5Model> b = a != nullptr ? make(spec_hi, devs_old) : nullptr;
        if (a == nullptr || b == nullptr) fatal("the rebuild failed too");
        Glm5Model* ap = a.get();
        Glm5Model* bp = b.get();
        b->split_next_ = std::move(rest);
        a->split_next_ = std::move(b);
        prev->split_next_ = std::move(a);
        std::string e;
        if (!finish({ap, bp}, e)) fatal("the rebuilt split did not finish: " + e);
        if (!have_state || !slot_load(slot, n_pos, e)) {
            reset();
            for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) m->snap_pos_ = -1;
        }
        err = "glm lend: CUDA" + std::to_string(ldev) + " the merged part did not load; the split is as it was";
        return false;
    }
    const double t_load = ms_since(t3);

    // ---- link
    const auto t4 = clock::now();
    Glm5Model* npp = np.get();
    np->split_next_ = std::move(rest);
    prev->split_next_ = std::move(np);
    {
        std::string e;
        if (!finish({npp}, e)) fatal("the new split did not finish: " + e);
    }
    const double t_link = ms_since(t4);

    // ---- restore: the state onto the new split
    const auto t5 = clock::now();
    if (have_state) {
        std::string e;
        if (!slot_load(slot, n_pos, e)) {
            std::fprintf(stderr, "glm lend: the conversation state did not load back onto the new split (%s): the next "
                                 "request reads its prompt again\n", e.c_str());
            reset();
            for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) m->snap_pos_ = -1;
        }
    } else {
        reset();
        for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) m->snap_pos_ = -1;
    }
    const double t_restore = ms_since(t5);

    // ---- release: the lent card's primary context, and with it the process's presence on the card
    const auto t6 = clock::now();
    cudaSetDevice(ldev);
    const cudaError_t re = cudaDeviceReset();
    cudaSetDevice(dev_);
    if (re != cudaSuccess)
        std::fprintf(stderr, "glm lend: CUDA%d cudaDeviceReset: %s (the process may still show on the card)\n", ldev,
                     cudaGetErrorString(re));
    const double t_release = ms_since(t6);

    std::string where;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        where += (m == this ? "[" : ", [") + std::to_string(m->l0_) + ", " + std::to_string(m->l1_) + ") on CUDA" +
                 std::to_string(m->dev_);
    const std::string state = have_state ? "the conversation state at " + std::to_string(n_pos) + " positions carried over"
                                         : "no conversation state to carry";
    std::fprintf(stderr, "glm lend: CUDA%d lent whole: CUDA%d now holds layers [%d, %d) (RAM tier %.2f GB); the split %s; "
                         "%s | quiet %.0f ms, save %.0f ms (%.1f MB), destroy %.0f ms, load %.0f ms, link %.0f ms, "
                         "restore %.0f ms, release %.0f ms, %.1f s in all\n",
                 ldev, ndev, spec_merged.l0, spec_merged.l1, (double) npp->fast_->ram_bytes / 1073741824.0, where.c_str(),
                 state.c_str(), t_quiet, t_save, (double) slot.size() / 1048576.0, t_destroy, t_load, t_link, t_restore,
                 t_release, ms_since(t_all) / 1000.0);
    return true;
}

}  // namespace strata::core
