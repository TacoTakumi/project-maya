// src/core/glm_resplit.cu - the room check of a whole-card lend: may a card of the split be lent WHOLE (its part
// destroyed, the card free for another program), with its layers taken in by a neighbour part?  Two rules, on the
// figures the load already knows (resplit_room, glm_model.hpp):
//   room    a neighbour part - the previous or the next in the chain, the one with the most pool room above its own
//           floor (pool_keep slots a layer), the previous on a tie - has room in its pool for the card's dense
//           weights + KV + pool floor: the pool's slots give way to the layers that move in, down to the floor the
//           lend rule keeps anyway (glm_lend.cu);
//   RAM     the lent part's RAM tier, which the neighbour pins anew for the merged tier before the lent part's own is
//           released, fits in the RAM free now under the headroom the tier sizing keeps free (fast_setup:
//           MemAvailable plus the ZFS ARC's reclaimable part, less STRATA_GLM_RAM_HEADROOM_GB, 6 GB on a discrete
//           card, 16 on unified memory).  A pass also sizes the merged tier (ResplitRoom::ram, bound), as a FIRST
//           load would size a part of the merged range: the free RAM under the headroom, bounded by what the merged
//           layers can use - every routed expert of theirs outside the merged pool's main slots plus the spares a
//           layer (the load's own per-class cap, fast_setup's cls_cap; the merged pool keeps the neighbour's main
//           slots less the lent part's dense weights and KV, which move in) - and never below the two tiers' sum,
//           which is what the free RAM unknown gives.  The sum alone left the merged part short: on five cards the
//           merged part held 99-132 slots a layer in VRAM with the sum's 18 GB tier, read about 5 experts a token
//           from disk and decoded at half the speed, while some 57 GB of host RAM sat free.
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
//   save     the conversation state to RAM (resplit_state_save): Glm5Model::slot_save's bytes (the snapshot's
//            recurrent states, the open indexer pools and the DSA caches up to the snapshot's position, one section a
//            part).  A decode stopped at a token for the lend (the serve loop's lend stop) stands PAST the snapshot:
//            its live state - the KDA states, the open pool rows and the DSA caches up to the stop position, the same
//            on every part - goes into a second slot, so the decode goes on from where it stopped instead of reading
//            its tokens again.  No snapshot (no request yet): nothing to carry, the chain is reset after the rebuild.
//            Several sequences (STRATA_GLM_SEQS): every part is bound to the head's sequence first and that one's
//            state is what crosses; the others' states die with the destroyed parts (the serve loop forgets their
//            prefix caches and resets them).  The command loop runs a whole-card change only idle or at the one
//            decode's token stop, so the live state is never a half-read prompt or several decodes;
//   destroy  the lent part and the neighbour, unlinked from the chain first (a part's destructor takes the chain
//            after it along), the later one first: each part frees its device and pinned memory under its own
//            device, as at exit;
//   load     one part with the merged layer range on the neighbour's device, through load_pack as load_pack_split
//            loads a part - the same pack, context, drafter path and expert profile; its RAM tier budget is the
//            plan's merged tier (WholePlan::ram, the room check's figure above; ram_budget_: the tier is not derived
//            again from the free RAM, which the destroyed parts' tiers have just swollen - the load still caps each
//            slot class at what its layers can use, so the tier pinned may come out a little under the budget), the
//            pool carved from the card's free VRAM as at the load, the warm-up as at the load.  The part
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
//
// THE RETURN (lend_whole_room, lend_whole_return; the lender calls them from give_back on a whole-card token).  The
// plan the lend ran with carries the two parts as they were - device, layers, pinned RAM tier each - and the load's
// part count (WholePlan::lo / hi / n_parts), so the reload is the lend's load in reverse:
//   room     before anything changes: cudaMemGetInfo under the lent card (its primary context comes back for it)
//            against the footprint the part had (plan.mib: the pool as it was mapped, the dense weights, KV, the
//            rest).  Short of it another process still holds the card: one log line with the figures, the context
//            released again, busy - the lease stays open and the caller tries later;
//   quiet, save    as the lend's;
//   destroy  the merged part, unlinked;
//   load     the two parts through load_pack as before, the earlier first, each with its own device, layer range and
//            RAM tier budget (ram_budget_: the tiers come back as they were), under the load's part count (the shared
//            CPU lane pool is joined, not doubled); the lent card's context comes back with the first allocation.  A
//            part that did not load: the merged part is loaded again and finished, the state put back, the context
//            released, busy (the lease stays open);
//   link, restore  as the lend's, for the two fresh parts;
//   check    the room check as at the load (lend_whole_recheck) for the two reloaded cards --lend-whole named - the lent
//            card, and the neighbour when it is named too: both parts are fresh models without a footprint - so the
//            footprint is announced again (whole_mib(dev), LENDABLE whole=) and each may be lent whole once more.  The
//            rebuild after a failed lend runs the same check.  resplit_recheck_devs picks the cards: of the reloaded
//            parts' devices, those lend_whole_check was given (it keeps the list), a card not named never checked.
// The pools are carved from each card's free VRAM as at the load, so the parts come back with the layouts the load
// gave them; their tiers warm again from the page cache as the decode routes.
// The steps the two share - quiet, one part's load, the chain's finish - are resplit_quiet / resplit_make /
// resplit_finish.
#include "glm_fast_state.hpp"

#include "strata/core/glm_model.hpp"
#include "strata/kernels/glm_fast.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

const char* ram_bound_name(RamBound b) {
    switch (b) {
        case RamBound::kFree: return "the free RAM under the headroom";
        case RamBound::kLayers: return "the layers' bound";
        default: return "the two tiers' sum";
    }
}

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
    // the merged RAM tier: the two tiers' sum at least (the neighbour's is released and pinned again, the lent part's
    // pinned anew - the RAM rule below holds the lent part's in the free RAM), and more when the free RAM allows it
    const uint64_t sum = nb.ram + m.ram;
    r.ram = sum;
    r.bound = RamBound::kSum;
    if (ram_avail > 0) {
        const uint64_t free_room = ram_avail > ram_headroom ? ram_avail - ram_headroom : 0;
        if (m.ram > free_room) {
            r.ram = 0;
            r.reason = "RAM: " + card + "'s tier " + mib(m.ram) + " MiB, pinned anew by CUDA" + std::to_string(nb.dev) +
                       " for the merged tier, is over the " + mib(free_room) + " MiB free under the headroom " +
                       mib(ram_headroom) + " MiB (" + mib(ram_avail) + " MiB free): short " + mib(m.ram - free_room) +
                       " MiB";
            return r;
        }
        // what a first load would give a part of the merged size: the free RAM under the headroom, bounded by what the
        // merged layers can use - every routed expert of theirs outside the merged pool's main slots, plus the spares a
        // layer the tier sizing adds (the load caps each slot class the same way: fast_setup's cls_cap).  The merged
        // pool keeps the neighbour's main slots less the lent part's dense weights and KV, which move in
        const uint64_t moved = m.weights + m.kv;
        const uint64_t held = nb.held > moved ? nb.held - moved : 0;
        const uint64_t all = nb.experts + m.experts + nb.spares + m.spares;
        const uint64_t cap = all > held ? all - held : 0;
        if (free_room > sum && cap > sum) {
            r.ram = std::min(free_room, cap);
            r.bound = free_room >= cap ? RamBound::kLayers : RamBound::kFree;
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
                if (!Ly.moe) continue;
                stride_sum += glmfast::expert_stride(Ly.blob, Ly.gu_type, Ly.d_type);
                // the RAM tier's bound for the layer, at the tier's slot stride (fast_setup's class stride): every
                // routed expert, the spares a layer the tier sizing adds (kSpares + 2 + the slack), and the experts the
                // pool's main slots hold - the tail is borrowed by a prompt, so it counts as not held, as the sizing has it
                const uint64_t st = (Ly.blob + 4095u) & ~(uint64_t) 4095u;
                p.experts += (uint64_t) m->g_.n_expert * st;
                p.spares += (uint64_t) (gf::kSpares + 2 + F->ram_slack) * st;
                if ((size_t) il < F->lp.size())
                    p.held += (uint64_t) std::max(0, std::min(m->g_.n_expert, F->lp[(size_t) il].n_main)) * st;
            }
            p.floor = (uint64_t) pool_keep(m->g_.n_exp_used) * stride_sum;
        }
        parts.push_back(p);
        chain.push_back(m);
    }
}

std::vector<int> resplit_recheck_devs(const std::vector<int>& reloaded, const std::vector<int>& named) {
    std::vector<int> out;
    for (int dev : reloaded) {
        if (std::find(named.begin(), named.end(), dev) == named.end()) continue;
        if (std::find(out.begin(), out.end(), dev) != out.end()) continue;
        out.push_back(dev);
    }
    return out;
}

void Glm5Model::lend_whole_check(const std::vector<int>& devs) {
    lend_whole_devs_ = devs;   // (kept for the re-split reloads: they check the reloaded cards of this list again)
    lend_whole_recheck(devs);
}

void Glm5Model::lend_whole_recheck(const std::vector<int>& devs) {
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
                             "tier the lend pins %llu MiB (%s; the two tiers' sum %llu MiB, %llu MiB free under the "
                             "headroom)\n", dev, (unsigned long long) r.whole_mib,
                     (unsigned long long) (m.weights >> 20), (unsigned long long) (m.kv >> 20),
                     (unsigned long long) (m.pool >> 20), m.l0, m.l1, nb.dev, (unsigned long long) (room_of(nb) >> 20),
                     (unsigned long long) (r.ram >> 20), ram_bound_name(r.bound),
                     (unsigned long long) ((nb.ram + m.ram) >> 20),
                     (unsigned long long) ((avail > head_b ? avail - head_b : 0) >> 20));
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
    // (several sequences, STRATA_GLM_SEQS: no refusal here - the command loop runs the lend once every request has
    // ended, and lend_whole_run carries the bound sequence's state)
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
    plan.dev = dev;
    plan.ram = (int64_t) r.ram;         // the merged part's tier budget, sized on the RAM free now
    plan.ram_bound = r.bound;
    // the two parts as they are, in chain order: what the return loads again
    const size_t lo = std::min(at, (size_t) r.into);
    for (size_t i = 0; i < 2; ++i) {
        const Glm5Model* m = chain[lo + i];
        WholePlan::Part& p = i == 0 ? plan.lo : plan.hi;
        p.dev = m->dev_;
        p.l0 = m->l0_;
        p.l1 = m->l1_;
        p.ram = m->fast_ != nullptr ? (int64_t) m->fast_->ram_bytes : -1;
        // the pool as it is: the budget it was carved from and its block (the return carves the same pool again)
        p.pool = m->pool_b_.avail;
        p.block = m->fast_ != nullptr && m->fast_->vpool.valid() ? m->fast_->vpool.block_bytes() : 0;
    }
    plan.n_parts = (int) chain.size();
    return true;
}

// ---- the steps the lend and the return share

bool Glm5Model::resplit_quiet(const std::vector<Glm5Model*>& chain, std::string& err) {
    ckpt_join();
    win_commit_wait();
    win_pipe_drain();
    win_crew_stop();
    for (Glm5Model* m : chain) {
        FastState* F = m->fast_;
        cudaSetDevice(m->dev_);
        F->bg_hold = true;
        if (!glmfast::cuda_ok(cudaStreamSynchronize(F->copy), "glm lend copies", err) || !m->fast_boundary(err))
            return false;
        if (!F->bg.empty() || !F->draining.empty()) {
            err = "glm lend: CUDA" + std::to_string(m->dev_) + " tier moves were not retired";
            return false;
        }
        if (!glmfast::cuda_ok(cudaDeviceSynchronize(), "glm lend: the card's streams", err)) return false;
    }
    return true;
}

std::unique_ptr<Glm5Model> Glm5Model::resplit_make(const WholePlan::Part& sp, int idx, int n_parts,
                                                   const std::vector<int>& devs) {
    // (the pack, context, drafter and profile are this part's - the head, which no re-split destroys)
    std::unique_ptr<Glm5Model> p(new Glm5Model());
    p->n_parts_ = n_parts;   // (the CPU lane's shared pool is sized from the count: the survivors' pool is joined)
    p->part_ = idx;
    p->split_devs_ = devs;
    p->dflash_path_ = dflash_path_;
    p->profile_path_ = profile_path_;
    p->ram_budget_ = sp.ram;
    p->pool_want_ = (size_t) sp.pool;         // (the pool as the part had it; 0: from the free VRAM, as at a load)
    p->pool_block_want_ = (size_t) sp.block;
    p->defer_lane_ = true;   // (the lane and the service thread once the chain is linked, as load_pack_split)
    p->head_half_ = false;
    std::string e;
    if (!p->load_pack(pack_dir_, max_ctx_, e, sp.dev, sp.l0, sp.l1 >= g_.n_layers ? 0 : sp.l1)) {
        std::fprintf(stderr, "glm lend: CUDA%d layers [%d, %d) did not load: %s\n", sp.dev, sp.l0, sp.l1, e.c_str());
        return nullptr;
    }
    return p;
}

bool Glm5Model::resplit_finish(const std::vector<Glm5Model*>& fresh, std::string& e) {
    // the fresh parts' CPU lane and service thread first, still under the load's part count: the lane's thread count
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
    // several sequences (STRATA_GLM_SEQS): a fresh part comes up bound to its sequence 0; every path reads the bound
    // sequence on every part, so it joins the one the head is bound to (the lend bound every survivor to it)
    for (Glm5Model* f : fresh) f->seq_bind(seq_bound());
    // ... and the pipelined decode's per-sequence buffers (the hop rows and events a token crosses the parts with,
    // seq_pipe_ready: set up once, on the head's first step) on the fresh parts too - the first step after the first
    // lend under --parallel copied the hop through the merged part's null row (an invalid argument)
    if (!seq_.empty() && seq_[0].hop_h != nullptr && !seq_pipe_ready(e)) return false;
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
}

// ---- the conversation state a re-split carries (ResplitState, glm_model.hpp)
// The snapshot's slot (slot_save: its KDA states, its open pool rows and every DSA cache up to its position) is what an
// idle lend carries, and slot_load puts it back as the live state and the snapshot.  A decode stopped at a token for
// the lend stands PAST the snapshot: its KDA states, its open pool rows and its DSA caches up to the stop position are
// the live state the decode goes on from (the serve loop's lend_resume reads the one token still unread), so a second
// slot of the live state is taken too - snapshot_save makes the live state the snapshot for the moment, slot_save
// writes it, and the snapshot's own records come back from the first slot's bytes (snap_from_slot).  The restore loads
// the live slot (the live state and, for the moment, the snapshot) and then puts the snapshot's records back under it:
// the live state stays, the snapshot is the prompt's again, and the next turn's prefix reuse finds it.

// the snapshot's records - each part's KDA states (snap_) and open pool rows (snap_pool_) - from a slot of n_pos
// positions, whatever split saved it (Glm5SlotLayout maps its sections onto the parts as they stand); the live state,
// the position and the DSA caches are not touched
bool Glm5Model::snap_from_slot(const std::vector<uint8_t>& slot, int64_t n_pos, std::string& err) {
    ckpt_join();   // (ckpt_save_snapshot reads snap_ / snap_pool_)
    const Glm5SlotLayout::Read read = [&](uint64_t at, void* p, size_t bytes) {
        if (at + bytes > slot.size()) return false;
        std::memcpy(p, slot.data() + at, bytes);
        return true;
    };
    // the record sizes as the slot writer has them (glm_fast_path.cu's slot_sizes)
    Glm5SlotLayout::Sizes sz;
    sz.run = (int64_t) g_.d_inner() * g_.kda_head_dim + (int64_t) 3 * g_.d_inner() * (g_.d_conv - 1);
    sz.pool_rec = (int64_t) 2 * (g_.idx_kpool - 1) * g_.idx_key;
    sz.lat = n_pos * (lat_q8_ ? (int64_t) gf::lat8_rec_bytes(g_.kv_lora) / 4 : fast_mode_ ? g_.kv_lora / 2 : g_.kv_lora);
    sz.pool = ((n_pos + g_.idx_kpool - 1) / g_.idx_kpool) * g_.idx_key;
    std::vector<Glm5SlotLayout::Section> secs;
    int64_t n = 0;
    if (!Glm5SlotLayout::parse(read, slot.size(), sz.pool_rec, secs, n, err)) return false;
    if (n != n_pos) {
        err = "the slot holds " + std::to_string(n) + " positions, not " + std::to_string(n_pos);
        return false;
    }
    std::vector<Glm5Model*> halves;
    std::vector<Glm5SlotLayout::Part> parts;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        halves.push_back(m);
        parts.push_back({m->l0_, m->l1_, m->mtp_il_ >= 0});
    }
    std::vector<bool> recr((size_t) g_.n_layers);
    for (int il = 0; il < g_.n_layers; ++il) recr[(size_t) il] = g_.is_recr(il);
    std::vector<std::vector<Glm5SlotLayout::Piece>> pieces;
    if (!Glm5SlotLayout::map(secs, parts, recr, sz, pieces, err)) return false;
    for (size_t h = 0; h < halves.size(); ++h) {
        Glm5Model* m = halves[h];
        if (m->fast_ == nullptr) {
            err = "no fast path";
            return false;
        }
        int32_t n_rec = 0;
        for (int il = m->l0_; il < m->l1_; ++il) n_rec += m->g_.is_recr(il);
        const size_t n_dsa = m->dsa_layers().size();
        const std::vector<Glm5SlotLayout::Piece>& src = pieces[h];
        if (src.size() != (size_t) n_rec + 3 * n_dsa) {
            err = "the slot's layout differs from this model's";
            return false;
        }
        cudaSetDevice(m->dev_);
        cudaStreamSynchronize(m->fast_->cs);
        if (m->snap_ == nullptr && n_rec > 0 &&
            cudaMalloc(&m->snap_, (size_t) n_rec * (size_t) sz.run * sizeof(float)) != cudaSuccess) {
            cudaGetLastError();
            m->snap_ = nullptr;
            cudaSetDevice(dev_);
            err = "the snapshot did not allocate";
            return false;
        }
        if (n_dsa > 0 && m->snap_pool_ == nullptr &&
            cudaMalloc(&m->snap_pool_, n_dsa * (size_t) sz.pool_rec * sizeof(float)) != cudaSuccess) {
            cudaGetLastError();
            m->snap_pool_ = nullptr;
            cudaSetDevice(dev_);
            err = "the snapshot did not allocate";
            return false;
        }
        // the mapping's order: a KDA record a recurrent layer, then a closed-pool record a DSA layer (the runs follow,
        // not wanted here)
        for (size_t k = 0; k < (size_t) n_rec + n_dsa; ++k) {
            const bool kda = k < (size_t) n_rec;
            float* dst = kda ? m->snap_ + (int64_t) k * sz.run : m->snap_pool_ + (int64_t) (k - (size_t) n_rec) * sz.pool_rec;
            const int64_t floats = kda ? sz.run : sz.pool_rec;
            if (src[k].floats != floats || src[k].at + (uint64_t) floats * sizeof(float) > slot.size()) {
                cudaSetDevice(dev_);
                err = "the slot's layout differs from this model's";
                return false;
            }
            if (cudaMemcpy(dst, slot.data() + src[k].at, (size_t) floats * sizeof(float), cudaMemcpyHostToDevice) !=
                cudaSuccess) {
                cudaGetLastError();
                cudaSetDevice(dev_);
                err = "the snapshot did not read back";
                return false;
            }
        }
        m->snap_pos_ = n;
    }
    cudaSetDevice(dev_);
    return true;
}

bool Glm5Model::resplit_state_save(ResplitState& st, std::string& err) {
    st = ResplitState{};
    bool have = snap_pos_ >= 0;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        have = have && m->fast_ != nullptr && m->snap_pos_ == snap_pos_;
    if (!have) return true;   // (nothing to carry: the chain is reset after the rebuild)
    std::string e;
    if (slot_save(st.snap, e) == 0) {
        err = "glm lend: the conversation state did not save: " + e;
        return false;
    }
    st.snap_pos = snap_pos_;
    // the live state: a decode stopped at a token past the snapshot, every part at that position
    bool live = pos_ > snap_pos_;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) live = live && m->pos_ == pos_;
    if (!live) return true;
    const int64_t p = pos_;
    const auto snap_back = [&](const char* when) -> bool {   // the snapshot as it was, from its slot
        std::string e2;
        if (snap_from_slot(st.snap, st.snap_pos, e2)) return true;
        // (the snapshot is the live state's now and belongs to no prompt: dropped, the next turn reads its prompt)
        for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) m->snap_pos_ = -1;
        err = std::string("glm lend: the snapshot did not come back ") + when + ": " + e2;
        return false;
    };
    if (!snapshot_save()) {
        if (!snap_back("after the live state's snapshot failed")) return false;
        std::fprintf(stderr, "glm lend: the decode's live state at %lld positions did not snapshot: the decode reads "
                             "its tokens again after the re-split\n", (long long) p);
        return true;
    }
    const uint64_t n = slot_save(st.live, e);
    if (!snap_back("after the live state's save")) return false;
    if (n == 0) {
        std::fprintf(stderr, "glm lend: the decode's live state at %lld positions did not save (%s): the decode reads "
                             "its tokens again after the re-split\n", (long long) p, e.c_str());
        st.live.clear();
        return true;
    }
    st.live_pos = p;
    return true;
}

bool Glm5Model::resplit_state_load(const ResplitState& st, std::string& err) {
    if (st.snap_pos < 0) {
        err = "nothing saved";
        return false;
    }
    std::string e;
    if (st.live_pos < 0) {
        if (!slot_load(st.snap, st.snap_pos, e)) {
            err = "the conversation state did not load back: " + e;
            return false;
        }
        return true;
    }
    // the live state first (slot_load makes it the snapshot too, and sets the position), then the snapshot's own
    // records under it
    if (!slot_load(st.live, st.live_pos, e)) {
        err = "the decode's live state did not load back: " + e;
        return false;
    }
    if (!snap_from_slot(st.snap, st.snap_pos, e)) {
        err = "the snapshot did not load back under the live state: " + e;
        return false;
    }
    return true;
}

namespace {

// the chain from the head, and the layout line the log prints for it
std::vector<Glm5Model*> chain_of(Glm5Model* head) {
    std::vector<Glm5Model*> chain;
    for (Glm5Model* m = head; m != nullptr; m = m->split_next_.get()) chain.push_back(m);
    return chain;
}

// the background hold off again on every part listed, whatever happens (a part about to be destroyed leaves the list)
struct Holds {
    std::vector<Glm5Model*> parts;
    void drop(const Glm5Model* m) { parts.erase(std::remove(parts.begin(), parts.end(), m), parts.end()); }
    ~Holds() {
        for (Glm5Model* m : parts) m->fast_->bg_hold = false;
    }
};

// the process off a card: its primary context released (nothing of this process is allocated there any more)
void leave_card(int dev, int back_to) {
    cudaSetDevice(dev);
    const cudaError_t re = cudaDeviceReset();
    // the kernels' per-device caches of what belonged to the context (scratch pointers, shared-memory opt-ins, a
    // constant upload) are stale now: the first decode on the card after its return read a freed pointer
    gf::device_context_reset(dev);
    cudaSetDevice(back_to);
    if (re != cudaSuccess)
        std::fprintf(stderr, "glm lend: CUDA%d cudaDeviceReset: %s (the process may still show on the card)\n", dev,
                     cudaGetErrorString(re));
}

}  // namespace

namespace {

// the log line's words for the state a re-split carried
std::string resplit_state_words(const Glm5Model::ResplitState& st) {
    if (st.snap_pos < 0) return "no conversation state to carry";
    std::string w = "the conversation state at " + std::to_string(st.snap_pos) + " positions carried over";
    if (st.live_pos >= 0) w += ", and the decode's live state at " + std::to_string(st.live_pos) + " positions";
    return w;
}

}  // namespace

std::string Glm5Model::resplit_where() const {
    std::string where;
    for (const Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        where += (m == this ? "layers [" : ", [") + std::to_string(m->l0_) + ", " + std::to_string(m->l1_) + ") on CUDA" +
                 std::to_string(m->dev_);
    return where;
}

bool Glm5Model::lend_whole_run(const WholePlan& plan, std::string& err) {
    using clock = std::chrono::steady_clock;
    const auto t_all = clock::now();
    std::vector<Glm5Model*> chain = chain_of(this);
    const int old_n = (int) chain.size();
    int at = -1, into = -1;
    for (int i = 0; i < old_n; ++i) {
        if (chain[(size_t) i] == plan.lent) at = i;
        if (chain[(size_t) i] == plan.into) into = i;
    }
    const int lo = std::min(at, into);
    if (at < 1 || into < 1 || at + 1 >= old_n || (into != at - 1 && into != at + 1) || old_n != plan.n_parts ||
        plan.lo.dev != chain[(size_t) lo]->dev_ || plan.hi.dev != chain[(size_t) lo + 1]->dev_ ||
        plan.dev != plan.lent->dev_) {
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
    Glm5Model* prev = chain[(size_t) lo - 1];
    Glm5Model* tail = chain.back();
    int prev_dev = 0;
    cudaGetDevice(&prev_dev);
    struct DevBack {   // the caller's device again on every return
        int dev;
        ~DevBack() { cudaSetDevice(dev); }
    } dev_back{prev_dev};

    // ---- quiet: nothing runs, nothing moves, every stream of every card done
    Holds holds{chain};
    if (!resplit_quiet(chain, err)) return false;
    const double t_quiet = ms_since(t_all);

    // ---- save: the conversation state to RAM (nothing to carry without a snapshot; the live state too when a
    //      decode stopped here).  Several sequences: every part bound to the head's, whose state crosses
    const auto t1 = clock::now();
    if (seq_count() > 1) seq_bind_all(seq_bound());
    ResplitState st;
    if (!resplit_state_save(st, err)) return false;
    const bool have_state = st.snap_pos >= 0;
    const double t_save = ms_since(t1);

    // ---- what the reload needs, read before the parts go (the merged tier's budget: the plan's figure, sized on the
    //      RAM free at plan time under the headroom and bounded by the merged layers; the load caps each slot class
    //      at what its layers can use on top of it, so the tier pinned may come out a little under the budget)
    const WholePlan::Part merged{ndev, std::min(L->l0_, N->l0_), std::max(L->l1_, N->l1_),
                                 plan.ram >= 0 ? plan.ram : plan.lo.ram + plan.hi.ram};
    std::vector<int> devs_old, devs_new;
    for (Glm5Model* m : chain) {
        devs_old.push_back(m->dev_);
        if (m != L) devs_new.push_back(m->dev_);
    }
    // an EXL3 head split whose helper dies: the last part's pick reads the whole head row from now on
    if (tail->head_helper_ == L || tail->head_helper_ == N) tail->head_helper_ = nullptr;

    // ---- destroy: the two parts out of the chain, the later one first
    const auto t2 = clock::now();
    holds.drop(L);
    holds.drop(N);
    std::unique_ptr<Glm5Model> first = std::move(prev->split_next_);
    std::unique_ptr<Glm5Model> second = std::move(first->split_next_);
    std::unique_ptr<Glm5Model> rest = std::move(second->split_next_);
    second.reset();
    first.reset();
    L = N = nullptr;
    const double t_destroy = ms_since(t2);

    // ---- load: one part through load_pack, as load_pack_split loads a part
    const auto fatal = [&](const std::string& what) {
        std::fprintf(stderr, "glm lend: FATAL: %s - layers [%d, %d) are gone and could not be loaded again: the engine "
                             "ends\n", what.c_str(), merged.l0, merged.l1);
        std::fflush(stderr);
        std::_Exit(1);
    };
    const auto t3 = clock::now();
    std::unique_ptr<Glm5Model> np;
    // a TEST FAULT, STRATA_GLM_RESPLIT_FAIL_LOAD=1: the first merged load of the process fails right here, before
    // resplit_make - as early as a real failure could, the two parts gone - so the rebuild below runs; it is spent
    // after that one use, the next lend loads as always
    static bool fail_load_spent = false;
    const char* fail_load = std::getenv("STRATA_GLM_RESPLIT_FAIL_LOAD");
    if (!fail_load_spent && fail_load != nullptr && std::atoi(fail_load) != 0) {
        fail_load_spent = true;
        std::fprintf(stderr, "glm lend: TEST FAULT (STRATA_GLM_RESPLIT_FAIL_LOAD): the merged load of layers [%d, %d) on "
                             "CUDA%d is made to fail, once\n", merged.l0, merged.l1, ndev);
    } else {
        np = resplit_make(merged, lo, old_n, devs_new);
    }
    if (np == nullptr) {
        // the split as it was: both parts loaded again (the lent card still has its context)
        std::fprintf(stderr, "glm lend: CUDA%d the merged part did not load: the split is rebuilt as it was\n", ndev);
        std::unique_ptr<Glm5Model> a = resplit_make(plan.lo, lo, old_n, devs_old);
        std::unique_ptr<Glm5Model> b = a != nullptr ? resplit_make(plan.hi, lo + 1, old_n, devs_old) : nullptr;
        if (a == nullptr || b == nullptr) fatal("the rebuild failed too");
        Glm5Model* ap = a.get();
        Glm5Model* bp = b.get();
        b->split_next_ = std::move(rest);
        a->split_next_ = std::move(b);
        prev->split_next_ = std::move(a);
        std::string e;
        if (!resplit_finish({ap, bp}, e)) fatal("the rebuilt split did not finish: " + e);
        const bool restored = have_state && resplit_state_load(st, e);
        if (!restored) {
            reset();
            for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) m->snap_pos_ = -1;
        }
        // the load's room check again for the rebuilt cards the operator named, as the return runs it: both rebuilt
        // parts are fresh models (whole_mib_ 0), and without the check the lent card - and a neighbour named too -
        // would stay un-lendable whole until a restart
        lend_whole_recheck(resplit_recheck_devs({plan.lo.dev, plan.hi.dev}, lend_whole_devs_));
        std::fprintf(stderr, "glm lend: CUDA%d rebuilt after the failed lend: whole-lendable %s; the split %s; %s\n", ldev,
                     whole_mib(ldev) > 0 ? "again" : "NOT (see above)", resplit_where().c_str() + 7,
                     restored ? resplit_state_words(st).c_str()
                              : "no conversation state carried (the next request reads its prompt again)");
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
        if (!resplit_finish({npp}, e)) fatal("the new split did not finish: " + e);
    }
    const double t_link = ms_since(t4);

    // ---- restore: the state onto the new split
    const auto t5 = clock::now();
    if (have_state) {
        std::string e;
        if (!resplit_state_load(st, e)) {
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
    leave_card(ldev, dev_);
    const double t_release = ms_since(t6);

    const std::string state = resplit_state_words(st);
    std::fprintf(stderr, "glm lend: CUDA%d lent whole: CUDA%d now holds layers [%d, %d) (RAM tier %.2f GB pinned of a "
                         "%.2f GB budget, %s; the two tiers' sum %.2f GB); the split %s; %s | quiet %.0f ms, save %.0f ms "
                         "(%.1f MB), destroy %.0f ms, load %.0f ms, link %.0f ms, restore %.0f ms, release %.0f ms, %.1f s "
                         "in all\n",
                 ldev, ndev, merged.l0, merged.l1, (double) npp->fast_->ram_bytes / 1073741824.0,
                 (double) merged.ram / 1073741824.0, ram_bound_name(plan.ram_bound),
                 (double) (plan.lo.ram + plan.hi.ram) / 1073741824.0, resplit_where().c_str() + 7, state.c_str(),
                 t_quiet, t_save, (double) (st.snap.size() + st.live.size()) / 1048576.0, t_destroy, t_load, t_link,
                 t_restore, t_release,
                 ms_since(t_all) / 1000.0);
    return true;
}

bool Glm5Model::lend_whole_room(const WholePlan& plan, std::string& why) {
    int prev_dev = 0;
    cudaGetDevice(&prev_dev);
    cudaSetDevice(plan.dev);
    size_t free_b = 0, total_b = 0;
    const cudaError_t e = cudaMemGetInfo(&free_b, &total_b);   // (the primary context comes back for this)
    const std::string card = "CUDA" + std::to_string(plan.dev);
    if (e != cudaSuccess) {
        cudaGetLastError();
        why = card + " cudaMemGetInfo: " + cudaGetErrorString(e);
        leave_card(plan.dev, prev_dev);
        return false;
    }
    const uint64_t need = plan.mib << 20;
    if ((uint64_t) free_b < need) {
        // the context is in the card's used figure now; what is used beside it is another process's
        why = card + " has " + mib(free_b) + " MiB free of " + mib(total_b) + " with this process's context up; the "
              "part needs its footprint of " + mib(need) + " MiB back (short " + mib(need - free_b) + " MiB): another "
              "process still holds the card";
        leave_card(plan.dev, prev_dev);   // (the card as it was: the lease stays open, the process leaves again)
        return false;
    }
    cudaSetDevice(prev_dev);
    return true;
}

bool Glm5Model::lend_whole_return(const WholePlan& plan, std::string& err) {
    using clock = std::chrono::steady_clock;
    const auto t_all = clock::now();
    const int ldev = plan.dev;
    const int ndev = plan.lo.dev == ldev ? plan.hi.dev : plan.lo.dev;
    // (the merged part as the lend loaded it, should a part not load again and the merged part be rebuilt)
    const WholePlan::Part merged{ndev, std::min(plan.lo.l0, plan.hi.l0), std::max(plan.lo.l1, plan.hi.l1),
                                 plan.ram >= 0 ? plan.ram : plan.lo.ram + plan.hi.ram};
    std::vector<Glm5Model*> chain = chain_of(this);
    int mi = -1;
    for (size_t i = 0; i < chain.size(); ++i)
        if (chain[i]->dev_ == ndev && chain[i]->l0_ == merged.l0 && chain[i]->l1_ == merged.l1) mi = (int) i;
    int prev_dev = 0;
    cudaGetDevice(&prev_dev);
    struct DevBack {
        int dev;
        ~DevBack() { cudaSetDevice(dev); }
    } dev_back{prev_dev};
    if (mi < 1 || plan.lo.dev == plan.hi.dev || (plan.lo.dev != ldev && plan.hi.dev != ldev) ||
        plan.n_parts != (int) chain.size() + 1) {
        err = "glm lend: the return's plan does not fit the split (no merged part [" + std::to_string(merged.l0) + ", " +
              std::to_string(merged.l1) + ") on CUDA" + std::to_string(ndev) + ")";
        leave_card(ldev, dev_);
        return false;
    }
    for (Glm5Model* m : chain) {
        if (m->fast_ == nullptr) {
            err = "glm lend: CUDA" + std::to_string(m->dev_) + " has no expert pool (the slow path)";
            leave_card(ldev, dev_);
            return false;
        }
        if (m->dev_ == ldev) {
            err = "glm lend: CUDA" + std::to_string(ldev) + " holds a part of the split already";
            return false;
        }
    }
    Glm5Model* M = chain[(size_t) mi];
    Glm5Model* prev = chain[(size_t) mi - 1];
    Glm5Model* tail = chain.back();

    // ---- quiet
    Holds holds{chain};
    if (!resplit_quiet(chain, err)) {
        leave_card(ldev, dev_);
        return false;
    }
    const double t_quiet = ms_since(t_all);

    // ---- save (as the lend's)
    const auto t1 = clock::now();
    if (seq_count() > 1) seq_bind_all(seq_bound());
    ResplitState st;
    if (!resplit_state_save(st, err)) {
        leave_card(ldev, dev_);
        return false;
    }
    const bool have_state = st.snap_pos >= 0;
    const double t_save = ms_since(t1);

    // the device lists: the split as it is (the rebuild's) and as it was (the two parts')
    std::vector<int> devs_now, devs_was;
    for (Glm5Model* m : chain) {
        devs_now.push_back(m->dev_);
        if (m == M) {
            devs_was.push_back(plan.lo.dev);
            devs_was.push_back(plan.hi.dev);
        } else {
            devs_was.push_back(m->dev_);
        }
    }
    if (tail->head_helper_ == M) tail->head_helper_ = nullptr;

    // ---- destroy: the merged part out of the chain
    const auto t2 = clock::now();
    holds.drop(M);
    std::unique_ptr<Glm5Model> mp = std::move(prev->split_next_);
    std::unique_ptr<Glm5Model> rest = std::move(mp->split_next_);
    mp.reset();
    M = nullptr;
    const double t_destroy = ms_since(t2);

    // ---- load: the two parts as they were, the earlier first (the lent card's context comes back with its first
    //      allocation)
    const auto fatal = [&](const std::string& what) {
        std::fprintf(stderr, "glm lend: FATAL: %s - layers [%d, %d) are gone and could not be loaded again: the engine "
                             "ends\n", what.c_str(), merged.l0, merged.l1);
        std::fflush(stderr);
        std::_Exit(1);
    };
    const auto t3 = clock::now();
    std::unique_ptr<Glm5Model> a = resplit_make(plan.lo, mi, plan.n_parts, devs_was);
    std::unique_ptr<Glm5Model> b = a != nullptr ? resplit_make(plan.hi, mi + 1, plan.n_parts, devs_was) : nullptr;
    if (a == nullptr || b == nullptr) {
        // the merged part again: the lease stays open
        a.reset();
        std::fprintf(stderr, "glm lend: CUDA%d a part did not load again: the merged part is rebuilt on CUDA%d\n", ldev,
                     ndev);
        std::unique_ptr<Glm5Model> np = resplit_make(merged, mi, plan.n_parts, devs_now);
        if (np == nullptr) fatal("the rebuild failed too");
        Glm5Model* npp = np.get();
        np->split_next_ = std::move(rest);
        prev->split_next_ = std::move(np);
        std::string e;
        if (!resplit_finish({npp}, e)) fatal("the rebuilt merged split did not finish: " + e);
        if (!have_state || !resplit_state_load(st, e)) {
            reset();
            for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) m->snap_pos_ = -1;
        }
        leave_card(ldev, dev_);
        err = "glm lend: CUDA" + std::to_string(ldev) + " a part did not load again; the merged split stands and the "
              "lease stays open";
        return false;
    }
    const double t_load = ms_since(t3);

    // ---- link
    const auto t4 = clock::now();
    Glm5Model* ap = a.get();
    Glm5Model* bp = b.get();
    b->split_next_ = std::move(rest);
    a->split_next_ = std::move(b);
    prev->split_next_ = std::move(a);
    {
        std::string e;
        if (!resplit_finish({ap, bp}, e)) fatal("the returned split did not finish: " + e);
    }
    const double t_link = ms_since(t4);

    // ---- restore
    const auto t5 = clock::now();
    if (have_state) {
        std::string e;
        if (!resplit_state_load(st, e)) {
            std::fprintf(stderr, "glm lend: the conversation state did not load back onto the returned split (%s): the "
                                 "next request reads its prompt again\n", e.c_str());
            reset();
            for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) m->snap_pos_ = -1;
        }
    } else {
        reset();
        for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) m->snap_pos_ = -1;
    }
    const double t_restore = ms_since(t5);

    // ---- check: the load's room check again for the returned cards the operator named - the lent card, and the
    //      neighbour when it is named too (both parts are fresh models) - so each may be lent whole again (whole_mib(dev)
    //      set anew)
    const auto t6 = clock::now();
    lend_whole_recheck(resplit_recheck_devs({plan.lo.dev, plan.hi.dev}, lend_whole_devs_));
    const double t_check = ms_since(t6);

    const Glm5Model* lp = plan.lo.dev == ldev ? ap : bp;
    const Glm5Model* op = plan.lo.dev == ldev ? bp : ap;
    const std::string state = resplit_state_words(st);
    std::fprintf(stderr, "glm lend: CUDA%d returned: it holds layers [%d, %d) again (RAM tier %.2f GB), CUDA%d layers "
                         "[%d, %d) (RAM tier %.2f GB), whole-lendable %s; the split %s; %s | quiet %.0f ms, save %.0f ms "
                         "(%.1f MB), destroy %.0f ms, load %.0f ms, link %.0f ms, restore %.0f ms, check %.0f ms, %.1f s "
                         "in all\n",
                 ldev, lp->l0_, lp->l1_, (double) lp->fast_->ram_bytes / 1073741824.0, op->dev_, op->l0_, op->l1_,
                 (double) op->fast_->ram_bytes / 1073741824.0, whole_mib(ldev) > 0 ? "again" : "NOT (see above)",
                 resplit_where().c_str() + 7, state.c_str(), t_quiet, t_save,
                 (double) (st.snap.size() + st.live.size()) / 1048576.0, t_destroy, t_load, t_link, t_restore, t_check,
                 ms_since(t_all) / 1000.0);
    return true;
}

}  // namespace strata::core
