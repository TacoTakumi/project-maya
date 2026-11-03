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
#include "glm_fast_state.hpp"

#include "strata/core/glm_model.hpp"
#include "strata/kernels/glm_fast.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

void Glm5Model::lend_whole_check(const std::vector<int>& devs) {
    if (devs.empty()) return;
    // the split's figures, part by part (the VRAM line's, and the pool floor and RAM tier the rules add)
    std::vector<ResplitPart> parts;
    std::vector<Glm5Model*> chain;
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
    // the host's room: MemAvailable now and the headroom the RAM tier was sized against
    double head_gb = fast_ != nullptr && fast_->unified_memory ? 16.0 : 6.0;
    if (const char* h = getenv("STRATA_GLM_RAM_HEADROOM_GB")) head_gb = std::max(0.0, std::atof(h));
    const uint64_t head_b = (uint64_t) (head_gb * 1073741824.0);
    const uint64_t avail = ram_avail_now();
    std::set<int> seen;
    for (int dev : devs) {
        if (!seen.insert(dev).second) continue;   // (named twice: checked once)
        if (fast_ == nullptr) {
            std::fprintf(stderr, "glm lend: CUDA%d is not lent whole: no expert pool (the slow path)\n", dev);
            continue;
        }
        const ResplitRoom r = resplit_room(parts, dev, avail, head_b);
        if (!r.ok) {
            std::fprintf(stderr, "glm lend: CUDA%d is not lent whole: %s\n", dev, r.reason.c_str());
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

}  // namespace strata::core
