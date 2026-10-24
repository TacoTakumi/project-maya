// src/core/glm_lend.cu - lending the GLM expert pool's main slots to another program on the same machine, card by
// card, and taking them back: the strata::core::VramLender over a model's parts (make_glm_lender, glm_model.hpp).
//
// What a card lends.  A part's main expert slots are one VramPool on its card (fast_setup): blocks of whole slots, a
// layer's slots one region, the regions ordered by slot size (a block holds one slot size but may hold two layers).
// A lend takes whole blocks out of that pool: their slots leave the decode's tables (lend_take), the card is
// synchronized and the blocks go back to the driver (lend_unmap).  A return maps them again at the same addresses and
// the slots are free (lend_free): the boundaries hand them out as spares and refill them with background promotions as
// the routes ask (fast_boundary steps 3 and 5).  The pool's tail - the prompt path's buffers while a prompt runs, its
// own allocation - is never lent here: it stays what the prompt path borrows (lend_tail), and that touches only slots
// [n_main, n), this only [0, n_main).
//
// The lendable rule (strata::core::lendable_bytes) on a part:
//   pool    = its main-slot blocks still mapped + the tail (FastState::xpool_bytes, fast_setup's k_extra slots a layer)
//   borrow  = the tail: what the prompt path borrows for its buffers (prefill_lend)
//   minimum = Glm5Model::pool_keep(n_exp_used) slots a layer - one route's experts, the device's spares and 2, the
//             floor the prompt path's lending already keeps (and route + spares the pool's own sizing needs) - in the
//             pool bytes they take (a slot's share of its block, block / per_block)
// rounded down to whole blocks and capped at the blocks lend_choose can take (no layer below its minimum, a block two
// layers share counted in both).  The tail is not lendable, so this is in effect the mapped main slots less the
// minimum; the prompt path's own borrow is never touched, and a lend cannot push it back to token by token.  A card
// with several parts lends what they lend together (the parts of its first part's block size).
//
// Where their experts go.  A lent slot's expert moves into an empty slot of its layer or over a colder resident (a
// device copy, cheap); whatever leaves VRAM - that colder one, or the lent slot's own expert where no slot is colder -
// goes to the RAM tier hottest first while the lend's park budget lasts (lend_park_budget: STRATA_GLM_LEND_PARK_MB,
// 512 MiB), the rest to disk.  The decode is stopped for the lend, and a D2H copy of 8 GiB of experts over a PCIe 4.0
// x4 link took 1.9 s of it; dropped experts cost nothing now and are read from disk when a route wants them.
//
// Which blocks.  lend_choose, one block at a time: a block of one layer before a block two layers share (so how many
// can go does not follow the heat), and of those the one whose leaving costs least a slot.  A layer that loses c
// slots loses its c coldest experts (a lent slot's expert moves into an empty slot of its layer or over a colder
// resident), so the cost is the layer's c coldest slots' worth (an empty slot 0, a resident 1 + its aged LFU count).
// Equal worth - a fresh start, or cold layers - spreads the lend over the layers and takes a region's end first.
//
// The order (lend_part):  bg_hold (no new background move), the copy stream synchronized and fast_boundary (the
// moves in flight land, the service thread has taken every route: the part is quiet) -> the blocks chosen -> under
// F.mu the slots leave the tables and the device tables learn it on F.cs (synchronized) -> cudaDeviceSynchronize on the
// card (every stream: compute, copy, prefetch, the prompt path's, window lanes, MTP, DFlash) -> VramPool::unmap ->
// bg_hold off.  A return: VramPool::map (all or nothing; Busy: the lease stays open, nothing changed) -> the slots
// kFree under F.mu.
//
// When to call: see make_glm_lender in glm_model.hpp - from the engine's command loop, with nothing in flight.
#include "glm_fast_state.hpp"

#include "strata/core/vram_lend.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>

namespace strata::core {
namespace glmfast {

std::vector<int> lend_choose(const std::vector<LendBlock>& cand, const std::vector<int>& have,
                             const std::vector<int>& keep, const std::vector<std::vector<uint64_t>>& heat, int want) {
    const int NLy = (int) have.size();
    // each layer's worth of its coldest c slots: prefix sums of heat
    std::vector<std::vector<uint64_t>> pre((size_t) NLy);
    for (int l = 0; l < NLy; ++l) {
        pre[(size_t) l].assign(1, 0);
        if ((size_t) l < heat.size())
            for (uint64_t h : heat[(size_t) l]) pre[(size_t) l].push_back(pre[(size_t) l].back() + h);
    }
    std::vector<int> taken((size_t) NLy, 0), out;
    std::vector<char> used(cand.size(), 0);
    const auto worth = [&](int l, int c) {
        const auto& p = pre[(size_t) l];
        const size_t a = std::min(p.size() - 1, (size_t) taken[(size_t) l]);
        const size_t b = std::min(p.size() - 1, (size_t) (taken[(size_t) l] + c));
        return p[b] - p[a];
    };
    while (want < 0 || (int) out.size() < want) {
        int best = -1, b_spans = 0, b_room = 0;
        double b_per = 0.0;
        for (size_t i = 0; i < cand.size(); ++i) {
            if (used[i]) continue;
            bool ok = true;
            uint64_t w = 0;
            int n = 0, room = INT_MAX;
            for (const auto& [l, c] : cand[i].spans) {
                const int lim = (size_t) l < keep.size() ? keep[(size_t) l] : INT_MAX;
                if (l < 0 || l >= NLy || c < 0 || have[(size_t) l] - taken[(size_t) l] - c < lim) {
                    ok = false;
                    break;
                }
                w += worth(l, c);
                n += c;
                room = std::min(room, have[(size_t) l] - taken[(size_t) l] - lim);   // slots above the minimum now
            }
            if (!ok) continue;
            const double per = n > 0 ? (double) w / n : 0.0;
            const int ns = (int) cand[i].spans.size();
            // (the span count first: a shared block taken early, its few slots the cheapest, can leave a layer short
            //  of a whole block of its own - how many blocks can go would then follow the heat)
            const bool better =
                best < 0 || ns < b_spans ||
                (ns == b_spans && (per < b_per ||
                                   (per == b_per && (room > b_room ||
                                                     (room == b_room && cand[i].block > cand[(size_t) best].block)))));
            if (better) {
                best = (int) i;
                b_per = per;
                b_spans = ns;
                b_room = room;
            }
        }
        if (best < 0) break;
        used[(size_t) best] = 1;
        for (const auto& [l, c] : cand[(size_t) best].spans) taken[(size_t) l] += c;
        out.push_back(cand[(size_t) best].block);
    }
    return out;
}

size_t lend_capacity(size_t pool_bytes, size_t borrow_bytes, size_t min_bytes, size_t block_bytes, int blocks) {
    const size_t rule = strata::core::lendable_bytes(pool_bytes, borrow_bytes, min_bytes, block_bytes);
    return std::min(rule, (size_t) std::max(0, blocks) * block_bytes);
}

LendView lend_view(Glm5Model::FastState& F, int l0, int lt, int n_expert, int keep) {
    using FS = Glm5Model::FastState;
    LendView v;
    const size_t NL = F.lp.size();
    v.have.assign(NL, 0);
    v.keep.assign(NL, 0);
    v.heat.assign(NL, {});
    if (!F.vpool.valid() || !F.vpool.lendable()) return v;
    std::lock_guard<std::mutex> lk(F.mu);
    for (int il = l0; il < lt; ++il) {
        if ((size_t) il >= NL || (size_t) il >= F.L.size() || !F.L[(size_t) il].moe) continue;
        const auto& P = F.lp[(size_t) il];
        if (P.region < 0) continue;
        auto& h = v.heat[(size_t) il];
        for (int s = 0; s < P.n_main; ++s) {
            const char st = P.st[(size_t) s];
            if (st == FS::kLent) continue;
            ++v.have[(size_t) il];
            const int e = P.key[(size_t) s];
            h.push_back(st == FS::kResident && e >= 0 ? 1 + (uint64_t) F.cnt[(size_t) il * n_expert + e] : 0);
        }
        std::sort(h.begin(), h.end());
        v.keep[(size_t) il] = keep;
        v.min_bytes += (size_t) keep * P.block / (size_t) std::max(1, P.per_block);
    }
    for (int b = 0; b < F.vpool.block_count(); ++b) {
        if (!F.vpool.mapped(b)) continue;
        LendBlock x;
        x.block = b;
        bool lent = false;
        for (const auto& sp : F.vpool.block_slots(b)) {
            const int il = F.pool_layer[(size_t) sp.region];
            const auto& P = F.lp[(size_t) il];
            for (int s = sp.first; s < sp.first + sp.count && !lent; ++s) lent = P.st[(size_t) s] == FS::kLent;
            x.spans.push_back({il < l0 || il >= lt ? -1 : il, sp.count});
        }
        if (!lent && !x.spans.empty()) v.cand.push_back(std::move(x));
    }
    return v;
}

size_t lend_park_budget() {
    constexpr size_t kDefault = (size_t) 512 << 20;
    const char* v = std::getenv("STRATA_GLM_LEND_PARK_MB");
    if (v == nullptr || v[0] < '0' || v[0] > '9') return kDefault;
    char* end = nullptr;
    const unsigned long long mb = std::strtoull(v, &end, 10);
    if (*end != '\0' || mb > ((unsigned long long) SIZE_MAX >> 20)) return kDefault;
    return (size_t) mb << 20;
}

bool lend_take(Glm5Model::FastState& F, int l0, int lt, int n_expert, const std::vector<int>& blocks, LendCounts& c,
               std::string& err, size_t park_max) {
    using FS = Glm5Model::FastState;
    const int NE = n_expert;
    std::lock_guard<std::mutex> lk(F.mu);
    // the set, per layer: in[il][s] for its main slots (checked before anything changes)
    std::map<int, std::vector<char>> in;
    for (int b : blocks) {
        if (!F.vpool.mapped(b)) {
            err = "glm lend: block " + std::to_string(b) + " is not mapped";
            return false;
        }
        for (const auto& sp : F.vpool.block_slots(b)) {
            const int il = F.pool_layer[(size_t) sp.region];
            auto& P = F.lp[(size_t) il];
            if (il < l0 || il >= lt) {
                err = "glm lend: block " + std::to_string(b) + " holds slots of layer " + std::to_string(il) +
                      ", not this part's";
                return false;
            }
            auto& m = in[il];
            if (m.empty()) m.assign((size_t) P.n_main, 0);
            for (int s = sp.first; s < sp.first + sp.count; ++s) {
                const char st = P.st[(size_t) s];
                if (st != FS::kFree && st != FS::kResident && st != FS::kSpare) {
                    err = "glm lend: layer " + std::to_string(il) + " slot " + std::to_string(s) +
                          (st == FS::kLent ? " is lent already" : " has a tier move in flight (not quiesced)");
                    return false;
                }
                m[(size_t) s] = 1;
            }
        }
    }
    int nu = 0;
    bool ok = true;
    const auto flush = [&] {
        if (nu > 0 && ok) {
            ok = cuda_ok(cudaMemcpyAsync(F.upd_key_d, F.upd_key_h, (size_t) nu * sizeof(int), cudaMemcpyHostToDevice,
                                         F.cs), "glm lend keys", err) &&
                 cuda_ok(cudaMemcpyAsync(F.upd_val_d, F.upd_val_h, (size_t) nu * sizeof(unsigned long long),
                                         cudaMemcpyHostToDevice, F.cs), "glm lend values", err);
            if (ok) {
                gf::tab_update(F.tab, F.upd_key_d, F.upd_val_d, nu, F.cs);
                ok = cuda_ok(cudaGetLastError(), "glm lend table update", err);
            }
        }
        // (the parks and moves below are on F.cs too: they have read their slots once this returns)
        const cudaError_t e = cudaStreamSynchronize(F.cs);
        if (ok) ok = cuda_ok(e, "glm lend sync", err);
        nu = 0;
        ++F.upd_g;
    };
    ++F.upd_g;
    // a batch holds each table key once, its last value (as fast_boundary's)
    const auto upd = [&](size_t k, unsigned long long v) {
        if (!ok) return;
        if (F.upd_gen[k] == F.upd_g) {
            F.upd_val_h[F.upd_at[k]] = v;
            return;
        }
        if (nu == FS::kMaxUpd) {
            flush();
            if (!ok) return;
        }
        F.upd_gen[k] = F.upd_g;
        F.upd_at[k] = nu;
        F.upd_key_h[nu] = (int) k;
        F.upd_val_h[nu] = v;
        ++nu;
    };
    const auto left = [&](int key, uint8_t why) {
        if ((size_t) key < F.left.size()) F.left[(size_t) key] = why;
    };
    // an expert that leaves VRAM goes to the RAM tier where it has room - a free slot, else over a colder held expert
    // (one this lend parked is never the victim) - so the decode finds it in RAM, not on disk; false: it is dropped.
    // `tick`: its recency in the slot it leaves
    std::vector<char> parked;
    const auto park = [&](int il, int s, int key, uint64_t tick) -> bool {
        const int rc = (size_t) il < F.layer_rc.size() ? F.layer_rc[(size_t) il] : -1;
        if (rc < 0 || (size_t) rc >= F.rc.size()) return false;
        if (parked.empty()) parked.assign(F.ram_of.size(), 0);
        auto& R = F.rc[(size_t) rc];
        int rs = -1;
        for (int s2 = 0; s2 < R.n && rs < 0; ++s2)
            if (R.st[(size_t) s2] == FS::kRFree) rs = s2;
        if (rs < 0) {
            uint32_t bc = F.cnt[(size_t) key];
            uint64_t bt = tick;
            for (int s2 = 0; s2 < R.n; ++s2) {
                if (R.st[(size_t) s2] != FS::kRHold || parked[(size_t) R.key[(size_t) s2]]) continue;
                const uint32_t c2 = F.cnt[(size_t) R.key[(size_t) s2]];
                const uint64_t t2 = R.tick[(size_t) s2];
                if (c2 < bc || (c2 == bc && t2 < bt)) {
                    bc = c2;
                    bt = t2;
                    rs = s2;
                }
            }
            if (rs < 0) return false;
            const int old = R.key[(size_t) rs];
            upd(F.rtab_key(old), 0ull);
            F.ram_of[(size_t) old] = -1;
            left(old, 4);
        }
        if (!cuda_ok(cudaMemcpyAsync(R.slot((size_t) rs), F.lp[(size_t) il].slot_ptr(s), F.L[(size_t) il].blob,
                                     cudaMemcpyDeviceToHost, F.cs), "glm lend park", err)) {
            ok = false;
            return false;
        }
        R.key[(size_t) rs] = key;
        R.tick[(size_t) rs] = F.clock;
        R.st[(size_t) rs] = FS::kRHold;
        F.ram_of[(size_t) key] = rs;
        parked[(size_t) key] = 1;
        upd(F.rtab_key(key), (unsigned long long) R.slot((size_t) rs));
        return true;
    };
    const auto clear_spare = [&](int il, int s) {
        auto& P = F.lp[(size_t) il];
        for (int j = 0; j < gf::kSpares; ++j)
            if (P.spare[j] == s) {
                P.spare[j] = -1;
                upd(F.spare_key(il, j), 0ull);
            }
    };
    // the plan first, on the host: where each lent slot's expert goes and which experts leave VRAM (a leaving one's
    // slot still holds it: a victim's until its move below, a lent slot's until the unmap).  Then the leaving ones'
    // copies to RAM, hottest first within the budget, and only then the moves, which overwrite the victims' slots
    // (all on F.cs, in that order)
    struct Leave {
        int il, s, key;
        uint32_t cnt;
        uint64_t tick;
    };
    struct Move {
        int il, from, to;
    };
    std::vector<Leave> leaving;
    std::vector<Move> moves;
    for (auto& [il, m] : in) {
        auto& P = F.lp[(size_t) il];
        const auto colder = [&](int a, int b) {   // slot a's expert before slot b's, coldest first
            const uint32_t ca = F.cnt[(size_t) il * NE + P.key[(size_t) a]], cb = F.cnt[(size_t) il * NE + P.key[(size_t) b]];
            return ca != cb ? ca < cb : P.tick[(size_t) a] < P.tick[(size_t) b];
        };
        // the targets outside the set: its empty slots (free or a spare), then its residents, coldest first
        std::vector<int> empty, cold, movers;
        for (int s = 0; s < P.n_main; ++s) {
            const char st = P.st[(size_t) s];
            const bool res = st == FS::kResident && P.key[(size_t) s] >= 0;
            if (m[(size_t) s]) {
                if (res) movers.push_back(s);
            } else if (st == FS::kFree || st == FS::kSpare) {
                empty.push_back(s);
            } else if (res) {
                cold.push_back(s);
            }
        }
        std::sort(cold.begin(), cold.end(), colder);
        std::sort(movers.begin(), movers.end(), [&](int a, int b) { return colder(b, a); });   // hottest first
        size_t ei = 0, ci = 0;
        for (int s : movers) {
            const int key = il * NE + P.key[(size_t) s];
            int t = -1;
            if (ei < empty.size()) {
                t = empty[ei++];
                clear_spare(il, t);
            } else if (ci < cold.size() && colder(cold[ci], s)) {
                t = cold[ci++];
                const int vkey = il * NE + P.key[(size_t) t];
                upd(F.tab_key(vkey), 0ull);
                F.slot_of[(size_t) vkey] = -1;
                leaving.push_back({il, t, vkey, F.cnt[(size_t) vkey], P.tick[(size_t) t]});
            }
            if (!ok) return false;
            if (t >= 0) {
                moves.push_back({il, s, t});
                upd(F.tab_key(key), (unsigned long long) P.slot_ptr(t));
                F.slot_of[(size_t) key] = t;
                P.key[(size_t) t] = P.key[(size_t) s];
                P.tick[(size_t) t] = P.tick[(size_t) s];
                P.st[(size_t) t] = FS::kResident;
                ++c.moved;
            } else {
                upd(F.tab_key(key), 0ull);
                F.slot_of[(size_t) key] = -1;
                leaving.push_back({il, s, key, F.cnt[(size_t) key], P.tick[(size_t) s]});
                if (!ok) return false;
            }
        }
        for (int s = 0; s < P.n_main; ++s) {
            if (!m[(size_t) s]) continue;
            if (P.st[(size_t) s] == FS::kSpare) clear_spare(il, s);
            P.key[(size_t) s] = -1;
            P.st[(size_t) s] = FS::kLent;
            ++P.n_lent;
            ++c.slots;
        }
    }
    std::stable_sort(leaving.begin(), leaving.end(), [](const Leave& a, const Leave& b) {
        return a.cnt != b.cnt ? a.cnt > b.cnt : a.tick > b.tick;
    });
    for (const Leave& x : leaving) {
        const size_t blob = F.L[(size_t) x.il].blob;
        if (blob <= park_max - c.park_bytes && park(x.il, x.s, x.key, x.tick)) {
            ++c.parked;
            c.park_bytes += blob;
        } else {
            if (!ok) return false;
            left(x.key, 3);
            ++c.dropped;
        }
    }
    for (const Move& mv : moves) {
        const auto& P = F.lp[(size_t) mv.il];
        if (!cuda_ok(cudaMemcpyAsync(P.slot_ptr(mv.to), P.slot_ptr(mv.from), F.L[(size_t) mv.il].blob,
                                     cudaMemcpyDeviceToDevice, F.cs), "glm lend move", err))
            return false;
    }
    flush();
    if (!ok) return false;
    F.check_resident(l0, lt, NE);   // (the lent slots' experts left VRAM: their routes wait and fetch again)
    return true;
}

void lend_free(Glm5Model::FastState& F, const std::vector<int>& blocks) {
    using FS = Glm5Model::FastState;
    std::lock_guard<std::mutex> lk(F.mu);
    for (int b : blocks)
        for (const auto& sp : F.vpool.block_slots(b)) {
            auto& P = F.lp[(size_t) F.pool_layer[(size_t) sp.region]];
            for (int s = sp.first; s < sp.first + sp.count; ++s)
                if (P.st[(size_t) s] == FS::kLent) {
                    P.st[(size_t) s] = FS::kFree;
                    P.key[(size_t) s] = -1;
                    --P.n_lent;
                }
        }
}

bool lend_unmap(Glm5Model::FastState& F, int dev, const std::vector<int>& blocks, std::string& err) {
    // cuMemUnmap does not wait for kernels in flight: every stream of the card finishes first
    if (!cuda_ok(cudaDeviceSynchronize(), "glm lend: the card's streams", err)) return false;
    if (F.vpool.unmap(blocks, err) == VramPool::Result::Ok) return true;
    // what did go comes back and every slot of the set is free again (they hold nothing)
    std::vector<int> gone, back;
    for (int b : blocks) (F.vpool.mapped(b) ? back : gone).push_back(b);
    std::string e2;
    if (!gone.empty() && F.vpool.map(gone, e2) != VramPool::Result::Ok) {
        std::fprintf(stderr, "glm lend: CUDA%d %zu blocks did not map back after a failed lend (%s): their slots stay "
                             "out of the pool\n", dev, gone.size(), e2.c_str());
        lend_free(F, back);
        return false;
    }
    lend_free(F, blocks);
    return false;
}

}  // namespace glmfast

namespace {

using FastState = Glm5Model::FastState;

// a part's lending figures now: its block, the blocks it can lend
struct PartLend {
    size_t block = 0;
    int blocks = 0;
};

bool part_lends(const Glm5Model& m) {
    const FastState* F = m.fast_;
    return F != nullptr && F->vpool.valid() && F->vpool.lendable();
}

PartLend part_lendable(Glm5Model& m) {
    PartLend p;
    if (!part_lends(m)) return p;
    FastState& F = *m.fast_;
    const Glm5Geometry& g = m.geometry();
    const glmfast::LendView v = glmfast::lend_view(F, m.l0_, m.lt_, g.n_expert, Glm5Model::pool_keep(g.n_exp_used));
    const int nb = (int) glmfast::lend_choose(v.cand, v.have, v.keep, v.heat, -1).size();
    p.block = F.vpool.block_bytes();
    const size_t pool = (size_t) F.vpool.mapped_count() * p.block + F.xpool_bytes;
    p.blocks = (int) (glmfast::lend_capacity(pool, F.xpool_bytes, v.min_bytes, p.block, nb) / p.block);
    return p;
}

// n blocks of part m out of the pool (see the order at the top), at most park_max bytes of experts copied to RAM;
// `blocks` the ones taken
bool lend_part(Glm5Model& m, int n, size_t park_max, std::vector<int>& blocks, glmfast::LendCounts& c,
               std::string& err) {
    FastState* F = m.fast_;
    const Glm5Geometry& g = m.geometry();
    cudaSetDevice(m.dev_);
    F->bg_hold = true;   // (fast_boundary step 3 and 5: no new demotions, promotions or helper swaps)
    struct Hold {
        FastState* F;
        ~Hold() { F->bg_hold = false; }
    } hold{F};
    if (!glmfast::cuda_ok(cudaStreamSynchronize(F->copy), "glm lend copies", err) || !m.fast_boundary(err))
        return false;
    if (!F->bg.empty() || !F->draining.empty()) {
        err = "glm lend: tier moves were not retired";
        return false;
    }
    const glmfast::LendView v = glmfast::lend_view(*F, m.l0_, m.lt_, g.n_expert, Glm5Model::pool_keep(g.n_exp_used));
    blocks = glmfast::lend_choose(v.cand, v.have, v.keep, v.heat, n);
    if ((int) blocks.size() < n) {
        err = "glm lend: CUDA" + std::to_string(m.dev_) + " can lend " + std::to_string(blocks.size()) +
              " blocks, not " + std::to_string(n);
        return false;
    }
    return glmfast::lend_take(*F, m.l0_, m.lt_, g.n_expert, blocks, c, err, park_max) &&
           glmfast::lend_unmap(*F, m.dev_, blocks, err);
}

class GlmLender final : public VramLender {
public:
    explicit GlmLender(Glm5Model& head) : head_(head) {}

    std::vector<LendCard> lend_cards() override {
        std::vector<LendCard> out;
        for (const Card& k : cards()) {
            size_t bytes = 0;
            for (Glm5Model* m : k.parts) bytes += (size_t) part_lendable(*m).blocks * k.block;
            out.push_back({k.dev, bytes >> 20, k.block >> 20});
        }
        return out;
    }

    LendResult lend(int dev, size_t mib) override {
        LendResult r;
        int prev = 0;
        cudaGetDevice(&prev);
        const std::vector<Card> ks = cards();
        const Card* k = nullptr;
        for (const Card& x : ks)
            if (x.dev == dev) k = &x;
        if (k == nullptr || k->block == 0) return r;
        const size_t bmib = k->block >> 20;
        const int want = (int) ((mib + bmib - 1) / bmib);
        std::vector<int> can, n(k->parts.size(), 0);
        int total = 0;
        for (Glm5Model* m : k->parts) {
            can.push_back(part_lendable(*m).blocks);
            total += can.back();
        }
        r.max_mib = (size_t) total * bmib;
        if (want <= 0 || want > total) return r;
        // the parts of one card share the lend: each block to the part with the most left to lend
        for (int i = 0; i < want; ++i) {
            size_t best = 0;
            for (size_t j = 1; j < n.size(); ++j)
                if (can[j] - n[j] > can[best] - n[best]) best = j;
            ++n[best];
        }
        Lease L;
        L.dev = dev;
        size_t park_left = glmfast::lend_park_budget();   // (the card's: its parts share it)
        for (size_t j = 0; j < n.size(); ++j) {
            if (n[j] == 0) continue;
            Glm5Model* m = k->parts[j];
            std::vector<int> blocks;
            glmfast::LendCounts c;
            std::string err;
            const auto t0 = std::chrono::steady_clock::now();
            if (!lend_part(*m, n[j], park_left, blocks, c, err)) {
                std::fprintf(stderr, "glm lend: CUDA%d the lend of %zu MiB failed: %s\n", dev, (size_t) want * bmib,
                             err.c_str());
                // the parts that lent already take theirs back (nothing uses those blocks yet)
                if (!put_back(L, err))
                    std::fprintf(stderr, "glm lend: CUDA%d ... and %s\n", dev, err.c_str());
                cudaSetDevice(prev);
                size_t now = 0;   // (the refusal's figure: what the card can lend after it)
                for (Glm5Model* p : k->parts) now += (size_t) part_lendable(*p).blocks * bmib;
                r.max_mib = now;
                return r;
            }
            park_left -= c.park_bytes;
            std::fprintf(stderr, "glm lend: CUDA%d layers [%d,%d) %zu MiB lent in %.0f ms (%zu blocks, %d slots; %llu "
                                 "experts moved to colder slots, %llu to the RAM tier (%zu MiB), %llu to disk)\n",
                         dev, m->l0_, m->l1_, blocks.size() * bmib,
                         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
                         blocks.size(), c.slots, (unsigned long long) c.moved, (unsigned long long) c.parked,
                         c.park_bytes >> 20, (unsigned long long) c.dropped);
            L.sets.push_back({m, std::move(blocks)});
        }
        cudaSetDevice(prev);
        const uint64_t token = next_++;
        leases_[token] = std::move(L);
        r.ok = true;
        r.mib = (size_t) want * bmib;
        r.token = token;
        return r;
    }

    GiveBack give_back(int dev, uint64_t token) override {
        const auto it = leases_.find(token);
        if (it == leases_.end()) return GiveBack::returned;   // (never lent by this lender: nothing to put back)
        int prev = 0;
        cudaGetDevice(&prev);
        std::string err;
        const bool ok = put_back(it->second, err);
        cudaSetDevice(prev);
        if (!ok) {
            std::fprintf(stderr, "glm lend: CUDA%d a return waits: %s\n", dev, err.c_str());
            return GiveBack::busy;
        }
        size_t nb = 0;
        for (const auto& s : it->second.sets) nb += s.second.size();
        std::fprintf(stderr, "glm lend: CUDA%d %zu blocks are back in the expert pool\n", dev, nb);
        leases_.erase(it);
        return GiveBack::returned;
    }

private:
    // a card and the parts on it that lend in its block size (the first lending part's)
    struct Card {
        int dev = 0;
        size_t block = 0;
        std::vector<Glm5Model*> parts;
    };
    struct Lease {
        int dev = 0;
        std::vector<std::pair<Glm5Model*, std::vector<int>>> sets;   // part, its blocks
    };

    std::vector<Card> cards() {
        std::vector<Card> out;
        for (Glm5Model* m = &head_; m != nullptr; m = m->split_next_.get()) {
            Card* k = nullptr;
            for (Card& x : out)
                if (x.dev == m->dev_) k = &x;
            if (k == nullptr) {
                out.push_back({m->dev_, 0, {}});
                k = &out.back();
            }
            if (!part_lends(*m)) continue;
            const size_t b = m->fast_->vpool.block_bytes();
            if (k->block == 0) k->block = b;
            if (b == k->block) k->parts.push_back(m);
        }
        return out;
    }

    // every set of the lease mapped back, all or nothing (a set mapped before one that cannot is unmapped again -
    // its slots are still lent, nothing reads them), then its slots free
    bool put_back(const Lease& L, std::string& err) {
        size_t done = 0;
        for (; done < L.sets.size(); ++done) {
            const auto& [m, blocks] = L.sets[done];
            cudaSetDevice(m->dev_);
            if (m->fast_->vpool.map(blocks, err) != VramPool::Result::Ok) break;
        }
        if (done < L.sets.size()) {
            for (size_t i = 0; i < done; ++i) {
                const auto& [m, blocks] = L.sets[i];
                cudaSetDevice(m->dev_);
                std::string e2;
                if (m->fast_->vpool.unmap(blocks, e2) != VramPool::Result::Ok)
                    std::fprintf(stderr, "glm lend: CUDA%d %s\n", m->dev_, e2.c_str());
            }
            return false;
        }
        for (const auto& [m, blocks] : L.sets) glmfast::lend_free(*m->fast_, blocks);
        return true;
    }

    Glm5Model& head_;
    std::map<uint64_t, Lease> leases_;   // by token
    uint64_t next_ = 1;
};

}  // namespace

std::unique_ptr<VramLender> make_glm_lender(Glm5Model& head) { return std::make_unique<GlmLender>(head); }

}  // namespace strata::core
