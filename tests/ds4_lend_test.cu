// tests/ds4_lend_test.cu - DeepSeek's lend of its expert pool (src/ds4/ds4_lend.cu), no model:
//  - the block choice over made-up blocks: the fixed slots' blocks (recent slots), foreign and lent blocks are never
//    candidates, a region short of its floor keeps the blocks that keep the most floors (then the most of its slots,
//    the lowest index), the lendable rule's figures, the pick spread evenly over the regions and coldest first;
//  - lendable the layout's: the same for full, half empty and differently hot layers, and after a lend of all of it,
//    its return (the slots empty) and the cache's fill; the same wherever the swap spares are, and after swaps move
//    them;
//  - the slot-table changes: slots out (their experts dropped, off the free list), back (on it when no card lends
//    them any more); the spare's new slot when a lend takes its own (an empty one, else the coldest expert's);
//  - the lender over a real VMM pool on one card (needs a GPU with ~2 GiB free; STRATA_TEST_DEV picks it) and a
//    made-up cache of four layers (one that cannot lend): a lend frees at least its MiB to the driver, no lent slot is
//    left in a slot table, the spare, the recent slots and the floor stay, the other slots keep their data; more than
//    lendable is refused with the max; a return while another allocation holds the memory is busy and keeps the lease;
//    a return maps the blocks back, their slots come back empty and a fill writes them; the plain fallback lends 0;
//    the steady layout in a pool of its own: lend the whole lendable, return, fill - lendable as before, lent again;
//    and with the spares moved through every slot: lendable the same, a lend of it with a spare in a block it takes
//    succeeds (the spare moved out first), swaps during a lease leave what is left to lend, the return what was.
#include "ds4_model.hpp"
#include "strata/core/vram_pool.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace lend = strata::ds4::lend;
using strata::core::VramPool;
using strata::core::VramPoolRegion;

namespace {

int g_fail = 0;

#define CHECK(c, ...)                                                                                            \
    do {                                                                                                         \
        if (!(c)) {                                                                                              \
            std::printf("FAIL %s:%d: %s: ", __FILE__, __LINE__, #c);                                             \
            std::printf(__VA_ARGS__);                                                                            \
            std::printf("\n");                                                                                   \
            ++g_fail;                                                                                            \
        }                                                                                                        \
    } while (0)

constexpr size_t kMiB = (size_t) 1 << 20;

bool has(const std::vector<int>& v, int x) { return std::find(v.begin(), v.end(), x) != v.end(); }

lend::Block blk(std::vector<lend::Slot> s, bool mapped = true, bool foreign = false) {
    lend::Block b;
    b.mapped = mapped;
    b.foreign = foreign;
    b.slots = std::move(s);
    return b;
}

// a layer of the made-up cache: main slots [0, n) in region `region` of the pool, n_exp experts
struct FakeLayer {
    int n = 0, n_exp = 0, spare = -1, rc_base = -1, rc_n = 0;
    bool lends = true;
    std::vector<int> slot;   // expert -> slot
    std::vector<float> heat;
    std::vector<int> out, free;
    bool recent(int s) const { return rc_n > 0 && s >= rc_base && s < rc_base + rc_n; }
    bool fixed(int s) const { return recent(s); }   // (lend_view's rule: the spare is a role, it moves)
    // the cache's swap (cache_switch): CPU-lane expert `in` into the spare, the coldest expert in a slot of its own
    // (not a recent one) out to the CPU lane, its slot the spare
    void swap(int in) {
        int cold = -1;
        for (int e = 0; e < n_exp; ++e) {
            const int s = slot[(size_t) e];
            if (s >= 0 && !recent(s) && (cold < 0 || heat[(size_t) e] < heat[(size_t) cold])) cold = e;
        }
        slot[(size_t) in] = spare;
        spare = slot[(size_t) cold];
        slot[(size_t) cold] = -1;
    }
    int cpu_expert() const {   // (the lowest on the CPU lane)
        for (int e = 0; e < n_exp; ++e)
            if (slot[(size_t) e] < 0) return e;
        return -1;
    }
    // the spare to slot s, the expert there to the spare's old slot: the same cache, the spare in another place
    void spare_to(int s) {
        for (int& x : slot)
            if (x == s) x = spare;
        spare = s;
    }
};

// a layer of the steady layout (steady_blocks): 5 slots, the spare in slot 0 (region 0) or 4 (region 1), one recent
// slot (1, 3), experts 0-3 in the other slots (region 0's hottest in slot 4), experts 4-7 on the CPU lane
FakeLayer steady_layer(int r) {
    FakeLayer l;
    l.n = 5;
    l.n_exp = 8;
    l.spare = r == 0 ? 0 : 4;
    l.rc_base = r == 0 ? 1 : 3;
    l.rc_n = 1;
    l.slot.assign(8, -1);
    l.out.assign(5, 0);
    for (int e = 0; e < 4; ++e) l.slot[(size_t) e] = r == 0 ? e + 1 : e;
    l.heat = r == 0 ? std::vector<float>{0.5f, 1.0f, 1.0f, 9.0f, 20.0f, 0.0f, 0.0f, 0.0f}
                    : std::vector<float>{2.0f, 2.0f, 2.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    return l;
}

// ---- the block choice, no GPU ----

void test_choice() {
    // region 0: blocks 0-3 (block 1 holds a recent slot); region 1: blocks 4-6 (5 the hottest); region 2 cannot lend
    // (block 7, foreign); block 8 lent already; block 9 the only one of region 3, shared with region 1
    std::vector<lend::Block> B;
    B.push_back(blk({{0, false, true, 1.0f}, {0, false, true, 2.0f}}));       // 0: heat 3
    B.push_back(blk({{0, true, true, 0.0f}, {0, false, true, 9.0f}}));        // 1: a recent slot
    B.push_back(blk({{0, false, true, 0.5f}, {0, false, true, 0.0f}}));       // 2: heat 0.5
    B.push_back(blk({{0, false, true, 4.0f}, {0, false, true, 4.0f}}));       // 3: heat 8
    B.push_back(blk({{1, false, true, 1.0f}, {1, false, true, 1.0f}}));       // 4: heat 2
    B.push_back(blk({{1, false, true, 7.0f}, {1, false, true, 7.0f}}));       // 5: heat 14
    B.push_back(blk({{1, false, true, 0.0f}, {1, false, false, 0.0f}}));      // 6: heat 0 (one slot lent elsewhere)
    B.push_back(blk({{2, false, true, 0.0f}, {1, false, true, 0.0f}}, true, true));   // 7: foreign
    B.push_back(blk({{1, false, true, 0.0f}, {1, false, true, 0.0f}}, false));        // 8: lent
    B.push_back(blk({{1, false, true, 3.0f}, {3, false, true, 3.0f}}));       // 9: region 3's only block
    const std::vector<int> c = lend::candidates(B, 4, 1);
    // region 0 keeps its floor in block 1 (a recent slot's, not lendable), region 1 in block 7 (foreign: never lent)
    // and block 9 (region 3's floor), region 3 in block 9: blocks 0, 2, 3, 4, 5 and 6 may go
    CHECK(c == std::vector<int>({0, 2, 3, 4, 5, 6}), "candidates (keep 1): %zu of them", c.size());
    // a floor of 3: region 0 has 1 in block 1 and keeps 0 (its lowest of three alike: 2 more, whatever their heat);
    // region 1 has 1 in block 7, keeps 9 (region 3's too) and then 4 (two of its own, the lowest); region 3 has 1 and
    // nothing else to keep
    const std::vector<int> c3 = lend::candidates(B, 4, 3);
    CHECK(c3 == std::vector<int>({2, 3, 5, 6}), "candidates (keep 3): %zu of them", c3.size());
    for (int b : c) CHECK(b != 1 && b != 7 && b != 8 && b != 9, "block %d is a candidate", b);

    const lend::Figures f = lend::figures(B, c.size(), 32 * kMiB);
    CHECK(f.pool == 8 * 32 * kMiB, "pool %zu MiB (mapped, not foreign: 8 blocks)", f.pool / kMiB);
    CHECK(f.min == 2 * 32 * kMiB, "min %zu MiB", f.min / kMiB);
    CHECK(f.lendable == 6 * 32 * kMiB, "lendable %zu MiB", f.lendable / kMiB);
    CHECK(lend::figures(B, c.size(), 0).lendable == 0, "a card with no blocks lends");

    // the pick: one block a region in turn, the coldest first
    const std::vector<int> p2 = lend::pick(B, c, 2);
    CHECK(p2.size() == 2 && has(p2, 2) && has(p2, 6), "pick 2: one of each region, the coldest (2, 6)");
    const std::vector<int> p4 = lend::pick(B, c, 4);
    CHECK(p4.size() == 4 && has(p4, 2) && has(p4, 6) && has(p4, 0) && has(p4, 4), "pick 4: then 0 and 4");
    const std::vector<int> p5 = lend::pick(B, c, 5);
    CHECK(p5.size() == 5 && has(p5, 3) && !has(p5, 5), "pick 5: then region 0's 3 (8) before region 1's 5 (14)");
    CHECK(lend::pick(B, c, 99).size() == c.size(), "pick more than there are");
    std::printf("choice: candidates %zu (keep 1), %zu (keep 3); lendable %zu MiB of a pool of %zu\n", c.size(),
                c3.size(), f.lendable / kMiB, f.pool / kMiB);
}

// two layers of 5 slots side by side, 2 a block (the pool's layout): region 0's two fixed slots in block 0,
// slots 2-3 in block 1, slot 4 in block 2 with region 1's slot 0; region 1's slots 1-2 in block 3, its two fixed
// slots in block 4 (recent slots, in two); keep 1.  h0, h1: each slot's expert's heat (0: empty)
std::vector<lend::Block> steady_blocks(const std::vector<float>& h0, const std::vector<float>& h1) {
    std::vector<lend::Block> B;
    B.push_back(blk({{0, true, true, h0[0]}, {0, true, true, h0[1]}}));
    B.push_back(blk({{0, false, true, h0[2]}, {0, false, true, h0[3]}}));
    B.push_back(blk({{0, false, true, h0[4]}, {1, false, true, h1[0]}}));
    B.push_back(blk({{1, false, true, h1[1]}, {1, false, true, h1[2]}}));
    B.push_back(blk({{1, true, true, h1[3]}, {1, true, true, h1[4]}}));
    return B;
}

void test_steady() {
    // both regions are short of their floor outside the fixed blocks; block 2 keeps both floors, so blocks 1 and 3
    // may go - whatever the heat, whichever slots hold an expert
    const std::vector<std::vector<float>> h0s = {{1, 1, 9, 9, 1}, {1, 1, 1, 1, 9}, {0, 0, 0, 0, 3}, {0, 0, 4, 0, 0},
                                                 {0, 0, 0, 0, 0}},
                                          h1s = {{2, 2, 2, 2, 2}, {0, 9, 9, 0, 0}, {5, 0, 0, 0, 0}, {0, 0, 0, 0, 0},
                                                 {0, 0, 0, 0, 0}};
    for (size_t i = 0; i < h0s.size(); ++i)
        for (size_t j = 0; j < h1s.size(); ++j) {
            const std::vector<lend::Block> B = steady_blocks(h0s[i], h1s[j]);
            const std::vector<int> c = lend::candidates(B, 2, 1);
            CHECK(c == std::vector<int>({1, 3}), "heat case %zu/%zu: %zu candidates", i, j, c.size());
        }

    // a lend of all of it, its return (the slots empty) and the cache's fill of them (the hottest experts)
    std::vector<lend::Block> B = steady_blocks({1, 1, 1, 1, 9}, {2, 2, 2, 2, 2});
    const std::vector<int> c0 = lend::candidates(B, 2, 1);
    const size_t l0 = lend::figures(B, c0.size(), 32 * kMiB).lendable;
    const std::vector<int> p = lend::pick(B, c0, (int) (l0 / (32 * kMiB)));
    CHECK(p.size() * 32 * kMiB == l0, "a pick of the lendable %zu MiB took %zu blocks", l0 / kMiB, p.size());
    for (int b : p) B[(size_t) b].mapped = false;
    CHECK(lend::candidates(B, 2, 1).empty(), "candidates with it all lent");
    for (int b : p) {
        B[(size_t) b].mapped = true;
        for (lend::Slot& s : B[(size_t) b].slots) s.heat = 0.0f;
    }
    const size_t l1 = lend::figures(B, lend::candidates(B, 2, 1).size(), 32 * kMiB).lendable;
    CHECK(l1 == l0, "lendable after the return %zu MiB, before %zu", l1 / kMiB, l0 / kMiB);
    for (int b : p)
        for (lend::Slot& s : B[(size_t) b].slots) s.heat = 20.0f;
    const size_t l2 = lend::figures(B, lend::candidates(B, 2, 1).size(), 32 * kMiB).lendable;
    CHECK(l2 == l0, "lendable after the fill %zu MiB, before %zu", l2 / kMiB, l0 / kMiB);
    std::printf("steady: lendable %zu MiB for any heat; %zu after a return, %zu after the fill\n", l0 / kMiB, l1 / kMiB,
                l2 / kMiB);
}

// the steady layout's blocks as lend_view sees two made-up layers (steady_layer) in it
std::vector<lend::Block> cache_blocks(const FakeLayer& a, const FakeLayer& b) {
    const int at[5][2][2] = {{{0, 0}, {0, 1}}, {{0, 2}, {0, 3}}, {{0, 4}, {1, 0}}, {{1, 1}, {1, 2}}, {{1, 3}, {1, 4}}};
    std::vector<lend::Block> B;
    for (const auto& k : at) {
        lend::Block x;
        for (const auto& rs : k) {
            const FakeLayer& l = rs[0] == 0 ? a : b;
            const int s = rs[1];
            float h = 0.0f;
            for (int e = 0; e < l.n_exp; ++e)
                if (l.slot[(size_t) e] == s) h = l.heat[(size_t) e];
            x.slots.push_back({rs[0], l.fixed(s), l.out.empty() || l.out[(size_t) s] == 0, h});
        }
        B.push_back(std::move(x));
    }
    return B;
}

void test_spare() {
    // the spare is a role: wherever it is (in any slot but a recent one, both regions), the same count
    FakeLayer a = steady_layer(0), b = steady_layer(1);
    const size_t n0 = lend::candidates(cache_blocks(a, b), 2, lend::kKeep).size();
    for (int p0 : {0, 2, 3, 4})
        for (int p1 : {0, 1, 2, 4}) {
            a.spare_to(p0);
            b.spare_to(p1);
            const size_t n = lend::candidates(cache_blocks(a, b), 2, lend::kKeep).size();
            CHECK(n == n0, "the spares in slots %d / %d: %zu candidates, %zu with them in 0 / 4", p0, p1, n, n0);
        }
    // the swaps of a decode move it (to the coldest expert's slot): the count stays
    a = steady_layer(0);
    b = steady_layer(1);
    for (int k = 0; k < 6; ++k) {
        FakeLayer& l = k % 2 == 0 ? a : b;
        l.heat[(size_t) (4 + k / 2)] = 50.0f + (float) k;
        l.swap(4 + k / 2);
        const size_t n = lend::candidates(cache_blocks(a, b), 2, lend::kKeep).size();
        CHECK(n == n0, "swap %d (spares in %d / %d): %zu candidates, %zu before", k, a.spare, b.spare, n, n0);
    }
    std::printf("spare: %zu candidates wherever the spares are, and after 6 swaps\n", n0);
}

void test_slots() {
    std::vector<int> slot = {3, -1, 0, 5, 1, -1, 4}, out(6, 0), free = {2}, dropped;
    lend::slots_out(slot, out, free, 1, 3, dropped);   // slots 1, 2, 3
    std::sort(dropped.begin(), dropped.end());
    CHECK(dropped == std::vector<int>({0, 4}), "dropped the experts of slots 1 and 3");
    CHECK(slot == std::vector<int>({-1, -1, 0, 5, -1, -1, 4}), "slot table after the lend");
    CHECK(out == std::vector<int>({0, 1, 1, 1, 0, 0}), "lent counts");
    CHECK(free.empty(), "slot 2 left the free list");
    dropped.clear();
    lend::slots_out(slot, out, free, 2, 1, dropped);   // slot 2 on a second card too
    CHECK(dropped.empty() && out[2] == 2, "a slot lent twice");
    lend::slots_back(out, free, 1, 3);
    std::sort(free.begin(), free.end());
    CHECK(free == std::vector<int>({1, 3}) && out[2] == 1, "back: 1 and 3 free, 2 still lent once");
    lend::slots_back(out, free, 2, 1);
    CHECK(has(free, 2) && out[2] == 0, "back on the second card: 2 free");
    // the spare's new slot: an empty one it may go to, else the coldest expert's (dropped), none: -1
    std::vector<char> ok = {0, 1, 1, 0, 1};
    const std::vector<int> table = {0, 1, 2, 4};   // expert -> slot (slot 3 empty, not ok)
    const std::vector<float> heat = {5.0f, 3.0f, 1.0f, 2.0f};
    int e = -2;
    CHECK(lend::spare_slot(table, heat, ok, e) == 2 && e == 2, "spare: the coldest expert's slot 2 (expert %d)", e);
    ok[3] = 1;
    CHECK(lend::spare_slot(table, heat, ok, e) == 3 && e == -1, "spare: the empty slot 3 (expert %d)", e);
    CHECK(lend::spare_slot(table, heat, std::vector<char>(5, 0), e) == -1, "spare: no slot to go to");
    std::printf("slot tables: ok\n");
}

// ---- the lender over a real pool and a made-up cache ----

size_t free_bytes() {
    size_t f = 0, t = 0;
    cudaMemGetInfo(&f, &t);
    return f;
}

struct FakeSite : lend::Site {
    int dev = 0;
    VramPool* P = nullptr;
    std::vector<FakeLayer> L;   // L[r]: region r
    int quiesced = 0, taken = 0, spare_moves = 0;
    std::vector<int> cards() override { return {dev}; }
    VramPool* pool(int d) override { return d == dev ? P : nullptr; }
    void quiesce() override { ++quiesced; }
    void view(int d, std::vector<lend::Block>& out, int& nr) override {
        out.clear();
        nr = (int) L.size();
        if (d != dev || P == nullptr) return;
        std::vector<std::vector<int>> who(L.size());
        for (size_t r = 0; r < L.size(); ++r) {
            who[r].assign((size_t) L[r].n, -1);
            for (int e = 0; e < L[r].n_exp; ++e)
                if (L[r].slot[(size_t) e] >= 0) who[r][(size_t) L[r].slot[(size_t) e]] = e;
        }
        for (int b = 0; b < P->block_count(); ++b) {
            lend::Block B;
            B.mapped = P->mapped(b);
            for (const auto& sp : P->block_slots(b)) {
                const FakeLayer& l = L[(size_t) sp.region];
                if (!l.lends) B.foreign = true;
                for (int s = sp.first; s < sp.first + sp.count; ++s) {
                    const int e = who[(size_t) sp.region][(size_t) s];
                    B.slots.push_back({sp.region, l.fixed(s), l.out.empty() || l.out[(size_t) s] == 0,
                                       e >= 0 ? l.heat[(size_t) e] : 0.0f});
                }
            }
            out.push_back(std::move(B));
        }
    }
    bool take(int d, const std::vector<int>& blocks, std::string& err) override {
        if (d != dev) {
            err = "no such card";
            return false;
        }
        // (lend_take's order: a spare in the blocks moves to a slot they leave first, every move found before any)
        std::vector<std::vector<char>> took(L.size());
        for (int b : blocks)
            for (const auto& sp : P->block_slots(b)) {
                std::vector<char>& t = took[(size_t) sp.region];
                if (t.empty()) t.assign((size_t) L[(size_t) sp.region].n, 0);
                for (int s = sp.first; s < sp.first + sp.count; ++s) t[(size_t) s] = 1;
            }
        std::vector<std::array<int, 3>> moves;
        for (size_t r = 0; r < L.size(); ++r) {
            const FakeLayer& l = L[r];
            if (took[r].empty() || l.spare < 0 || !took[r][(size_t) l.spare]) continue;
            std::vector<char> ok((size_t) l.n, 0);
            for (int s = 0; s < l.n; ++s)
                ok[(size_t) s] = !took[r][(size_t) s] && s != l.spare && !l.recent(s) && l.out[(size_t) s] == 0;
            int e = -1;
            const int to = lend::spare_slot(l.slot, l.heat, ok, e);
            if (to < 0) {
                err = "no slot left for the swap spare";
                return false;
            }
            moves.push_back({(int) r, to, e});
        }
        for (const std::array<int, 3>& mv : moves) {
            FakeLayer& l = L[(size_t) mv[0]];
            if (mv[2] >= 0) l.slot[(size_t) mv[2]] = -1;
            l.free.erase(std::remove(l.free.begin(), l.free.end(), mv[1]), l.free.end());
            l.spare = mv[1];
            ++spare_moves;
        }
        for (int b : blocks)
            for (const auto& sp : P->block_slots(b)) {
                FakeLayer& l = L[(size_t) sp.region];
                for (int s = sp.first; s < sp.first + sp.count; ++s)
                    CHECK(l.lends && !l.fixed(s) && s != l.spare, "a lend took region %d slot %d (fixed, the spare or "
                          "foreign)", sp.region, s);
                std::vector<int> dropped;
                lend::slots_out(l.slot, l.out, l.free, sp.first, sp.count, dropped);
            }
        ++taken;
        return cudaDeviceSynchronize() == cudaSuccess;
    }
    void put(int d, const std::vector<int>& blocks) override {
        if (d != dev) return;
        for (int b : blocks)
            for (const auto& sp : P->block_slots(b)) {
                FakeLayer& l = L[(size_t) sp.region];
                lend::slots_back(l.out, l.free, sp.first, sp.count);
            }
    }
    // the cache's fill of a returned slot: the hottest expert not in VRAM into a free slot (written on the card)
    int fill_all() {
        int n = 0;
        for (size_t r = 0; r < L.size(); ++r) {
            FakeLayer& l = L[r];
            while (!l.free.empty()) {
                int best = -1;
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] < 0 && (best < 0 || l.heat[(size_t) e] > l.heat[(size_t) best])) best = e;
                if (best < 0) break;
                const int s = l.free.back();
                l.free.pop_back();
                CHECK(l.out[(size_t) s] == 0, "a fill into region %zu slot %d, lent", r, s);
                l.slot[(size_t) best] = s;
                cudaMemset(P->slot_ptr((int) r, s), 0x5a, P->region((int) r).slot_bytes);
                ++n;
            }
        }
        return n;
    }
    // the swap's pick: the coldest expert in the slot table (not a recent slot's) - its slot must be a live one
    void check_tables(const char* when) {
        for (size_t r = 0; r < L.size(); ++r) {
            const FakeLayer& l = L[r];
            for (int e = 0; e < l.n_exp; ++e) {
                const int s = l.slot[(size_t) e];
                if (s < 0) continue;
                const int b = (int) ((size_t) (P->slot_ptr((int) r, s) - P->block_ptr(0)) / P->block_bytes());
                CHECK(P->mapped(b) && (l.out.empty() || l.out[(size_t) s] == 0),
                      "%s: region %zu expert %d in slot %d, whose block %d is lent", when, r, e, s, b);
            }
            if (l.spare >= 0) {
                CHECK(slot_mapped((int) r, l.spare) && (l.out.empty() || l.out[(size_t) l.spare] == 0),
                      "%s: region %zu's spare (slot %d) lent", when, r, l.spare);
                for (int e = 0; e < l.n_exp; ++e)
                    CHECK(l.slot[(size_t) e] != l.spare, "%s: region %zu expert %d in the spare", when, r, e);
            }
            for (int i = 0; i < l.rc_n; ++i)
                CHECK(slot_mapped((int) r, l.rc_base + i), "%s: region %zu's recent slot unmapped", when, r);
            if (!l.lends) {
                for (int s = 0; s < l.n; ++s) CHECK(slot_mapped((int) r, s), "%s: region %zu (no lending) slot %d", when, r, s);
                continue;
            }
            int live = 0;   // the floor: live slots beyond the fixed ones, mapped
            for (int s = 0; s < l.n; ++s) live += !l.fixed(s) && slot_mapped((int) r, s) && l.out[(size_t) s] == 0;
            CHECK(live >= lend::kKeep, "%s: region %zu keeps %d slots", when, r, live);
        }
    }
    bool slot_mapped(int r, int s) const {
        return P->mapped((int) ((size_t) (P->slot_ptr(r, s) - P->block_ptr(0)) / P->block_bytes()));
    }
};

uint8_t pattern(int r, int s) { return (uint8_t) ((r * 37 + s * 11) | 1); }

bool slot_is(const VramPool& P, int r, int s, uint8_t v) {
    std::vector<uint8_t> h(P.region(r).slot_bytes);
    if (cudaMemcpy(h.data(), P.slot_ptr(r, s), h.size(), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    for (uint8_t x : h)
        if (x != v) return false;
    return true;
}

// the made-up cache: slots of 12.75 MiB (layers 0, 1 and 2, side by side: they share blocks) and 9.0 MiB (layer 3);
// layer 2 holds every expert (no spare: it cannot lend); layer 0 has two recent slots
std::unique_ptr<FakeSite> make_site(int dev, VramPool& P, bool vmm, std::string& err) {
    const size_t s1 = (size_t) 13369600, s2 = (size_t) 9437440;
    auto site = std::make_unique<FakeSite>();
    site->dev = dev;
    site->L.resize(4);
    const int n[4] = {20, 20, 12, 30};
    std::vector<VramPoolRegion> rg;
    for (int r = 0; r < 4; ++r) {
        FakeLayer& l = site->L[(size_t) r];
        l.n = n[r];
        l.n_exp = r == 2 ? 12 : 48;
        l.lends = r != 2;
        l.spare = r == 2 ? -1 : l.n - 1;
        if (r == 0) l.rc_base = 15, l.rc_n = 2;
        l.slot.assign((size_t) l.n_exp, -1);
        l.heat.assign((size_t) l.n_exp, 0.0f);
        l.out.assign((size_t) l.n, 0);
        for (int e = 0; e < l.n_exp; ++e) l.heat[(size_t) e] = (float) ((e * 7919 + r * 104729) % 1000) / 10.0f;
        for (int s = 0, k = 0; s < l.n; ++s)   // (every slot but the spare holds an expert)
            if (s != l.spare) l.slot[(size_t) (k++ * 7 % l.n_exp)] = s;
        rg.push_back({r == 3 ? s2 : s1, l.n});
    }
    VramPool::Options o;
    o.lend = vmm;
    o.tag = vmm ? "ds4 lend test" : "ds4 lend test (plain)";
    if (!P.create(dev, rg, o, err)) return nullptr;
    site->P = &P;
    return site;
}

void test_lender(int dev) {
    VramPool P;
    std::string err;
    auto own = make_site(dev, P, true, err);
    if (own == nullptr) {
        CHECK(false, "pool: %s", err.c_str());
        return;
    }
    std::printf("%s\n", P.summary().c_str());
    CHECK(P.lendable(), "the pool is not lendable on CUDA%d", dev);
    if (!P.lendable()) return;
    FakeSite* S = own.get();
    lend::Lender lender(std::move(own));
    const size_t block = P.block_bytes(), block_mib = block / kMiB;

    // the slots' data, to see the ones a lend leaves keep it
    for (int r = 0; r < P.region_count(); ++r)
        for (int s = 0; s < P.region(r).n_slots; ++s) cudaMemset(P.slot_ptr(r, s), pattern(r, s), P.region(r).slot_bytes);
    cudaDeviceSynchronize();

    std::vector<strata::core::LendCard> cards = lender.lend_cards();
    CHECK(cards.size() == 1 && cards[0].dev == dev, "lend_cards: %zu cards", cards.size());
    if (cards.size() != 1) return;
    CHECK(cards[0].block_mib == block_mib, "block %zu MiB, pool's %zu", cards[0].block_mib, block_mib);
    const size_t lendable0 = cards[0].lendable_mib;
    {
        std::vector<lend::Block> B;
        int nr = 0;
        S->view(dev, B, nr);
        const std::vector<int> c = lend::candidates(B, nr, lend::kKeep);
        CHECK(lendable0 == c.size() * block_mib, "lendable %zu MiB, %zu candidates of %zu MiB", lendable0, c.size(),
              block_mib);
        int foreign = 0;
        for (const lend::Block& b : B) foreign += b.foreign;
        CHECK(foreign > 0, "no block holds the layer that cannot lend");
        std::printf("pool: %d blocks of %zu MiB, %d foreign; lendable %zu MiB\n", P.block_count(), block_mib, foreign,
                    lendable0);
    }
    CHECK(lendable0 >= 256, "lendable %zu MiB: too little to test", lendable0);
    S->check_tables("before");

    // a lend of 256 MiB, rounded up to whole blocks
    const size_t ask = (256 + block_mib - 1) / block_mib * block_mib;
    const size_t f0 = free_bytes();
    const strata::core::LendResult r = lender.lend(dev, ask);
    const size_t f1 = free_bytes();
    CHECK(r.ok && r.mib == ask && r.token != 0, "lend %zu: ok %d mib %zu token %llu", ask, (int) r.ok, r.mib,
          (unsigned long long) r.token);
    CHECK(S->quiesced >= 1 && S->taken == 1, "quiesced %d, taken %d", S->quiesced, S->taken);
    CHECK(f1 >= f0 + ask * kMiB, "free rose %zu MiB for a lease of %zu", (f1 - f0) / kMiB, ask);
    std::printf("lend %zu MiB: ok, lease %zu MiB, free %zu -> %zu MiB (+%zu)\n", ask, r.mib, f0 / kMiB, f1 / kMiB,
                (f1 - f0) / kMiB);
    S->check_tables("after the lend");
    {
        int kept = 0, lost = 0;
        for (int rr = 0; rr < P.region_count(); ++rr)
            for (int s = 0; s < P.region(rr).n_slots; ++s)
                if (S->slot_mapped(rr, s)) (slot_is(P, rr, s, pattern(rr, s)) ? kept : lost)++;
        CHECK(lost == 0, "%d slots a lend left lost their data", lost);
        std::printf("the slots left: %d keep their data\n", kept);
    }
    cards = lender.lend_cards();
    CHECK(cards[0].lendable_mib == lendable0 - r.mib, "lendable after the lend %zu MiB, want %zu",
          cards[0].lendable_mib, lendable0 - r.mib);

    // more than the card can lend now: refused with what it can
    const size_t now = cards[0].lendable_mib;
    const strata::core::LendResult big = lender.lend(dev, now + block_mib);
    CHECK(!big.ok && big.max_mib == now, "a lend past lendable: ok %d max %zu, want %zu", (int) big.ok, big.max_mib, now);
    CHECK(S->taken == 1, "a refused lend took blocks");

    // a return while another allocation holds the memory: busy, the lease stays
    void* hog = nullptr;
    {
        const size_t f = free_bytes();
        for (size_t want = f > block / 2 ? f - block / 2 : 0; want > r.mib * kMiB; want -= 8 * kMiB) {
            if (cudaMalloc(&hog, want) == cudaSuccess) break;
            cudaGetLastError();
            hog = nullptr;
        }
        CHECK(hog != nullptr, "could not hold the card's memory");
    }
    if (hog != nullptr) {
        CHECK(lender.give_back(dev, r.token) == strata::core::GiveBack::busy, "a return while held is not busy");
        CHECK(lender.lend_cards()[0].lendable_mib == now, "a busy return changed lendable");
        S->check_tables("after a busy return");
        std::printf("return while held: busy\n");
        cudaFree(hog);
    }
    const size_t f2 = free_bytes();
    CHECK(lender.give_back(dev, r.token) == strata::core::GiveBack::returned, "the return");
    const size_t f3 = free_bytes();
    CHECK(f2 >= f3 + r.mib * kMiB - 2 * P.granularity(), "free fell %zu MiB at the return of %zu", (f2 - f3) / kMiB,
          r.mib);
    CHECK(lender.lend_cards()[0].lendable_mib == lendable0, "lendable after the return %zu, before %zu",
          lender.lend_cards()[0].lendable_mib, lendable0);
    S->check_tables("after the return");
    {
        size_t nfree = 0;
        for (const FakeLayer& l : S->L) nfree += l.free.size();
        CHECK(nfree > 0, "no slot came back empty");
        const int filled = S->fill_all();
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError() == cudaSuccess, "writing the returned slots");
        CHECK((size_t) filled == nfree, "filled %d of %zu", filled, nfree);
        int ok = 0;
        for (size_t rr = 0; rr < S->L.size(); ++rr)
            for (int e = 0; e < S->L[rr].n_exp; ++e) {
                const int s = S->L[rr].slot[(size_t) e];
                if (s >= 0 && slot_is(P, (int) rr, s, 0x5a)) ++ok;
            }
        CHECK(ok == filled, "%d of %d filled slots read back", ok, filled);
        std::printf("return: ok, %zu slots back empty, filled and read back\n", nfree);
    }
    S->check_tables("after the fill");

    // the whole lendable at once, then back
    const size_t all = lender.lend_cards()[0].lendable_mib;
    const size_t g0 = free_bytes();
    const strata::core::LendResult a = lender.lend(dev, all);
    const size_t g1 = free_bytes();
    CHECK(a.ok && a.mib == all, "lend all %zu: ok %d mib %zu", all, (int) a.ok, a.mib);
    CHECK(g1 >= g0 + all * kMiB, "free rose %zu MiB for %zu", (g1 - g0) / kMiB, all);
    CHECK(lender.lend_cards()[0].lendable_mib == 0, "lendable after lending it all: %zu",
          lender.lend_cards()[0].lendable_mib);
    S->check_tables("after lending it all");
    CHECK(!lender.lend(dev, block_mib).ok, "a lend with nothing lendable");
    CHECK(lender.give_back(dev, a.token) == strata::core::GiveBack::returned, "the return of it all");
    CHECK(lender.lend_cards()[0].lendable_mib == all, "lendable after it all came back");
    std::printf("lend all %zu MiB: free +%zu MiB, then back\n", all, (g1 - g0) / kMiB);

    // no such card
    CHECK(!lender.lend(dev + 100, block_mib).ok, "a lend on a card with no pool");
}

// the steady layout (steady_blocks) in a real pool: blocks of 16 MiB, slots of 8 MiB.  Region 0 holds its hottest
// expert in slot 4 (block 2), and one hotter than all waits outside VRAM; the lend of the whole lendable, its return
// and the fill (that expert into block 1) leave lendable where it was
void test_steady_lender(int dev) {
    VramPool P;
    std::string err;
    auto own = std::make_unique<FakeSite>();
    own->dev = dev;
    own->L = {steady_layer(0), steady_layer(1)};
    VramPool::Options o;
    o.lend = true;
    o.block_bytes = 16 * kMiB;
    o.tag = "ds4 lend test (steady)";
    if (!P.create(dev, {{8 * kMiB, 5}, {8 * kMiB, 5}}, o, err)) {
        CHECK(false, "steady pool: %s", err.c_str());
        return;
    }
    if (!P.lendable()) return;
    CHECK(P.block_count() == 5 && P.block_bytes() == 16 * kMiB, "steady pool: %d blocks of %zu MiB", P.block_count(),
          P.block_bytes() / kMiB);
    own->P = &P;
    FakeSite* S = own.get();
    lend::Lender lender(std::move(own));
    const size_t l0 = lender.lend_cards()[0].lendable_mib;
    CHECK(l0 == 32, "steady lendable %zu MiB, want 2 blocks", l0);
    const strata::core::LendResult a = lender.lend(dev, l0);
    CHECK(a.ok && a.mib == l0, "lend the lendable %zu: ok %d mib %zu", l0, (int) a.ok, a.mib);
    S->check_tables("steady: after the lend");
    CHECK(lender.give_back(dev, a.token) == strata::core::GiveBack::returned, "steady: the return");
    const size_t l1 = lender.lend_cards()[0].lendable_mib;
    CHECK(l1 == l0, "steady: lendable after the return %zu, before %zu", l1, l0);
    S->fill_all();
    cudaDeviceSynchronize();
    const size_t l2 = lender.lend_cards()[0].lendable_mib;
    CHECK(l2 == l0, "steady: lendable after the fill %zu, before %zu", l2, l0);
    const strata::core::LendResult b = lender.lend(dev, l2);
    CHECK(b.ok && b.mib == l2, "steady: lend the lendable again %zu: ok %d mib %zu", l2, (int) b.ok, b.mib);
    if (b.ok) CHECK(lender.give_back(dev, b.token) == strata::core::GiveBack::returned, "steady: the second return");
    std::printf("steady lender: lendable %zu MiB, %zu after the return, %zu after the fill\n", l0, l1, l2);
}

// the steady layout in a real pool with the spares moving: lendable the same wherever they are; region 0's spare in a
// block a lend takes (slot 2, block 1) moves out of it, and a lend of the lendable just read succeeds; swaps during a
// lease leave what is left to lend as it was; after the return, lendable as before
void test_spare_lender(int dev) {
    VramPool P;
    std::string err;
    auto own = std::make_unique<FakeSite>();
    own->dev = dev;
    own->L = {steady_layer(0), steady_layer(1)};
    VramPool::Options o;
    o.lend = true;
    o.block_bytes = 16 * kMiB;
    o.tag = "ds4 lend test (spare)";
    if (!P.create(dev, {{8 * kMiB, 5}, {8 * kMiB, 5}}, o, err)) {
        CHECK(false, "spare pool: %s", err.c_str());
        return;
    }
    if (!P.lendable()) return;
    own->P = &P;
    FakeSite* S = own.get();
    lend::Lender lender(std::move(own));
    const size_t block_mib = P.block_bytes() / kMiB;
    auto lendable = [&] { return lender.lend_cards()[0].lendable_mib; };
    const size_t l0 = lendable();
    for (int p0 : {0, 2, 3, 4})
        for (int p1 : {0, 1, 2, 4}) {
            S->L[0].spare_to(p0);
            S->L[1].spare_to(p1);
            CHECK(lendable() == l0, "spare: the spares in slots %d / %d: lendable %zu, %zu with them in 0 / 4", p0, p1,
                  lendable(), l0);
        }
    S->L[0].spare_to(2);
    S->L[1].spare_to(4);
    const size_t l1 = lendable();
    const strata::core::LendResult a = lender.lend(dev, l1);
    CHECK(a.ok && a.mib == l1, "spare: lend the lendable %zu with the spare in a candidate: ok %d mib %zu", l1,
          (int) a.ok, a.mib);
    CHECK(S->spare_moves >= 1, "spare: region 0's spare (slot %d) did not move", S->L[0].spare);
    S->check_tables("spare: after the lend");
    CHECK(lender.give_back(dev, a.token) == strata::core::GiveBack::returned, "spare: the return");
    S->fill_all();
    CHECK(lendable() == l0, "spare: lendable after the return %zu, before %zu", lendable(), l0);

    // a lend of one block, swaps in both regions while it is out, then a lend of the rest as it reads then
    const strata::core::LendResult b = lender.lend(dev, block_mib);
    CHECK(b.ok, "spare: lend one block");
    const size_t left = lendable();
    for (int k = 0; k < 6; ++k) {
        FakeLayer& l = S->L[(size_t) (k % 2)];
        const int in = l.cpu_expert();
        l.heat[(size_t) in] = 50.0f + (float) k;
        l.swap(in);
        CHECK(lendable() == left, "spare: swap %d during a lease (spares in %d / %d): lendable %zu, %zu before", k,
              S->L[0].spare, S->L[1].spare, lendable(), left);
    }
    const size_t now = lendable();
    const strata::core::LendResult c = lender.lend(dev, now);
    CHECK(now == 0 || (c.ok && c.mib == now), "spare: lend the rest %zu after the swaps: ok %d mib %zu", now,
          (int) c.ok, c.mib);
    S->check_tables("spare: after the swaps and the second lend");
    if (c.ok) CHECK(lender.give_back(dev, c.token) == strata::core::GiveBack::returned, "spare: the second return");
    if (b.ok) CHECK(lender.give_back(dev, b.token) == strata::core::GiveBack::returned, "spare: the first return");
    S->fill_all();
    S->L[0].swap(S->L[0].cpu_expert());
    CHECK(lendable() == l0, "spare: lendable after both returns, the fill and a swap %zu, before %zu", lendable(), l0);
    std::printf("spare lender: lendable %zu MiB wherever the spares are; %zu, then %zu during a lease with swaps\n", l0,
                left, now);
}

void test_fallback(int dev) {
    VramPool P;
    std::string err;
    auto own = make_site(dev, P, false, err);
    CHECK(own != nullptr, "plain pool: %s", err.c_str());
    if (own == nullptr) return;
    lend::Lender lender(std::move(own));
    const std::vector<strata::core::LendCard> cards = lender.lend_cards();
    CHECK(cards.size() == 1 && cards[0].lendable_mib == 0 && cards[0].block_mib == 0, "the plain pool lends");
    const strata::core::LendResult r = lender.lend(dev, 64);
    CHECK(!r.ok && r.max_mib == 0, "a lend on the plain pool");
    std::printf("plain pool: lends 0\n");
}

}  // namespace

int main() {
    test_choice();
    test_steady();
    test_spare();
    test_slots();
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("ds4_lend_test: no GPU, the lender's part skipped\n");
    } else {
        int dev = 0;
        if (const char* e = std::getenv("STRATA_TEST_DEV")) dev = std::atoi(e);
        if (dev < 0 || dev >= n) {
            std::printf("ds4_lend_test: no device %d (%d devices)\n", dev, n);
            return 1;
        }
        cudaSetDevice(dev);
        std::printf("ds4_lend_test: CUDA%d, %zu MiB free\n", dev, free_bytes() / kMiB);
        test_fallback(dev);
        test_lender(dev);
        test_steady_lender(dev);
        test_spare_lender(dev);
    }
    if (g_fail == 0) std::printf("ds4_lend_test: ok\n");
    return g_fail == 0 ? 0 : 1;
}
