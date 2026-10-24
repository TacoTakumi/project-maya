// tests/glm_lend_test.cu - lending the GLM expert pool's main-slot blocks (src/core/glm_lend.cu), without a model:
//  - lend_choose: the coldest blocks first, spread over the layers on equal worth, the later block on a tie, never a
//    layer below its minimum (also through a block two layers share), fewer when no more can go;
//  - lend_capacity: the lendable rule, capped at the blocks the choice can take;
//  - on a GPU (STRATA_TEST_DEV, default 0; skipped without one): a synthetic part - two layers of main slots on a real
//    VMM pool sharing a block, the device slot table, a RAM tier of two slots - lends two blocks: the lent slots'
//    experts move over colder ones (bytes and table entries checked), the leaving ones go to RAM hottest first, the
//    rest to disk, a spare leaves the spare table, every lent slot is kLent and in no table; the blocks unmap (the
//    card's free memory rises by them), map back and the slots are free; a slot still draining refuses the lend and
//    changes nothing;
//  - the RAM tier's share of a lend (lend_park_budget, STRATA_GLM_LEND_PARK_MB): with a budget of one slot only the
//    hottest leaving expert is copied to RAM, with none every one goes to disk, the moves the same either way;
//  - the lender (make_glm_lender) and the lend commands over two such parts behind Glm5Model shells, a split on one
//    card (and with STRATA_TEST_DEV2=<n> a second run with the second part on card n): each card's lendable, a lend
//    whose blocks free the card's memory (on one card shared by the parts), a return refused as busy while another
//    allocation holds the memory and then accepted, the tables consistent throughout, the other card unchanged.
#include "glm_fast_state.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using strata::core::Glm5Model;
using strata::core::VramPool;
using strata::core::VramPoolRegion;
namespace glmfast = strata::core::glmfast;
namespace gf = strata::kernels::glmf;
using FastState = Glm5Model::FastState;

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

std::string str(const std::vector<int>& v) {
    std::string s = "{";
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
    return s + "}";
}

glmfast::LendBlock blk(int b, std::vector<std::pair<int, int>> spans) {
    glmfast::LendBlock x;
    x.block = b;
    x.spans = std::move(spans);
    return x;
}

// ---- the block choice and the lendable rule (host only)
void test_choose() {
    // two layers of 16 slots, four slots a block, blocks 0-3 layer 0's and 4-7 layer 1's; each keeps 8
    std::vector<glmfast::LendBlock> cand;
    for (int b = 0; b < 8; ++b) cand.push_back(blk(b, {{b / 4, 4}}));
    const std::vector<int> have{16, 16}, keep{8, 8};
    std::vector<std::vector<uint64_t>> flat(2, std::vector<uint64_t>(16, 0));

    // as many as can go: two blocks a layer (16 - 8 = 8 slots each)
    auto r = glmfast::lend_choose(cand, have, keep, flat, -1);
    CHECK(r.size() == 4, "all: %s", str(r).c_str());
    // equal worth: spread over the layers, the later block first (a region's end)
    r = glmfast::lend_choose(cand, have, keep, flat, 2);
    CHECK(r == (std::vector<int>{7, 3}), "spread: %s", str(r).c_str());
    r = glmfast::lend_choose(cand, have, keep, flat, 4);
    CHECK(r == (std::vector<int>{7, 3, 6, 2}), "spread 4: %s", str(r).c_str());
    // more than can go: what can
    r = glmfast::lend_choose(cand, have, keep, flat, 6);
    CHECK(r.size() == 4, "capped: %s", str(r).c_str());
    // the coldest first: layer 0 cold, layer 1 hot - both of layer 0's blocks go before any of layer 1's
    std::vector<std::vector<uint64_t>> heat(2);
    for (int i = 0; i < 16; ++i) {
        heat[0].push_back(1 + (uint64_t) i);
        heat[1].push_back(100 + (uint64_t) i);
    }
    r = glmfast::lend_choose(cand, have, keep, heat, 3);
    CHECK(r == (std::vector<int>{3, 2, 7}), "cold first: %s", str(r).c_str());
    // a layer at its minimum lends nothing; one below it neither
    r = glmfast::lend_choose(cand, {8, 16}, keep, flat, -1);
    CHECK(r == (std::vector<int>{7, 6}), "at min: %s", str(r).c_str());
    r = glmfast::lend_choose(cand, {5, 9}, keep, flat, -1);
    CHECK(r.empty(), "below min: %s", str(r).c_str());
    // empty slots are worth nothing: a layer whose coldest 4 are empty goes first even if its other slots are hot
    std::vector<std::vector<uint64_t>> h2(2);
    for (int i = 0; i < 16; ++i) {
        h2[0].push_back(i < 4 ? 0 : 1000);
        h2[1].push_back(10);
    }
    r = glmfast::lend_choose(cand, have, keep, h2, 2);
    CHECK(r == (std::vector<int>{3, 7}), "empty first: %s", str(r).c_str());

    // a block two layers share: layer 0 slots in blocks 0, 1 and half of 2; layer 1 the rest of 2, then 3, 4
    // (ten slots each, four a block).  Each keeps 6: four slots of each may go.
    std::vector<glmfast::LendBlock> sh{blk(0, {{0, 4}}), blk(1, {{0, 4}}), blk(2, {{0, 2}, {1, 2}}), blk(3, {{1, 4}}),
                                       blk(4, {{1, 4}})};
    const std::vector<int> h10{10, 10}, k6{6, 6};
    std::vector<std::vector<uint64_t>> f10(2, std::vector<uint64_t>(10, 0));
    r = glmfast::lend_choose(sh, h10, k6, f10, -1);
    // a whole block of each layer (a shared block goes last on equal worth: it would stop both layers' whole blocks)
    {
        int t0 = 0, t1 = 0;
        for (int b : r)
            for (auto [l, c] : sh[(size_t) b].spans) (l == 0 ? t0 : t1) += c;
        CHECK(t0 == 4 && t1 == 4 && r == (std::vector<int>{4, 1}), "shared: %s (%d, %d)", str(r).c_str(), t0, t1);
    }
    // the shared block alone, then neither layer can give a whole other block
    r = glmfast::lend_choose({sh[2], sh[1], sh[3]}, {6 + 2 + 1, 6 + 2 + 1}, k6, f10, -1);
    CHECK(r == (std::vector<int>{2}), "shared only: %s", str(r).c_str());
    // a span naming no layer is never chosen
    r = glmfast::lend_choose({blk(9, {{5, 1}})}, have, keep, flat, -1);
    CHECK(r.empty(), "bad layer: %s", str(r).c_str());

    // the lendable rule capped at the blocks the choice can take
    CHECK(glmfast::lend_capacity(1000 * kMiB, 100 * kMiB, 300 * kMiB, 64 * kMiB, 20) == 576 * kMiB, "rule");
    CHECK(glmfast::lend_capacity(1000 * kMiB, 100 * kMiB, 300 * kMiB, 64 * kMiB, 5) == 320 * kMiB, "cap");
    CHECK(glmfast::lend_capacity(1000 * kMiB, 100 * kMiB, 300 * kMiB, 0, 5) == 0, "no blocks");
    CHECK(glmfast::lend_capacity(300 * kMiB, 100 * kMiB, 300 * kMiB, 64 * kMiB, 5) == 0, "under the minimum");
    CHECK(glmfast::lend_capacity(1000 * kMiB, 100 * kMiB, 300 * kMiB, 64 * kMiB, 0) == 0, "no block can go");
}

// ---- the RAM tier's share of a lend: STRATA_GLM_LEND_PARK_MB, 512 MiB by default (host only)
void test_park_budget() {
    unsetenv("STRATA_GLM_LEND_PARK_MB");
    CHECK(glmfast::lend_park_budget() == 512 * kMiB, "default %zu", glmfast::lend_park_budget());
    setenv("STRATA_GLM_LEND_PARK_MB", "0", 1);
    CHECK(glmfast::lend_park_budget() == 0, "0: every leaving expert to disk, %zu", glmfast::lend_park_budget());
    setenv("STRATA_GLM_LEND_PARK_MB", "2048", 1);
    CHECK(glmfast::lend_park_budget() == 2048 * kMiB, "2048: %zu", glmfast::lend_park_budget());
    for (const char* bad : {"-1", "x", "", "12x"}) {
        setenv("STRATA_GLM_LEND_PARK_MB", bad, 1);
        CHECK(glmfast::lend_park_budget() == 512 * kMiB, "'%s' is not a size: the default, %zu", bad,
              glmfast::lend_park_budget());
    }
    unsetenv("STRATA_GLM_LEND_PARK_MB");
}

// ---- a synthetic part on a real pool
constexpr int NE = 16, NL = 2;
constexpr size_t kStride = 4 * kMiB, kBlock = 16 * kMiB;   // four slots a block
constexpr int kMain = 10;                                   // slots a layer: layer 1 starts in layer 0's last block

uint8_t tag(int il, int e) { return (uint8_t) (1 + il * 32 + e); }

struct Part {
    std::unique_ptr<FastState> F = std::make_unique<FastState>();
    int dev = 0;
    ~Part() {
        FastState& f = *F;
        cudaSetDevice(dev);
        if (f.cs) cudaStreamSynchronize(f.cs);
        f.vpool.destroy();
        cudaFree(f.tab);
        cudaFree(f.upd_key_d);
        cudaFree(f.upd_val_d);
        cudaFreeHost(f.upd_key_h);
        cudaFreeHost(f.upd_val_h);
        if (!f.rc.empty()) cudaFreeHost(f.rc[0].base);
        if (f.cs) cudaStreamDestroy(f.cs);
        if (f.copy) cudaStreamDestroy(f.copy);
    }
};

// part layers [l0, l0 + NL) of nlt, on card dev
bool setup(Part& p, int dev, std::string& err, int l0 = 0, int nlt = NL) {
    FastState& F = *p.F;
    cudaSetDevice(dev);   // (the part's streams and tables on its card)
    p.dev = dev;
    VramPool::Options o;
    o.block_bytes = kBlock;
    o.tag = "glm lend test";
    o.log = false;
    if (!F.vpool.create(dev, {{kStride, kMain}, {kStride, kMain}}, o, err)) return false;
    if (!F.vpool.lendable()) {
        err = "the pool is not lendable here (no VMM)";
        return false;
    }
    cudaStreamCreate(&F.cs);
    cudaStreamCreate(&F.copy);
    F.n_keys = nlt * NE;
    const size_t nt = 2 * (size_t) F.n_keys + (size_t) nlt * gf::kSpares;
    cudaMalloc(&F.tab, nt * sizeof(unsigned long long));
    cudaMemset(F.tab, 0, nt * sizeof(unsigned long long));
    cudaMallocHost(&F.upd_key_h, FastState::kMaxUpd * sizeof(int));
    cudaMallocHost(&F.upd_val_h, FastState::kMaxUpd * sizeof(unsigned long long));
    cudaMalloc(&F.upd_key_d, FastState::kMaxUpd * sizeof(int));
    cudaMalloc(&F.upd_val_d, FastState::kMaxUpd * sizeof(unsigned long long));
    F.upd_at.assign(nt, 0);
    F.upd_gen.assign(nt, 0);
    F.L.resize((size_t) nlt);
    F.lp.resize((size_t) nlt);
    F.pool_layer = {l0, l0 + 1};
    for (int r = 0; r < NL; ++r) {
        const int il = l0 + r;
        F.L[(size_t) il].moe = true;
        F.L[(size_t) il].blob = kStride;
        auto& P = F.lp[(size_t) il];
        const auto& R = F.vpool.region(r);
        P.stride = kStride;
        P.n = P.n_main = kMain;
        P.origin = R.origin;
        P.block = F.vpool.block_bytes();
        P.per_block = R.per_block;
        P.first_pos = R.first_pos;
        P.region = r;
        P.key.assign(kMain, -1);
        P.tick.assign(kMain, 0);
        P.st.assign(kMain, FastState::kFree);
    }
    F.slot_of.assign((size_t) nlt * NE, -1);
    F.cnt.assign((size_t) nlt * NE, 0);
    F.usage.assign((size_t) nlt * NE, 0);
    F.left.assign((size_t) nlt * NE, 0);
    F.ram_of.assign((size_t) nlt * NE, -1);
    // one RAM class of two free slots, both layers'
    F.rc.resize(1);
    auto& RC = F.rc[0];
    RC.stride = kStride;
    RC.n = 2;
    cudaMallocHost(&RC.base, 2 * kStride);
    RC.key.assign(2, -1);
    RC.tick.assign(2, 0);
    RC.st.assign(2, FastState::kRFree);
    F.layer_rc.assign((size_t) nlt, 0);
    F.clock = 100;
    // the first layer: experts 0-8 in slots 0-8 (count 10 e, hotter up), slot 9 the device's spare 0;
    // the second: experts 0-9 in slots 0-9 (count 100 + e) but slot 8, which is free
    std::vector<unsigned long long> th(nt, 0ull);
    for (int r = 0; r < NL; ++r) {
        const int il = l0 + r;
        auto& P = F.lp[(size_t) il];
        for (int s = 0; s < kMain; ++s) {
            if ((r == 0 && s == 9) || (r == 1 && s == 8)) continue;
            const int e = s, key = il * NE + e;
            P.key[(size_t) s] = e;
            P.st[(size_t) s] = FastState::kResident;
            P.tick[(size_t) s] = (uint64_t) (10 + s);
            F.slot_of[(size_t) key] = s;
            F.cnt[(size_t) key] = r == 0 ? (uint32_t) (10 * e) : (uint32_t) (100 + e);
            th[(size_t) key] = (unsigned long long) P.slot_ptr(s);
            cudaMemset(P.slot_ptr(s), tag(il, e), kStride);
        }
    }
    auto& P0 = F.lp[(size_t) l0];
    P0.st[9] = FastState::kSpare;
    P0.spare[0] = 9;
    th[F.spare_key(l0, 0)] = (unsigned long long) P0.slot_ptr(9);
    cudaMemcpy(F.tab, th.data(), nt * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    // the table kernel once (an entry to its own value), as a running engine has it loaded: its first launch on a
    // card takes some of that card's memory, which the lend's free-memory checks would see
    F.upd_key_h[0] = 0;
    F.upd_val_h[0] = th[0];
    cudaMemcpy(F.upd_key_d, F.upd_key_h, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(F.upd_val_d, F.upd_val_h, sizeof(unsigned long long), cudaMemcpyHostToDevice);
    gf::tab_update(F.tab, F.upd_key_d, F.upd_val_d, 1, F.cs);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        err = cudaGetErrorString(cudaGetLastError());
        return false;
    }
    return true;
}

// a slot's first and last bytes
bool slot_has(const uint8_t* p, uint8_t v) {
    uint8_t a = 0, b = 0;
    cudaMemcpy(&a, p, 1, cudaMemcpyDeviceToHost);
    cudaMemcpy(&b, p + kStride - 1, 1, cudaMemcpyDeviceToHost);
    return a == v && b == v;
}

int block_of(const FastState& F, const void* p) {
    return (int) (((const uint8_t*) p - F.vpool.block_ptr(0)) / F.vpool.block_bytes());
}

void test_part(int dev) {
    cudaSetDevice(dev);
    Part part;
    std::string err;
    if (!setup(part, dev, err)) {
        std::printf("SKIP the GPU part: %s\n", err.c_str());
        return;
    }
    FastState& F = *part.F;
    CHECK(F.vpool.block_count() == 5, "blocks %d", F.vpool.block_count());
    CHECK(F.lp[1].first_pos == 2, "layer 1 starts at position %d of layer 0's last block", F.lp[1].first_pos);

    // the view: five mapped blocks, block 2 shared (two slots each)
    glmfast::LendView v = glmfast::lend_view(F, 0, NL, NE, 6);
    CHECK(v.cand.size() == 5 && v.have == (std::vector<int>{10, 10}) && v.keep == (std::vector<int>{6, 6}),
          "view: %zu blocks, have %s", v.cand.size(), str(v.have).c_str());
    CHECK(v.cand.size() == 5 && v.cand[2].spans.size() == 2, "the shared block's spans");
    CHECK(v.min_bytes == 2 * 6 * kBlock / 4, "min bytes %zu", v.min_bytes);
    CHECK(v.heat.size() == 2 && v.heat[0].size() == 10 && v.heat[1].size() == 10 && v.heat[0][0] == 0 && v.heat[0][1] == 1 && v.heat[1][0] == 0 && v.heat[1][1] == 101,
          "heat");
    // layer 0 is the colder: its last whole block first, then (layer 0 at its minimum) layer 1's last
    const std::vector<int> blocks = glmfast::lend_choose(v.cand, v.have, v.keep, v.heat, 2);
    CHECK(blocks == (std::vector<int>{1, 4}), "chosen %s", str(blocks).c_str());
    CHECK(glmfast::lend_choose(v.cand, v.have, v.keep, v.heat, -1).size() == 2, "at most two blocks");

    // a slot still draining in the set: refused, nothing changed
    F.lp[0].st[5] = FastState::kDraining;
    glmfast::LendCounts c0;
    CHECK(!glmfast::lend_take(F, 0, NL, NE, blocks, c0, err), "a draining slot must refuse the lend");
    CHECK(F.lp[0].st[4] == FastState::kResident && F.slot_of[4] == 4 && F.lp[0].n_lent == 0, "refused: unchanged");
    F.lp[0].st[5] = FastState::kResident;

    std::vector<uint8_t*> ptr0(kMain), ptr1(kMain);
    for (int s = 0; s < kMain; ++s) {
        ptr0[(size_t) s] = F.lp[0].slot_ptr(s);
        ptr1[(size_t) s] = F.lp[1].slot_ptr(s);
    }
    glmfast::LendCounts c;
    err.clear();
    CHECK(glmfast::lend_take(F, 0, NL, NE, blocks, c, err), "lend_take: %s", err.c_str());
    // layer 0 lends slots 4-7 (experts 4-7): 7 into the spare's slot 9, 6 / 5 / 4 over experts 0 / 1 / 2.  Layer 1
    // lends slots 6-9 (experts 6, 7, 9; slot 8 empty): 9 / 7 / 6 over experts 0 / 1 / 2.  Of the six that leave VRAM
    // the hottest two - layer 1's experts 2 and 1 (counts 102, 101) - take the RAM tier's two slots, the other four go
    // to disk (the RAM tier is full of this lend's own experts)
    CHECK(c.slots == 8 && c.moved == 7 && c.parked == 2 && c.dropped == 4, "counts: slots %d moved %llu parked %llu "
          "dropped %llu", c.slots, (unsigned long long) c.moved, (unsigned long long) c.parked,
          (unsigned long long) c.dropped);
    const auto& P0 = F.lp[0];
    const auto& P1 = F.lp[1];
    const int want0[kMain] = {6, 5, 4, 3, -1, -1, -1, -1, 8, 7};
    const int want1[kMain] = {9, 7, 6, 3, 4, 5, -1, -1, -1, -1};
    for (int s = 0; s < kMain; ++s) {
        CHECK(P0.key[(size_t) s] == want0[s], "layer 0 slot %d holds %d", s, P0.key[(size_t) s]);
        CHECK(P1.key[(size_t) s] == want1[s], "layer 1 slot %d holds %d", s, P1.key[(size_t) s]);
        const bool lent0 = s >= 4 && s < 8, lent1 = s >= 6;
        CHECK((P0.st[(size_t) s] == FastState::kLent) == lent0, "layer 0 slot %d state %d", s, P0.st[(size_t) s]);
        CHECK((P1.st[(size_t) s] == FastState::kLent) == lent1, "layer 1 slot %d state %d", s, P1.st[(size_t) s]);
    }
    CHECK(P0.n_lent == 4 && P1.n_lent == 4, "n_lent %d %d", P0.n_lent, P1.n_lent);
    CHECK(P0.spare[0] == -1, "the spare slot 9 took expert 7: no longer a spare");
    // the host tables: every expert where its slot says, none in a lent slot
    for (int il = 0; il < NL; ++il)
        for (int e = 0; e < NE; ++e) {
            const int s = F.slot_of[(size_t) (il * NE + e)];
            const auto& P = F.lp[(size_t) il];
            if (s >= 0) {
                CHECK(P.key[(size_t) s] == e && P.st[(size_t) s] == FastState::kResident, "layer %d expert %d slot %d",
                      il, e, s);
                CHECK(F.vpool.mapped(block_of(F, P.slot_ptr(s))), "layer %d expert %d in a lent block", il, e);
            }
        }
    CHECK(F.slot_of[0] < 0 && F.slot_of[1] < 0 && F.slot_of[2] < 0, "layer 0 experts 0-2 left VRAM");
    CHECK(F.slot_of[NE + 0] < 0 && F.slot_of[NE + 1] < 0 && F.slot_of[NE + 2] < 0, "layer 1 experts 0-2 left VRAM");
    CHECK(F.ram_of[NE + 2] >= 0 && F.ram_of[NE + 1] >= 0 && F.ram_of[0] < 0 && F.ram_of[1] < 0 && F.ram_of[2] < 0 &&
              F.ram_of[NE] < 0,
          "the RAM tier: %d %d | %d %d %d %d", F.ram_of[NE + 2], F.ram_of[NE + 1], F.ram_of[0], F.ram_of[1],
          F.ram_of[2], F.ram_of[NE]);
    CHECK(F.left[0] == 3 && F.left[1] == 3 && F.left[2] == 3 && F.left[NE] == 3, "dropped experts' cause");
    CHECK(c.park_bytes == 2 * kStride, "parked bytes %zu", c.park_bytes);
    // the device table as the routes will read it
    {
        const size_t nt = 2 * (size_t) F.n_keys + (size_t) NL * gf::kSpares;
        std::vector<unsigned long long> th(nt);
        cudaMemcpy(th.data(), F.tab, nt * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        for (int il = 0; il < NL; ++il)
            for (int e = 0; e < NE; ++e) {
                const int key = il * NE + e, s = F.slot_of[(size_t) key];
                const unsigned long long want = s >= 0 ? (unsigned long long) F.lp[(size_t) il].slot_ptr(s) : 0ull;
                CHECK(th[F.tab_key(key)] == want, "tab layer %d expert %d", il, e);
                const int rs = F.ram_of[(size_t) key];
                const unsigned long long rw = rs >= 0 ? (unsigned long long) F.rc[0].slot((size_t) rs) : 0ull;
                CHECK(th[F.rtab_key(key)] == rw, "rtab layer %d expert %d", il, e);
            }
        CHECK(th[F.spare_key(0, 0)] == 0ull, "the spare table entry");
    }
    // the bytes: moved experts in their new slots, parked ones in RAM
    for (int s = 0; s < kMain; ++s) {
        if (want0[s] >= 0) CHECK(slot_has(ptr0[(size_t) s], tag(0, want0[s])), "layer 0 slot %d bytes", s);
        if (want1[s] >= 0) CHECK(slot_has(ptr1[(size_t) s], tag(1, want1[s])), "layer 1 slot %d bytes", s);
    }
    for (int e = 1; e < 3; ++e) {
        const uint8_t* h = F.rc[0].slot((size_t) F.ram_of[(size_t) (NE + e)]);
        CHECK(h[0] == tag(1, e) && h[kStride - 1] == tag(1, e), "RAM copy of layer 1 expert %d", e);
    }
    // the lent blocks are no candidates any more; the three left cannot go (both layers at their minimum)
    v = glmfast::lend_view(F, 0, NL, NE, 6);
    CHECK(v.cand.size() == 3 && v.have == (std::vector<int>{6, 6}), "view after: %zu, %s", v.cand.size(),
          str(v.have).c_str());
    CHECK(glmfast::lend_choose(v.cand, v.have, v.keep, v.heat, -1).empty(), "nothing more to lend");

    // unmap: the card's free memory rises by the two blocks
    size_t f0 = 0, f1 = 0, tot = 0;
    cudaMemGetInfo(&f0, &tot);
    CHECK(glmfast::lend_unmap(F, dev, blocks, err), "lend_unmap: %s", err.c_str());
    cudaMemGetInfo(&f1, &tot);
    CHECK(!F.vpool.mapped(1) && !F.vpool.mapped(4) && F.vpool.mapped_count() == 3, "mapped %d", F.vpool.mapped_count());
    CHECK(f1 >= f0 + 2 * kBlock, "free %zu -> %zu MiB", f0 >> 20, f1 >> 20);
    // the kept experts still read right
    CHECK(slot_has(ptr0[0], tag(0, 6)) && slot_has(ptr1[5], tag(1, 5)), "kept slots after the unmap");

    // back: mapped at the same addresses, the slots free, writable
    CHECK(F.vpool.map(blocks, err) == VramPool::Result::Ok, "map: %s", err.c_str());
    glmfast::lend_free(F, blocks);
    for (int s = 4; s < 8; ++s) CHECK(F.lp[0].st[(size_t) s] == FastState::kFree, "layer 0 slot %d free", s);
    for (int s = 6; s < kMain; ++s) CHECK(F.lp[1].st[(size_t) s] == FastState::kFree, "layer 1 slot %d free", s);
    CHECK(F.lp[0].n_lent == 0 && F.lp[1].n_lent == 0, "n_lent back to 0");
    CHECK(F.lp[0].slot_ptr(5) == ptr0[5], "same address");
    cudaMemset(F.lp[1].slot_ptr(9), 0x5a, kStride);
    CHECK(slot_has(F.lp[1].slot_ptr(9), 0x5a), "a returned slot is writable");
    v = glmfast::lend_view(F, 0, NL, NE, 6);
    CHECK(v.cand.size() == 5 && v.have == (std::vector<int>{10, 10}), "view after the return");
    CHECK(cudaDeviceSynchronize() == cudaSuccess, "device: %s", cudaGetErrorString(cudaGetLastError()));
}

// the same lend with a park budget: `budget` bytes of the leaving experts to RAM, the hottest first
void test_part_budget(int dev, size_t budget, int want_parked) {
    std::printf("a lend with a park budget of %zu MiB\n", budget >> 20);
    cudaSetDevice(dev);
    Part part;
    std::string err;
    if (!setup(part, dev, err)) {
        std::printf("SKIP the GPU part: %s\n", err.c_str());
        return;
    }
    FastState& F = *part.F;
    const std::vector<int> blocks{1, 4};
    glmfast::LendCounts c;
    CHECK(glmfast::lend_take(F, 0, NL, NE, blocks, c, err, budget), "lend_take: %s", err.c_str());
    CHECK(c.slots == 8 && c.moved == 7 && (int) c.parked == want_parked && (int) c.dropped == 6 - want_parked &&
              c.park_bytes == (size_t) want_parked * kStride,
          "counts: moved %llu parked %llu dropped %llu (%zu bytes)", (unsigned long long) c.moved,
          (unsigned long long) c.parked, (unsigned long long) c.dropped, c.park_bytes);
    // the hottest leaving expert (layer 1's expert 2) is the one in RAM; the rest on disk
    const int hot[2] = {NE + 2, NE + 1};
    for (int i = 0; i < 2; ++i)
        CHECK((F.ram_of[(size_t) hot[i]] >= 0) == (i < want_parked), "key %d in RAM: %d", hot[i], F.ram_of[(size_t) hot[i]]);
    for (int key : {0, 1, 2, NE}) CHECK(F.ram_of[(size_t) key] < 0 && F.left[(size_t) key] == 3, "key %d to disk", key);
    if (want_parked > 0) {
        const uint8_t* h = F.rc[0].slot((size_t) F.ram_of[NE + 2]);
        CHECK(h[0] == tag(1, 2) && h[kStride - 1] == tag(1, 2), "RAM copy of layer 1 expert 2");
    }
    // the moves do not depend on the budget: the same experts in the same slots, their bytes there
    const int want0[kMain] = {6, 5, 4, 3, -1, -1, -1, -1, 8, 7};
    const int want1[kMain] = {9, 7, 6, 3, 4, 5, -1, -1, -1, -1};
    for (int s = 0; s < kMain; ++s) {
        CHECK(F.lp[0].key[(size_t) s] == want0[s] && F.lp[1].key[(size_t) s] == want1[s], "slot %d holds %d, %d", s,
              F.lp[0].key[(size_t) s], F.lp[1].key[(size_t) s]);
        if (want0[s] >= 0) CHECK(slot_has(F.lp[0].slot_ptr(s), tag(0, want0[s])), "layer 0 slot %d bytes", s);
        if (want1[s] >= 0) CHECK(slot_has(F.lp[1].slot_ptr(s), tag(1, want1[s])), "layer 1 slot %d bytes", s);
    }
    {
        const size_t nt = 2 * (size_t) F.n_keys + (size_t) NL * gf::kSpares;
        std::vector<unsigned long long> th(nt);
        cudaMemcpy(th.data(), F.tab, nt * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        for (int key = 0; key < NL * NE; ++key) {
            const int il = key / NE, s = F.slot_of[(size_t) key], rs = F.ram_of[(size_t) key];
            CHECK(th[F.tab_key(key)] == (s >= 0 ? (unsigned long long) F.lp[(size_t) il].slot_ptr(s) : 0ull), "tab %d",
                  key);
            CHECK(th[F.rtab_key(key)] == (rs >= 0 ? (unsigned long long) F.rc[0].slot((size_t) rs) : 0ull), "rtab %d",
                  key);
        }
    }
    CHECK(glmfast::lend_unmap(F, dev, blocks, err), "lend_unmap: %s", err.c_str());
    CHECK(F.vpool.map(blocks, err) == VramPool::Result::Ok, "map: %s", err.c_str());
    glmfast::lend_free(F, blocks);
    CHECK(cudaDeviceSynchronize() == cudaSuccess, "device: %s", cudaGetErrorString(cudaGetLastError()));
}

size_t card_free(int dev) {
    cudaSetDevice(dev);
    size_t f = 0, t = 0;
    cudaMemGetInfo(&f, &t);
    return f;
}

// the parts' tables agree with their pools: no expert in a lent or unmapped slot, the lent slots counted
void check_tables(const FastState& F, int l0, const char* when) {
    for (int r = 0; r < NL; ++r) {
        const int il = l0 + r;
        const auto& P = F.lp[(size_t) il];
        int lent = 0;
        for (int s = 0; s < P.n_main; ++s) {
            const bool mapped = F.vpool.mapped(block_of(F, P.slot_ptr(s)));
            lent += P.st[(size_t) s] == FastState::kLent;
            CHECK(mapped || P.st[(size_t) s] == FastState::kLent, "%s: layer %d slot %d unmapped but not lent", when, il,
                  s);
        }
        CHECK(lent == P.n_lent, "%s: layer %d lent %d, n_lent %d", when, il, lent, P.n_lent);
        for (int e = 0; e < NE; ++e) {
            const int sl = F.slot_of[(size_t) (il * NE + e)];
            if (sl >= 0)
                CHECK(P.st[(size_t) sl] == FastState::kResident && P.key[(size_t) sl] == e &&
                          F.vpool.mapped(block_of(F, P.slot_ptr(sl))),
                      "%s: layer %d expert %d slot %d", when, il, e, sl);
        }
    }
}

// what a command line answers
std::string ask(strata::core::LendCommands& c, const std::string& line) {
    std::string r;
    c.handle(line, r);
    std::printf("  %s -> %s", line.c_str(), r.c_str());
    return r;
}

// ---- the lender over a two-part split, part one on card da (layers 0-1), part two on card db (layers 2-3); both on
// one card when da == db.  The parts are synthetic as above, behind two Glm5Model shells.
void test_lender(int da, int db) {
    std::printf("the lender, parts on CUDA%d and CUDA%d\n", da, db);
    Part pa, pb;
    std::string err;
    if (!setup(pa, da, err, 0, 2 * NL) || !setup(pb, db, err, NL, 2 * NL)) {
        std::printf("SKIP the lender: %s\n", err.c_str());
        return;
    }
    // the shells are never destroyed: their destructor frees a loaded model's state, which they do not own (fast_
    // points at the parts above).  The geometry is the parts' (16 experts, routes of 1: a layer keeps 1 + 3 + 2 slots)
    Glm5Model* a = new Glm5Model();
    a->split_next_ = std::make_unique<Glm5Model>();
    Glm5Model* b = a->split_next_.get();
    const struct {
        Glm5Model* m;
        FastState* F;
        int dev, l0;
    } parts[2] = {{a, pa.F.get(), da, 0}, {b, pb.F.get(), db, NL}};
    for (const auto& p : parts) {
        p.m->fast_ = p.F;
        p.m->dev_ = p.dev;
        p.m->l0_ = p.l0;
        p.m->l1_ = p.m->lt_ = p.l0 + NL;
        auto& g = const_cast<strata::core::Glm5Geometry&>(p.m->geometry());
        g.n_expert = NE;
        g.n_exp_used = 1;
    }
    CHECK(Glm5Model::pool_keep(1) == 6, "the floor %d", Glm5Model::pool_keep(1));
    std::unique_ptr<strata::core::VramLender> lender = strata::core::make_glm_lender(*a);
    strata::core::LendCommands cmd(*lender);
    const bool one = da == db;
    // each part: pool 5 blocks (80 MiB), minimum 2 layers x 6 slots x 4 MiB = 48 MiB: 32 MiB, two blocks
    const std::vector<strata::core::LendCard> cards = lender->lend_cards();
    if (one)
        CHECK(cards.size() == 1 && cards[0].dev == da && cards[0].lendable_mib == 64 && cards[0].block_mib == 16,
              "one card: %zu cards, lendable %zu", cards.size(), cards.empty() ? 0 : cards[0].lendable_mib);
    else
        CHECK(cards.size() == 2 && cards[0].dev == da && cards[1].dev == db && cards[0].lendable_mib == 32 &&
                  cards[1].lendable_mib == 32 && cards[0].block_mib == 16 && cards[1].block_mib == 16,
              "two cards: %zu cards", cards.size());
    const std::string D = std::to_string(da), E = std::to_string(db);
    CHECK(ask(cmd, "LEND 9 16") == "ERR NODEV 9\n", "no such card");
    CHECK(ask(cmd, "LEND " + D + " 1000") == (one ? "ERR MAX 64\n" : "ERR MAX 32\n"), "too much");

    // a lend on the first part's card: 40 MiB -> three blocks (48 MiB); on one card the two parts share it
    const size_t f0 = card_free(da), fo0 = card_free(db);
    std::string r = ask(cmd, "LEND " + D + " 40");
    const size_t f1 = card_free(da), fo1 = card_free(db);
    const std::string want_mib = one ? "48" : "32";
    if (!one) CHECK(r == "ERR MAX 32\n", "two cards: 40 MiB is more than part one has");
    if (!one) r = ask(cmd, "LEND " + D + " 20");   // -> two blocks, 32 MiB
    const size_t f1b = card_free(da);
    unsigned long long id = 0;
    char dev_s[16] = {}, mib_s[16] = {};
    CHECK(std::sscanf(r.c_str(), "LENT %llu %15s %15s", &id, dev_s, mib_s) == 3 && dev_s == D && mib_s == want_mib,
          "lend: %s", r.c_str());
    const size_t lent_b = (size_t) std::atoi(want_mib.c_str()) * kMiB;
    CHECK((one ? f1 : f1b) >= f0 + lent_b, "CUDA%d free %zu -> %zu MiB (lent %zu)", da, f0 >> 20,
          (one ? f1 : f1b) >> 20, lent_b >> 20);
    if (!one) CHECK(fo1 + 64 * kMiB > fo0 && fo0 + 64 * kMiB > fo1, "the other card moved: %zu -> %zu MiB", fo0 >> 20,
                    fo1 >> 20);
    {
        int slots = 0;
        for (const auto& p : parts) {
            check_tables(*p.F, p.l0, "lent");
            for (int r2 = 0; r2 < NL; ++r2) slots += p.F->lp[(size_t) (p.l0 + r2)].n_lent;
        }
        CHECK((size_t) slots * kStride == lent_b, "lent slots %d", slots);
        if (!one) CHECK(pb.F->lp[NL].n_lent == 0 && pb.F->lp[NL + 1].n_lent == 0, "part two untouched");
    }
    CHECK(ask(cmd, "LENDABLE") == "LENDABLE dev=" + D + " lendable=" + (one ? "16" : "0") + " lent=" + want_mib + "\n" +
                                      (one ? "" : "LENDABLE dev=" + E + " lendable=32 lent=0\n") + "END\n",
          "lendable while lent");

    // the return while another allocation holds the memory: busy, the lease stays; then it goes back
    cudaSetDevice(da);
    void* hog = nullptr;
    {
        const size_t f = card_free(da);
        for (size_t w = f > kBlock / 2 ? f - kBlock / 2 : 0; w > lent_b; w -= 8 * kMiB) {
            if (cudaMalloc(&hog, w) == cudaSuccess) break;
            cudaGetLastError();
            hog = nullptr;
        }
        CHECK(hog != nullptr, "could not hold the card's memory");
    }
    if (hog != nullptr) {
        CHECK(ask(cmd, "RETURN " + std::to_string(id)) == "ERR BUSY " + std::to_string(id) + "\n", "busy");
        CHECK(cmd.open_leases() == 1, "the lease stays open");
        for (const auto& p : parts) check_tables(*p.F, p.l0, "busy");
        cudaSetDevice(da);
        cudaFree(hog);
    }
    const size_t f2 = card_free(da);
    CHECK(ask(cmd, "RETURN " + std::to_string(id)) == "RETURNED " + std::to_string(id) + "\n", "returned");
    const size_t f3 = card_free(da);
    CHECK(f2 >= f3 + lent_b, "CUDA%d free %zu -> %zu MiB after the return", da, f2 >> 20, f3 >> 20);
    CHECK(cmd.open_leases() == 0, "no lease left");
    for (const auto& p : parts) {
        check_tables(*p.F, p.l0, "returned");
        CHECK(p.F->vpool.mapped_count() == p.F->vpool.block_count(), "every block mapped");
        CHECK(!p.F->bg_hold, "the background moves are on again");
    }
    // ... and the second part's card lends too (on one card: its whole lendable at once)
    {
        const size_t g0 = card_free(db);
        const std::string mib = one ? "64" : "32";
        r = ask(cmd, "LEND " + E + " " + mib);
        CHECK(std::sscanf(r.c_str(), "LENT %llu %15s %15s", &id, dev_s, mib_s) == 3 && dev_s == E && mib_s == mib,
              "lend on the second card: %s", r.c_str());
        const size_t g1 = card_free(db);
        CHECK(g1 >= g0 + (size_t) std::atoi(mib.c_str()) * kMiB, "CUDA%d free %zu -> %zu MiB", db, g0 >> 20, g1 >> 20);
        for (const auto& p : parts) check_tables(*p.F, p.l0, "lent 2");
        CHECK(ask(cmd, "RETURN " + std::to_string(id)) == "RETURNED " + std::to_string(id) + "\n", "returned 2");
        for (const auto& p : parts) {
            check_tables(*p.F, p.l0, "returned 2");
            for (int r2 = 0; r2 < NL; ++r2) CHECK(p.F->lp[(size_t) (p.l0 + r2)].n_lent == 0, "n_lent 0");
        }
    }
    const auto after = lender->lend_cards();
    CHECK(after.size() == cards.size() && after[0].lendable_mib == cards[0].lendable_mib, "lendable as before");
    for (const auto& p : parts) cudaSetDevice(p.dev), cudaDeviceSynchronize();
    CHECK(cudaGetLastError() == cudaSuccess, "a CUDA error");
}

}  // namespace

int main() {
    test_choose();
    test_park_budget();
    int dev = 0, n = 0;
    if (const char* e = std::getenv("STRATA_TEST_DEV")) dev = std::atoi(e);
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= dev) {
        cudaGetLastError();
        std::printf("SKIP the GPU part: no CUDA device %d\n", dev);
    } else {
        test_part(dev);
        test_part_budget(dev, kStride + kStride / 2, 1);   // one slot's worth and a half: one expert
        test_part_budget(dev, 0, 0);
        test_lender(dev, dev);
        // STRATA_TEST_DEV2=<n>: the split's second part on that card
        if (const char* e = std::getenv("STRATA_TEST_DEV2"); e != nullptr && std::atoi(e) != dev && std::atoi(e) < n)
            test_lender(dev, std::atoi(e));
    }
    if (g_fail == 0) std::printf("glm_lend_test: all passed\n");
    return g_fail == 0 ? 0 : 1;
}
