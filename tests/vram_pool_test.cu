// tests/vram_pool_test.cu - the VMM expert-slot pool (src/core/vram_pool.cu) on one card (needs a GPU with ~2 GiB free,
// no model):
//  - a pool of ~1 GiB in two slot sizes: blocks of whole slots, the block size a multiple of the granularity;
//  - 256 MiB of its blocks unmapped: the card's free memory rises by at least that;
//  - the blocks mapped back at the same addresses and writable, the other blocks' slots unchanged;
//  - a map while another allocation holds the card's memory answers Busy and leaves the blocks unmapped, and
//    succeeds once that allocation is freed;
//  - the plain fallback (Options::lend false, and STRATA_VRAM_POOL_VMM=0) answers lendable 0;
//  - destroy() gives the pool's memory back.
// vram_pool_test [--dev N] (or STRATA_TEST_DEV=N); vram_pool_test --list prints every card's granularity and exits.
#include "strata/core/vram_pool.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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

size_t free_bytes() {
    size_t f = 0, t = 0;
    cudaMemGetInfo(&f, &t);
    return f;
}

uint8_t pattern(int r, int i, int round) { return (uint8_t) ((r * 31 + i * 7 + round * 101) | 1); }

void fill_slot(const VramPool& P, int r, int i, uint8_t v) {
    cudaMemset(P.slot_ptr(r, i), v, P.region(r).slot_bytes);
}

// every byte of the slot is v (read back in full)
bool slot_is(const VramPool& P, int r, int i, uint8_t v) {
    std::vector<uint8_t> h(P.region(r).slot_bytes);
    if (cudaMemcpy(h.data(), P.slot_ptr(r, i), h.size(), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    for (uint8_t x : h)
        if (x != v) return false;
    return true;
}

bool in_set(const std::vector<int>& s, int b) {
    for (int x : s)
        if (x == b) return true;
    return false;
}

void test_layout(const VramPool& P) {
    CHECK(P.block_bytes() % P.granularity() == 0, "block %zu, granularity %zu", P.block_bytes(), P.granularity());
    std::vector<int> per_block((size_t) P.block_count(), 0);
    size_t total = 0;
    for (int r = 0; r < P.region_count(); ++r) {
        const auto& L = P.region(r);
        CHECK(L.per_block >= 1 && (size_t) L.per_block * L.slot_bytes <= P.block_bytes(), "region %d", r);
        for (int i = 0; i < L.n_slots; ++i) {
            const uint8_t* p = P.slot_ptr(r, i);
            const size_t off = (size_t) (p - P.block_ptr(0));
            const size_t b0 = off / P.block_bytes(), b1 = (off + L.slot_bytes - 1) / P.block_bytes();
            CHECK(b0 == b1 && (int) b1 < P.block_count(), "region %d slot %d straddles blocks %zu..%zu", r, i, b0, b1);
            CHECK(P.slot_index(r, p) == i && P.slot_index(r, p + L.slot_bytes - 1) == i, "region %d slot %d", r, i);
            if (i > 0) CHECK(P.slot_ptr(r, i - 1) + L.slot_bytes <= p, "region %d slots %d, %d overlap", r, i - 1, i);
            ++per_block[b0];
            total += L.slot_bytes;
        }
    }
    // block_slots names exactly the slots found inside each block
    for (int b = 0; b < P.block_count(); ++b) {
        int n = 0;
        for (const auto& s : P.block_slots(b)) {
            n += s.count;
            for (int i = s.first; i < s.first + s.count; ++i)
                CHECK((size_t) (P.slot_ptr(s.region, i) - P.block_ptr(0)) / P.block_bytes() == (size_t) b,
                      "block %d span region %d slot %d", b, s.region, i);
        }
        CHECK(n == per_block[(size_t) b], "block %d: block_slots %d, found %d", b, n, per_block[(size_t) b]);
    }
    CHECK(total == P.slot_bytes_total(), "slot bytes %zu vs %zu", total, P.slot_bytes_total());
    std::printf("layout: %d blocks of %zu MiB, %.1f%% of the range in slots\n", P.block_count(),
                P.block_bytes() / kMiB, 100.0 * (double) total / (double) P.bytes());
}

void test_lend(int dev) {
    // ~1 GiB: three layers of 20 slots of 12.75 MiB and one of 30 slots of 9.0 MiB (both 256-byte multiples, neither
    // a divisor of a granularity multiple)
    const size_t s1 = (size_t) 13369600, s2 = (size_t) 9437440;
    const std::vector<VramPoolRegion> regions = {{s1, 20}, {s1, 20}, {s1, 20}, {s2, 30}};
    VramPool P;
    std::string err;
    const size_t f0 = free_bytes();
    if (!P.create(dev, regions, VramPool::Options{}, err)) {
        CHECK(false, "create: %s", err.c_str());
        return;
    }
    std::printf("%s\n", P.summary().c_str());
    CHECK(P.lendable(), "the pool is not lendable on CUDA%d", dev);
    if (!P.lendable()) return;
    CHECK(P.bytes() >= (size_t) 1 << 30, "pool %zu bytes", P.bytes());
    CHECK(f0 - free_bytes() >= P.bytes(), "free fell %zu, pool %zu", f0 - free_bytes(), P.bytes());
    test_layout(P);

    std::vector<uint8_t*> addr0;
    for (int r = 0; r < P.region_count(); ++r)
        for (int i = 0; i < P.region(r).n_slots; ++i) {
            addr0.push_back(P.slot_ptr(r, i));
            fill_slot(P, r, i, pattern(r, i, 0));
        }
    cudaDeviceSynchronize();

    // a chosen 256 MiB: whole blocks from block 2
    std::vector<int> lend;
    for (int b = 2; (size_t) lend.size() * P.block_bytes() < 256 * kMiB; ++b) lend.push_back(b);
    CHECK(lend.back() < P.block_count(), "the pool has %d blocks", P.block_count());
    const size_t lend_bytes = lend.size() * P.block_bytes();

    size_t f1 = free_bytes();
    CHECK(P.unmap(lend, err) == VramPool::Result::Ok, "unmap: %s", err.c_str());
    size_t f2 = free_bytes();
    std::printf("unmap %zu blocks (%zu MiB): free %zu -> %zu MiB (+%zu)\n", lend.size(), lend_bytes / kMiB, f1 / kMiB,
                f2 / kMiB, (f2 - f1) / kMiB);
    CHECK(f2 >= f1 + 256 * kMiB, "free rose by %zu MiB", (f2 - f1) / kMiB);
    CHECK(P.mapped_count() == P.block_count() - (int) lend.size(), "mapped %d", P.mapped_count());
    for (int b : lend) CHECK(!P.mapped(b), "block %d still mapped", b);
    CHECK(P.unmap(lend, err) == VramPool::Result::Error, "unmapping unmapped blocks was not refused");
    CHECK(P.map({0}, err) == VramPool::Result::Error, "mapping a mapped block was not refused");

    // map back: the same addresses, writable, and the other blocks kept their data
    CHECK(P.map(lend, err) == VramPool::Result::Ok, "map: %s", err.c_str());
    const size_t f3 = free_bytes();
    std::printf("map back: free %zu MiB (-%zu)\n", f3 / kMiB, (f2 - f3) / kMiB);
    CHECK(f3 + lend_bytes <= f2 + 2 * P.granularity(), "free fell only %zu MiB", (f2 - f3) / kMiB);
    CHECK(P.mapped_count() == P.block_count(), "mapped %d of %d", P.mapped_count(), P.block_count());
    {
        size_t k = 0;
        int lent = 0, kept = 0;
        for (int r = 0; r < P.region_count(); ++r)
            for (int i = 0; i < P.region(r).n_slots; ++i, ++k) {
                CHECK(P.slot_ptr(r, i) == addr0[k], "region %d slot %d moved", r, i);
                const int b = (int) ((size_t) (P.slot_ptr(r, i) - P.block_ptr(0)) / P.block_bytes());
                if (in_set(lend, b)) {
                    fill_slot(P, r, i, pattern(r, i, 1));
                    cudaDeviceSynchronize();
                    CHECK(slot_is(P, r, i, pattern(r, i, 1)), "remapped region %d slot %d did not take a write", r, i);
                    ++lent;
                } else {
                    CHECK(slot_is(P, r, i, pattern(r, i, 0)), "kept region %d slot %d lost its data", r, i);
                    ++kept;
                }
            }
        std::printf("after the map: %d remapped slots written and read back, %d kept slots unchanged\n", lent, kept);
        CHECK(cudaGetLastError() == cudaSuccess, "a CUDA error");
    }

    // busy: another allocation holds the memory the blocks gave back
    CHECK(P.unmap(lend, err) == VramPool::Result::Ok, "unmap 2: %s", err.c_str());
    void* hog = nullptr;
    {
        const size_t f = free_bytes();
        // leave less than one block free (and so far less than the lend)
        for (size_t want = f > P.block_bytes() / 2 ? f - P.block_bytes() / 2 : 0; want > lend_bytes;
             want -= 8 * kMiB) {
            if (cudaMalloc(&hog, want) == cudaSuccess) break;
            cudaGetLastError();
            hog = nullptr;
        }
        CHECK(hog != nullptr, "could not hold the card's memory");
        std::printf("held: free %zu MiB left\n", free_bytes() / kMiB);
    }
    if (hog != nullptr) {
        const VramPool::Result r = P.map(lend, err);
        CHECK(r == VramPool::Result::Busy, "map while held: %d (%s)", (int) r, err.c_str());
        std::printf("map while held: %s (%s)\n", r == VramPool::Result::Busy ? "busy" : "NOT busy", err.c_str());
        for (int b : lend) CHECK(!P.mapped(b), "block %d mapped after a busy map", b);
        CHECK(P.mapped_count() == P.block_count() - (int) lend.size(), "mapped %d", P.mapped_count());
        // a second try while still held: busy again, nothing leaked
        CHECK(P.map(lend, err) == VramPool::Result::Busy, "map while held, again");
        cudaFree(hog);
    }
    CHECK(P.map(lend, err) == VramPool::Result::Ok, "map after the hold: %s", err.c_str());
    {
        int ok = 0;
        for (int r = 0; r < P.region_count(); ++r)
            for (int i = 0; i < P.region(r).n_slots; ++i) {
                const int b = (int) ((size_t) (P.slot_ptr(r, i) - P.block_ptr(0)) / P.block_bytes());
                if (!in_set(lend, b)) continue;
                fill_slot(P, r, i, pattern(r, i, 2));
                cudaDeviceSynchronize();
                ok += slot_is(P, r, i, pattern(r, i, 2));
                CHECK(slot_is(P, r, i, pattern(r, i, 2)), "region %d slot %d after the hold", r, i);
            }
        std::printf("map after the hold: ok, %d slots written and read back\n", ok);
    }

    const size_t f4 = free_bytes(), pool_bytes = P.bytes();
    P.destroy();
    const size_t f5 = free_bytes();
    std::printf("destroy: free +%zu MiB\n", (f5 - f4) / kMiB);
    CHECK(f5 >= f4 + pool_bytes, "destroy freed %zu MiB of %zu", (f5 - f4) / kMiB, pool_bytes / kMiB);
    CHECK(!P.valid(), "valid after destroy");
}

void test_fallback(int dev, bool by_env) {
    std::string err;
    VramPool P;
    VramPool::Options o;
    o.tag = by_env ? "fallback (env)" : "fallback";
    if (by_env)
        setenv("STRATA_VRAM_POOL_VMM", "0", 1);
    else
        o.lend = false;
    const bool ok = P.create(dev, {{(size_t) 13369600, 6}, {(size_t) 9437440, 4}}, o, err);
    if (by_env) unsetenv("STRATA_VRAM_POOL_VMM");
    CHECK(ok, "create: %s", err.c_str());
    if (!ok) return;
    std::printf("%s\n", P.summary().c_str());
    CHECK(!P.lendable(), "the fallback is lendable");
    CHECK(P.mapped_count() == P.block_count(), "mapped %d of %d", P.mapped_count(), P.block_count());
    CHECK(P.unmap({0}, err) == VramPool::Result::NotLendable, "unmap on the fallback");
    CHECK(P.map({0}, err) == VramPool::Result::NotLendable, "map on the fallback");
    for (int r = 0; r < P.region_count(); ++r)
        for (int i = 0; i < P.region(r).n_slots; ++i) fill_slot(P, r, i, pattern(r, i, 3));
    cudaDeviceSynchronize();
    for (int r = 0; r < P.region_count(); ++r)
        for (int i = 0; i < P.region(r).n_slots; ++i)
            CHECK(slot_is(P, r, i, pattern(r, i, 3)), "fallback region %d slot %d", r, i);
    std::printf("fallback%s: lendable %d, %d slots written and read back\n", by_env ? " (env)" : "", (int) P.lendable(),
                P.region(0).n_slots + P.region(1).n_slots);
}

}  // namespace

int main(int argc, char** argv) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("vram_pool_test: no GPU, skipped\n");
        return 0;
    }
    int dev = 0;
    if (const char* e = std::getenv("STRATA_TEST_DEV")) dev = std::atoi(e);
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--dev") == 0 && i + 1 < argc) dev = std::atoi(argv[++i]);
        if (std::strcmp(argv[i], "--list") == 0) {
            for (int d = 0; d < n; ++d) {
                cudaDeviceProp p{};
                cudaGetDeviceProperties(&p, d);
                size_t g = 0;
                const bool vmm = VramPool::vmm_supported(d, &g);
                std::printf("CUDA%d %s: VMM %s, granularity %zu KiB\n", d, p.name, vmm ? "yes" : "no", g >> 10);
            }
            return 0;
        }
    }
    if (dev < 0 || dev >= n) {
        std::printf("vram_pool_test: no device %d (%d devices)\n", dev, n);
        return 1;
    }
    cudaSetDevice(dev);
    std::printf("vram_pool_test: CUDA%d, %zu MiB free\n", dev, free_bytes() / kMiB);
    test_fallback(dev, false);
    test_fallback(dev, true);
    test_lend(dev);
    if (g_fail == 0) std::printf("vram_pool_test: ok\n");
    return g_fail == 0 ? 0 : 1;
}
