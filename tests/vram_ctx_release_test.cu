// tests/vram_ctx_release_test.cu - one device's CUDA context released while another device keeps running (needs two
// GPUs with ~1 GiB free each, no model). The engine gives a card away by destroying its context in a process that
// keeps running on the other cards; the card must then show no compute context: NVML lists no PID of ours for it and
// its memory.used is back at the bare baseline. The open question the test answers: host memory pinned with
// cudaHostRegister while the released device was current - does the registration survive that device's
// cudaDeviceReset and keep serving DMA for the other device? Per cycle (20 cycles):
//  - A: 512 MiB, a kernel, a stream; B: 512 MiB, a kernel, a stream;
//  - with A current: two 64 MiB host buffers registered, one with flags 0 and one with cudaHostRegisterPortable;
//    B copies each into device memory and the bytes are compared;
//  - A's allocations freed, A's stream destroyed, cudaDeviceReset on A (the host buffers are NOT unregistered first);
//  - after the reset: B still runs a kernel and a memcpy; each registered buffer still copies to B (bytes compared),
//    and cudaHostGetFlags / cudaHostGetDevicePointer say whether it is still registered (printed as ANSWER, not a
//    failure either way); NVML on A: memory.used within 200 MiB of the baseline and no entry for this PID (hard
//    checks); A re-initializes (512 MiB, a kernel, free);
//  - the buffers unregistered (an error there is printed; the buffer is then freed and re-allocated).
// RSS after cycle 1 and after cycle 20 must not grow by 50 MiB or more. Exit 77 (skip) with fewer than two devices.
// vram_ctx_release_test [--dev A] [--peer B] [--cycles N] (or STRATA_TEST_DEV=A; B defaults to the next device).
#include <cuda_runtime.h>
#include <nvml.h>

#include <strings.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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
constexpr size_t kDevBytes = 512 * kMiB;
constexpr size_t kHostBytes = 64 * kMiB;
constexpr size_t kVerifyBytes = 4 * kMiB;  // the head of a device buffer read back after a kernel

const char* err_name(cudaError_t e) { return cudaGetErrorName(e); }

__global__ void fill_kernel(uint32_t* p, size_t n, uint32_t v) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = (uint32_t) i * 2654435761u + v;
}

// runs the kernel over d (kDevBytes) on stream s, reads kVerifyBytes back and compares; false on any error
bool run_kernel_checked(void* d, cudaStream_t s, uint32_t v, const char* who) {
    const size_t n = kDevBytes / sizeof(uint32_t);
    fill_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>((uint32_t*) d, n, v);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::printf("%s: kernel launch: %s\n", who, err_name(e));
        return false;
    }
    std::vector<uint32_t> h(kVerifyBytes / sizeof(uint32_t));
    e = cudaMemcpyAsync(h.data(), d, kVerifyBytes, cudaMemcpyDeviceToHost, s);
    if (e == cudaSuccess) e = cudaStreamSynchronize(s);
    if (e != cudaSuccess) {
        std::printf("%s: kernel result copy: %s\n", who, err_name(e));
        return false;
    }
    for (size_t i = 0; i < h.size(); ++i)
        if (h[i] != (uint32_t) i * 2654435761u + v) {
            std::printf("%s: kernel result wrong at %zu\n", who, i);
            return false;
        }
    return true;
}

void fill_host(uint8_t* p, size_t n, int seed) {
    for (size_t i = 0; i < n; ++i) p[i] = (uint8_t) ((i * 7 + (size_t) seed * 131) ^ (i >> 12));
}

// host -> d on stream s, then d -> scratch and compare with src; the error (or cudaSuccess) and the copy time in ms
cudaError_t copy_and_verify(const uint8_t* src, void* d, cudaStream_t s, std::vector<uint8_t>& scratch, double* ms,
                            bool* same) {
    *same = false;
    const auto t0 = std::chrono::steady_clock::now();
    cudaError_t e = cudaMemcpyAsync(d, src, kHostBytes, cudaMemcpyHostToDevice, s);
    if (e == cudaSuccess) e = cudaStreamSynchronize(s);
    *ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (e != cudaSuccess) {
        cudaGetLastError();
        return e;
    }
    e = cudaMemcpy(scratch.data(), d, kHostBytes, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return e;
    *same = std::memcmp(scratch.data(), src, kHostBytes) == 0;
    return cudaSuccess;
}

// ---- NVML on one CUDA device, found by its PCI bus id ----
struct NvmlCard {
    nvmlDevice_t h{};
    char bus[32] = {0};
    bool ok = false;
};

NvmlCard nvml_card(int cuda_dev) {
    NvmlCard c;
    if (cudaDeviceGetPCIBusId(c.bus, (int) sizeof(c.bus), cuda_dev) != cudaSuccess) return c;
    const nvmlReturn_t r = nvmlDeviceGetHandleByPciBusId_v2(c.bus, &c.h);
    if (r != NVML_SUCCESS) {
        std::printf("nvml: no handle for %s: %s\n", c.bus, nvmlErrorString(r));
        return c;
    }
    c.ok = true;
    return c;
}

// memory.used in MiB as nvidia-smi reports it (the v2 used without the card's reserved region; the v1 used is
// total - free and counts the ~350 MiB the driver reserves for itself), or -1. *reserved gets the reserved MiB.
long nvml_used_mib_h(nvmlDevice_t h, long* reserved = nullptr) {
    nvmlMemory_v2_t m{};
    m.version = nvmlMemory_v2;
    if (nvmlDeviceGetMemoryInfo_v2(h, &m) == NVML_SUCCESS) {
        if (reserved != nullptr) *reserved = (long) (m.reserved / kMiB);
        return (long) ((m.used - (m.used > m.reserved ? m.reserved : 0)) / kMiB);
    }
    nvmlMemory_t m1{};
    if (nvmlDeviceGetMemoryInfo(h, &m1) != NVML_SUCCESS) return -1;
    if (reserved != nullptr) *reserved = -1;
    return (long) (m1.used / kMiB);
}

long nvml_used_mib(const NvmlCard& c, long* reserved = nullptr) { return nvml_used_mib_h(c.h, reserved); }

// memory.used of every card, by PCI bus id, read before this process makes its first CUDA call
struct BareUsed {
    std::string bus;
    long used;
};

std::vector<BareUsed> nvml_snapshot() {
    std::vector<BareUsed> v;
    unsigned n = 0;
    if (nvmlDeviceGetCount(&n) != NVML_SUCCESS) return v;
    for (unsigned i = 0; i < n; ++i) {
        nvmlDevice_t h{};
        nvmlPciInfo_t pci{};
        if (nvmlDeviceGetHandleByIndex(i, &h) != NVML_SUCCESS || nvmlDeviceGetPciInfo(h, &pci) != NVML_SUCCESS)
            continue;
        v.push_back({pci.busId, nvml_used_mib_h(h)});
    }
    return v;
}

// the snapshot entry for bus (NVML writes the domain with 8 hex digits, CUDA with 4: compare the tail), or -1
long bare_used(const std::vector<BareUsed>& snap, const char* bus) {
    const std::string b = bus;
    for (const BareUsed& e : snap) {
        if (e.bus.size() < b.size()) continue;
        if (strcasecmp(e.bus.c_str() + (e.bus.size() - b.size()), b.c_str()) == 0) return e.used;
    }
    return -1;
}

// the compute processes NVML lists on the card; false on an NVML error
bool nvml_procs(const NvmlCard& c, std::vector<unsigned>& pids) {
    pids.clear();
    unsigned n = 0;
    nvmlReturn_t r = nvmlDeviceGetComputeRunningProcesses(c.h, &n, nullptr);
    if (r == NVML_SUCCESS) return true;  // none
    if (r != NVML_ERROR_INSUFFICIENT_SIZE) {
        std::printf("nvml: process list: %s\n", nvmlErrorString(r));
        return false;
    }
    std::vector<nvmlProcessInfo_t> infos((size_t) n + 8);
    n = (unsigned) infos.size();
    r = nvmlDeviceGetComputeRunningProcesses(c.h, &n, infos.data());
    if (r != NVML_SUCCESS) {
        std::printf("nvml: process list: %s\n", nvmlErrorString(r));
        return false;
    }
    for (unsigned i = 0; i < n; ++i) pids.push_back(infos[i].pid);
    return true;
}

bool has_pid(const std::vector<unsigned>& pids, unsigned pid) {
    for (unsigned p : pids)
        if (p == pid) return true;
    return false;
}

long rss_mib() {
    long pages = 0, resident = 0;
    FILE* f = std::fopen("/proc/self/statm", "r");
    if (f == nullptr) return -1;
    const int got = std::fscanf(f, "%ld %ld", &pages, &resident);
    std::fclose(f);
    if (got != 2) return -1;
    return (long) ((size_t) resident * (size_t) sysconf(_SC_PAGESIZE) / kMiB);
}

// one registered host buffer and what the test saw of it
struct Pinned {
    const char* name;
    unsigned flags;
    uint8_t* p = nullptr;
    bool registered = false;
    // after the reset, last cycle
    cudaError_t flags_err = cudaSuccess, devptr_err = cudaSuccess, copy_err = cudaSuccess;
    bool copy_same = false;
    double ms_before = 0, ms_after = 0;
    int cycles_still_registered = 0, cycles_copy_ok = 0;
    cudaError_t unregister_err = cudaSuccess;
    int unregister_errors = 0;
};

}  // namespace

int main(int argc, char** argv) {
    // NVML first: every card's memory.used before this process makes its first CUDA call (the bare baseline)
    nvmlReturn_t nr = nvmlInit();
    if (nr != NVML_SUCCESS) {
        std::printf("vram_ctx_release_test: nvmlInit: %s\n", nvmlErrorString(nr));
        return 1;
    }
    const std::vector<BareUsed> snap = nvml_snapshot();

    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 2) {
        std::printf("vram_ctx_release_test: needs two devices (%d visible), skipped\n", n);
        return 77;
    }
    int A = 0, B = -1, cycles = 20;
    if (const char* e = std::getenv("STRATA_TEST_DEV")) A = std::atoi(e);
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--dev") == 0 && i + 1 < argc) A = std::atoi(argv[++i]);
        if (std::strcmp(argv[i], "--peer") == 0 && i + 1 < argc) B = std::atoi(argv[++i]);
        if (std::strcmp(argv[i], "--cycles") == 0 && i + 1 < argc) cycles = std::atoi(argv[++i]);
    }
    if (B < 0) B = (A + 1) % n;
    if (A < 0 || A >= n || B < 0 || B >= n || A == B) {
        std::printf("vram_ctx_release_test: bad devices A=%d B=%d (%d devices)\n", A, B, n);
        return 1;
    }
    {
        cudaDeviceProp pa{}, pb{};
        cudaGetDeviceProperties(&pa, A);
        cudaGetDeviceProperties(&pb, B);
        int pageable = 0;  // with it, any host pointer is device-accessible, so cudaHostGetDevicePointer answers for
                           // unregistered memory too; cudaHostGetFlags is then the one that tells pinned from not
        cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, B);
        std::printf("vram_ctx_release_test: A = CUDA%d %s (released), B = CUDA%d %s (keeps running, "
                    "pageableMemoryAccess %d), %d cycles, pid %d\n",
                    A, pa.name, B, pb.name, pageable, cycles, (int) getpid());
    }

    // 1. NVML baseline on A: the runtime is up (cudaGetDeviceCount / properties create no context on A, but the
    // process is now attached to the card, which costs it a few MiB until exit) - the hard checks compare with this
    const NvmlCard card = nvml_card(A);
    if (!card.ok) {
        std::printf("vram_ctx_release_test: no NVML handle for CUDA%d\n", A);
        return 1;
    }
    const unsigned pid = (unsigned) getpid();
    long reserved0 = -1;
    const long used0 = nvml_used_mib(card, &reserved0);
    const long bare0 = bare_used(snap, card.bus);
    std::vector<unsigned> pids;
    CHECK(nvml_procs(card, pids), "nvml process list at start");
    std::printf("A %s at start: memory.used %ld MiB (bare, before this process' first CUDA call: %ld MiB; the card "
                "reserves %ld MiB besides), %zu compute process(es)%s\n",
                card.bus, used0, bare0, reserved0, pids.size(), has_pid(pids, pid) ? " INCLUDING THIS PID" : "");
    CHECK(used0 >= 0, "nvml memory info on A");
    CHECK(!has_pid(pids, pid), "this PID is on A before the test began");

    Pinned pinned[2] = {{"flags=0", 0u}, {"Portable", cudaHostRegisterPortable}};
    for (Pinned& P : pinned) {
        P.p = (uint8_t*) std::malloc(kHostBytes);
        if (P.p == nullptr) {
            std::printf("malloc failed\n");
            return 1;
        }
    }
    std::vector<uint8_t> scratch(kHostBytes);
    long rss1 = -1, rssN = -1, used_max = -1;
    std::string used_list;

    for (int cyc = 1; cyc <= cycles && g_fail == 0; ++cyc) {
        std::printf("--- cycle %d ---\n", cyc);
        cudaError_t e;

        // 2. A: 512 MiB, a kernel, a stream; B: the same
        void* dA = nullptr;
        cudaStream_t sA = nullptr;
        e = cudaSetDevice(A);
        CHECK(e == cudaSuccess, "cudaSetDevice(A): %s", err_name(e));
        e = cudaMalloc(&dA, kDevBytes);
        CHECK(e == cudaSuccess, "A cudaMalloc: %s", err_name(e));
        e = cudaStreamCreate(&sA);
        CHECK(e == cudaSuccess, "A stream: %s", err_name(e));
        if (g_fail) break;
        CHECK(run_kernel_checked(dA, sA, 1000u + (uint32_t) cyc, "A"), "A kernel before the reset");

        void* dB = nullptr;
        cudaStream_t sB = nullptr;
        e = cudaSetDevice(B);
        CHECK(e == cudaSuccess, "cudaSetDevice(B): %s", err_name(e));
        e = cudaMalloc(&dB, kDevBytes);
        CHECK(e == cudaSuccess, "B cudaMalloc: %s", err_name(e));
        e = cudaStreamCreate(&sB);
        CHECK(e == cudaSuccess, "B stream: %s", err_name(e));
        if (g_fail) break;
        CHECK(run_kernel_checked(dB, sB, 2000u + (uint32_t) cyc, "B"), "B kernel before the reset");
        if (cyc == 1) {
            // reference times on B for the host -> B copy: a buffer pinned under B and a plain pageable one. If
            // they match, the link is the limit and the times of the test buffers tell nothing about pinning.
            uint8_t* hp = nullptr;
            e = cudaHostAlloc((void**) &hp, kHostBytes, cudaHostAllocDefault);
            CHECK(e == cudaSuccess, "cudaHostAlloc on B: %s", err_name(e));
            std::vector<uint8_t> pageable(kHostBytes);
            fill_host(hp, kHostBytes, 7);
            fill_host(pageable.data(), kHostBytes, 8);
            double ms_pinned = 0, ms_pageable = 0;
            bool s1 = false, s2 = false;
            const cudaError_t e1 = copy_and_verify(hp, dB, sB, scratch, &ms_pinned, &s1);
            const cudaError_t e2 = copy_and_verify(pageable.data(), dB, sB, scratch, &ms_pageable, &s2);
            CHECK(e1 == cudaSuccess && s1 && e2 == cudaSuccess && s2, "reference copies on B: %s %s", err_name(e1),
                  err_name(e2));
            std::printf("reference host -> B, %zu MiB: pinned under B %.2f ms, pageable %.2f ms\n", kHostBytes / kMiB,
                        ms_pinned, ms_pageable);
            cudaFreeHost(hp);
        }

        // 3. register under A, DMA to B
        cudaSetDevice(A);
        for (Pinned& P : pinned) {
            e = cudaHostRegister(P.p, kHostBytes, P.flags);
            CHECK(e == cudaSuccess, "cudaHostRegister(%s) under A: %s", P.name, err_name(e));
            P.registered = e == cudaSuccess;
            fill_host(P.p, kHostBytes, cyc * 2 + (P.flags != 0));
        }
        cudaSetDevice(B);
        for (Pinned& P : pinned) {
            unsigned fl = 0;
            e = cudaHostGetFlags(&fl, P.p);
            CHECK(e == cudaSuccess, "cudaHostGetFlags(%s) with B current, before the reset: %s", P.name, err_name(e));
            bool same = false;
            e = copy_and_verify(P.p, dB, sB, scratch, &P.ms_before, &same);
            CHECK(e == cudaSuccess && same, "%s -> B before the reset: %s, bytes %s", P.name, err_name(e),
                  same ? "same" : "DIFFER");
            std::printf("%s -> B before the reset: %s, %.2f ms, flags 0x%x\n", P.name, err_name(e), P.ms_before, fl);
        }
        if (g_fail) break;

        // 4. release A: free, destroy the stream, reset. The host buffers stay registered.
        cudaSetDevice(A);
        e = cudaFree(dA);
        CHECK(e == cudaSuccess, "A cudaFree: %s", err_name(e));
        e = cudaStreamDestroy(sA);
        CHECK(e == cudaSuccess, "A cudaStreamDestroy: %s", err_name(e));
        e = cudaDeviceReset();
        CHECK(e == cudaSuccess, "A cudaDeviceReset: %s", err_name(e));
        dA = nullptr;
        sA = nullptr;

        // 5a. B still works
        cudaSetDevice(B);
        CHECK(run_kernel_checked(dB, sB, 3000u + (uint32_t) cyc, "B after the reset"), "B kernel after the reset");

        // 5b. the registered buffers after A's reset (B current)
        for (Pinned& P : pinned) {
            unsigned fl = 0;
            void* dp = nullptr;
            P.flags_err = cudaHostGetFlags(&fl, P.p);
            cudaGetLastError();
            P.devptr_err = cudaHostGetDevicePointer(&dp, P.p, 0);
            cudaGetLastError();
            fill_host(P.p, kHostBytes, cyc * 2 + 100 + (P.flags != 0));
            P.copy_err = copy_and_verify(P.p, dB, sB, scratch, &P.ms_after, &P.copy_same);
            const bool still = P.flags_err == cudaSuccess;
            if (still) ++P.cycles_still_registered;
            if (P.copy_err == cudaSuccess && P.copy_same) ++P.cycles_copy_ok;
            std::printf("%s after A's reset: cudaHostGetFlags %s%s, cudaHostGetDevicePointer %s, copy -> B %s "
                        "(bytes %s, %.2f ms, before %.2f ms)\n",
                        P.name, err_name(P.flags_err), still ? "" : " (no longer registered)", err_name(P.devptr_err),
                        err_name(P.copy_err), P.copy_same ? "same" : "DIFFER", P.ms_after, P.ms_before);
        }

        // 5c. NVML on A: used back near the baseline, no entry for this PID. NVML can lag the destroy a little.
        long used = -1;
        bool listed = true, procs_ok = false;
        for (int tries = 0; tries < 30; ++tries) {
            used = nvml_used_mib(card);
            procs_ok = nvml_procs(card, pids);
            listed = has_pid(pids, pid);
            if (procs_ok && !listed && used >= 0 && used <= used0 + 200) break;
            usleep(100 * 1000);
        }
        std::printf("A after the reset: memory.used %ld MiB (baseline %ld), %zu compute process(es)%s\n", used, used0,
                    pids.size(), listed ? " INCLUDING THIS PID" : "");
        CHECK(procs_ok, "nvml process list after the reset");
        CHECK(used >= 0 && used <= used0 + 200, "A memory.used %ld MiB, baseline %ld MiB", used, used0);
        CHECK(!listed, "this PID is still a compute process on A after cudaDeviceReset");
        if (used > used_max) used_max = used;
        used_list += (used_list.empty() ? "" : " ") + std::to_string(used);

        // 5d. A re-initializes
        e = cudaSetDevice(A);
        CHECK(e == cudaSuccess, "cudaSetDevice(A) after the reset: %s", err_name(e));
        e = cudaMalloc(&dA, kDevBytes);
        CHECK(e == cudaSuccess, "A cudaMalloc after the reset: %s", err_name(e));
        if (e == cudaSuccess) {
            CHECK(run_kernel_checked(dA, nullptr, 4000u + (uint32_t) cyc, "A re-init"), "A kernel after re-init");
            e = cudaFree(dA);
            CHECK(e == cudaSuccess, "A cudaFree after re-init: %s", err_name(e));
        }

        // 6. unregister (B current); an error means the registration died with A's context: free and re-allocate
        cudaSetDevice(B);
        for (Pinned& P : pinned) {
            P.unregister_err = cudaHostUnregister(P.p);
            cudaGetLastError();
            if (P.unregister_err != cudaSuccess) {
                ++P.unregister_errors;
                std::printf("cudaHostUnregister(%s) after A's reset: %s -> free and re-malloc\n", P.name,
                            err_name(P.unregister_err));
                std::free(P.p);
                P.p = (uint8_t*) std::malloc(kHostBytes);
                if (P.p == nullptr) {
                    std::printf("malloc failed\n");
                    return 1;
                }
            }
            P.registered = false;
        }
        e = cudaFree(dB);
        CHECK(e == cudaSuccess, "B cudaFree: %s", err_name(e));
        e = cudaStreamDestroy(sB);
        CHECK(e == cudaSuccess, "B cudaStreamDestroy: %s", err_name(e));

        const long rss = rss_mib();
        if (cyc == 1) rss1 = rss;
        if (cyc == cycles) rssN = rss;
        std::printf("cycle %d: A memory.used after the reset %ld MiB, RSS %ld MiB\n", cyc, used, rss);
    }

    // the final state: A released once more and left that way
    long used_final = -1;
    bool listed_final = true;
    {
        cudaSetDevice(A);
        const cudaError_t e = cudaDeviceReset();
        CHECK(e == cudaSuccess, "final cudaDeviceReset on A: %s", err_name(e));
        for (int tries = 0; tries < 30; ++tries) {
            used_final = nvml_used_mib(card);
            if (nvml_procs(card, pids)) listed_final = has_pid(pids, pid);
            if (!listed_final && used_final >= 0 && used_final <= used0 + 200) break;
            usleep(100 * 1000);
        }
        CHECK(used_final >= 0 && used_final <= used0 + 200, "final A memory.used %ld MiB, baseline %ld", used_final,
              used0);
        CHECK(!listed_final, "this PID is still a compute process on A at the end");
    }
    if (rss1 >= 0 && rssN >= 0) CHECK(rssN - rss1 < 50, "RSS grew %ld MiB over the cycles", rssN - rss1);

    std::printf("=== summary ===\n");
    std::printf("A %s memory.used: bare %ld MiB, baseline %ld MiB, after the first reset %s MiB, max over %d cycles "
                "%ld MiB, final %ld MiB\n",
                card.bus, bare0, used0, used_list.empty() ? "?" : used_list.substr(0, used_list.find(' ')).c_str(),
                cycles, used_max, used_final);
    std::printf("A memory.used after each reset: %s\n", used_list.c_str());
    for (const Pinned& P : pinned) {
        std::printf("ANSWER (%s): after the owning device's cudaDeviceReset the registration %s "
                    "(cudaHostGetFlags %s, cudaHostGetDevicePointer %s, still registered in %d of %d cycles); "
                    "the copy to B %s (%s, bytes %s, ok in %d of %d cycles; %.2f ms pinned before, %.2f ms after); "
                    "cudaHostUnregister afterwards %s (an error in %d of %d cycles)\n",
                    P.name, P.cycles_still_registered == cycles ? "SURVIVES" : "IS GONE", err_name(P.flags_err),
                    err_name(P.devptr_err), P.cycles_still_registered, cycles,
                    P.copy_err == cudaSuccess ? "still works" : "FAILS", err_name(P.copy_err),
                    P.copy_same ? "same" : "DIFFER", P.cycles_copy_ok, cycles, P.ms_before, P.ms_after,
                    err_name(P.unregister_err), P.unregister_errors, cycles);
    }
    std::printf("RSS: after cycle 1 %ld MiB, after cycle %d %ld MiB, delta %ld MiB\n", rss1, cycles, rssN,
                (rss1 >= 0 && rssN >= 0) ? rssN - rss1 : -1);
    std::printf("vram_ctx_release_test: %s\n", g_fail == 0 ? "PASS" : "FAIL");

    for (Pinned& P : pinned) std::free(P.p);
    nvmlShutdown();
    return g_fail == 0 ? 0 : 1;
}
