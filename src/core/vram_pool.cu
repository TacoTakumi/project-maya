// src/core/vram_pool.cu - the expert-slot pool: one reserved address range, backed by blocks the driver can take back
// and give again (include/strata/core/vram_pool.hpp).
#include "strata/core/vram_pool.hpp"

#include <cuda_runtime.h>
#if !defined(STRATA_USE_HIP)
#include <cuda.h>   // the driver's VMM calls (cuMem*): CUDA only, a HIP build always takes the plain fallback
#define STRATA_VRAM_POOL_HAS_VMM 1
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::core {

namespace {

constexpr size_t kMiB = (size_t) 1 << 20;
constexpr size_t kPlainGranularity = (size_t) 2 << 20;   // the layout's unit where there is no VMM

// Lays the regions out in blocks of `block` bytes (origin left null); returns the block count.  Consecutive regions of
// one slot size share blocks; a new slot size starts a new block.
int lay_out(const std::vector<VramPoolRegion>& in, size_t block, std::vector<VramPool::RegionLayout>& out) {
    out.assign(in.size(), {});
    int blocks = 0, pos = 0, per = 0;
    size_t cur = 0;
    for (size_t r = 0; r < in.size(); ++r) {
        auto& L = out[r];
        L.slot_bytes = in[r].slot_bytes;
        L.n_slots = in[r].n_slots;
        L.per_block = L.slot_bytes > 0 ? (int) std::max<size_t>(1, block / L.slot_bytes) : 1;
        if (L.n_slots <= 0) {
            L.first_block = blocks;
            continue;
        }
        if (L.slot_bytes != cur || pos == per || blocks == 0) {   // a new block
            cur = L.slot_bytes;
            per = L.per_block;
            pos = 0;
            ++blocks;
        }
        L.first_block = blocks - 1;
        L.first_pos = pos;
        const int end = pos + L.n_slots;   // slots from the start of the region's first block
        blocks += (end - 1) / per;
        pos = (end - 1) % per + 1;
    }
    return blocks;
}

#if defined(STRATA_VRAM_POOL_HAS_VMM)
const char* cu_name(CUresult r) {
    const char* s = nullptr;
    cuGetErrorName(r, &s);
    return s != nullptr ? s : "CUDA driver error";
}

CUmemAllocationProp block_prop(int dev) {
    CUmemAllocationProp p{};
    p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    p.location.id = dev;
    p.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
    return p;
}

// one block at `va`: a physical allocation, mapped, read/write for the owning device.  On failure nothing is left.
CUresult map_block(int dev, CUdeviceptr va, size_t bytes, CUmemGenericAllocationHandle& h) {
    const CUmemAllocationProp p = block_prop(dev);
    CUresult r = cuMemCreate(&h, bytes, &p, 0);
    if (r != CUDA_SUCCESS) return r;
    if ((r = cuMemMap(va, bytes, 0, h, 0)) != CUDA_SUCCESS) {
        cuMemRelease(h);
        return r;
    }
    CUmemAccessDesc a{};
    a.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    a.location.id = dev;
    a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if ((r = cuMemSetAccess(va, bytes, &a, 1)) != CUDA_SUCCESS) {
        cuMemUnmap(va, bytes);
        cuMemRelease(h);
    }
    return r;
}
#endif

// the runtime's current device, set to `dev` for the scope and put back after
struct DeviceScope {
    int prev = -1;
    explicit DeviceScope(int dev) {
        cudaGetDevice(&prev);
        if (prev != dev) cudaSetDevice(dev);
    }
    ~DeviceScope() {
        if (prev >= 0) cudaSetDevice(prev);
    }
};

}  // namespace

bool VramPool::vmm_supported(int dev, size_t* granularity) {
    if (granularity != nullptr) *granularity = kPlainGranularity;
#if defined(STRATA_VRAM_POOL_HAS_VMM)
    CUdevice d = 0;
    int vmm = 0;
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&d, dev) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, d) != CUDA_SUCCESS || !vmm)
        return false;
    const CUmemAllocationProp p = block_prop(dev);
    size_t g = 0;
    if (cuMemGetAllocationGranularity(&g, &p, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) != CUDA_SUCCESS || g == 0) return false;
    if (granularity != nullptr) *granularity = g;
    return true;
#else
    (void) dev;
    return false;
#endif
}

VramPool::~VramPool() { destroy(); }

bool VramPool::create(int dev, const std::vector<VramPoolRegion>& regions, const Options& o, std::string& err) {
    destroy();
    size_t smax = 0, need = 0;
    for (const auto& r : regions) {
        if (r.n_slots < 0 || (r.n_slots > 0 && r.slot_bytes == 0)) {
            err = "vram pool: a region of " + std::to_string(r.n_slots) + " slots of " + std::to_string(r.slot_bytes) +
                  " bytes";
            return false;
        }
        if (r.n_slots > 0) smax = std::max(smax, r.slot_bytes);
        need += (size_t) r.n_slots * r.slot_bytes;
    }
    if (need == 0) {
        err = "vram pool: no slots";
        return false;
    }
    // lend or not, and why not
    std::string why;
    size_t gran = kPlainGranularity;
    bool vmm = false;
#if !defined(STRATA_VRAM_POOL_HAS_VMM)
    why = "a HIP build";
#else
    const char* env = std::getenv("STRATA_VRAM_POOL_VMM");
    if (!o.lend)
        why = "lending off";
    else if (env != nullptr && std::strcmp(env, "0") == 0)
        why = "STRATA_VRAM_POOL_VMM=0";
    else if (!(vmm = vmm_supported(dev, &gran)))
        why = "the device has no virtual memory management";
#endif
    // the block size: a granularity multiple holding the largest slot, in [min, max], the one wasting least (the
    // larger on a tie: fewer blocks to map)
    std::vector<RegionLayout> lay;
    size_t block = 0;
    int nblocks = 0;
    if (o.block_bytes > 0) {
        block = (o.block_bytes + gran - 1) / gran * gran;
        if (block < smax) {
            err = "vram pool: a block of " + std::to_string(block) + " bytes is smaller than a slot (" +
                  std::to_string(smax) + ")";
            return false;
        }
        nblocks = lay_out(regions, block, lay);
    } else {
        const size_t lo = std::max((smax + gran - 1) / gran, (o.min_block_bytes + gran - 1) / gran);
        const size_t hi = std::max(lo, o.max_block_bytes / gran);
        size_t best = 0;
        for (size_t m = lo; m <= hi; ++m) {
            std::vector<RegionLayout> t;
            const int n = lay_out(regions, m * gran, t);
            if (best == 0 || (size_t) n * m * gran <= best) {
                best = (size_t) n * m * gran;
                block = m * gran;
                nblocks = n;
                lay.swap(t);
            }
        }
    }
    const size_t total = (size_t) nblocks * block;

    dev_ = dev;
    gran_ = gran;
    block_bytes_ = block;
    used_ = need;
    regions_ = std::move(lay);
    mapped_.assign((size_t) nblocks, 0);
    handle_.assign((size_t) nblocks, 0);
    DeviceScope ds(dev);
    cudaFree(nullptr);   // the device's context, for the driver calls below

#if defined(STRATA_VRAM_POOL_HAS_VMM)
    if (vmm) {
        CUdeviceptr va = 0;
        CUresult r = cuMemAddressReserve(&va, total, gran, 0, 0);
        int b = 0;
        if (r == CUDA_SUCCESS) {
            base_ = (uint8_t*) va;
            vmm_ = true;
            for (; b < nblocks; ++b) {
                CUmemGenericAllocationHandle h = 0;
                if ((r = map_block(dev, va + (size_t) b * block, block, h)) != CUDA_SUCCESS) break;
                handle_[(size_t) b] = (unsigned long long) h;
                mapped_[(size_t) b] = 1;
            }
        }
        if (r == CUDA_ERROR_OUT_OF_MEMORY) {
            err = "vram pool: CUDA" + std::to_string(dev) + " has no room for the expert pool (" +
                  std::to_string(total >> 20) + " MiB; block " + std::to_string(b) + " of " + std::to_string(nblocks) +
                  " did not allocate)";
            destroy();
            return false;
        }
        if (r != CUDA_SUCCESS) {   // VMM refused for another reason: the plain allocation instead
            why = std::string("VMM failed (") + cu_name(r) + ")";
            vmm = false;
            release();
        }
    }
#endif
    if (!vmm) {
        // one allocation, or one a region where the driver caps a single allocation below the whole (each region whole
        // in its own: its blocks from first_block on, so its addressing is unchanged)
        void* p = nullptr;
        if (cudaMalloc(&p, total) == cudaSuccess) {
            base_ = (uint8_t*) p;
            plain_.push_back(p);
        } else {
            cudaGetLastError();
            for (const auto& L : regions_) {
                if (L.n_slots <= 0) continue;
                const size_t g = (size_t) L.first_pos + (size_t) L.n_slots - 1;
                if (cudaMalloc(&p, (g / (size_t) L.per_block + 1) * block) != cudaSuccess) {
                    cudaGetLastError();
                    err = "vram pool: the expert pool (" + std::to_string(total >> 20) + " MiB) did not allocate on CUDA" +
                          std::to_string(dev);
                    destroy();
                    return false;
                }
                plain_.push_back(p);
            }
        }
        std::fill(mapped_.begin(), mapped_.end(), (char) 1);
    }
    size_t k = 0;
    for (auto& L : regions_)
        if (base_ != nullptr)
            L.origin = base_ + (size_t) L.first_block * block;
        else
            L.origin = L.n_slots > 0 ? (uint8_t*) plain_[k++] : nullptr;

    int pmin = INT32_MAX, pmax = 0;
    for (const auto& L : regions_)
        if (L.n_slots > 0) {
            pmin = std::min(pmin, L.per_block);
            pmax = std::max(pmax, L.per_block);
        }
    char per[32];
    if (pmin == pmax)
        std::snprintf(per, sizeof per, "%d", pmin);
    else
        std::snprintf(per, sizeof per, "%d-%d", pmin, pmax);
    char line[512];
    std::snprintf(line, sizeof line,
                  "%s: CUDA%d expert pool: %.2f GB in %d blocks of %.0f MiB (granularity %.0f MiB, %s slots a block, "
                  "%.1f%% of it in slots), %s%s",
                  o.tag, dev, (double) total / 1e9, nblocks, (double) block / kMiB, (double) gran / kMiB, per,
                  100.0 * (double) need / (double) total, vmm ? "lendable" : "not lendable: ", vmm ? "" : why.c_str());
    summary_ = line;
    if (o.log) std::fprintf(stderr, "%s\n", line);
    return true;
}

void VramPool::release() {
#if defined(STRATA_VRAM_POOL_HAS_VMM)
    if (vmm_) {
        for (size_t b = 0; b < mapped_.size(); ++b)
            if (mapped_[b]) {
                cuMemUnmap((CUdeviceptr) (base_ + b * block_bytes_), block_bytes_);
                cuMemRelease((CUmemGenericAllocationHandle) handle_[b]);
            }
        cuMemAddressFree((CUdeviceptr) base_, mapped_.size() * block_bytes_);
    }
#endif
    if (!plain_.empty()) {
        DeviceScope ds(dev_);
        for (void* p : plain_) cudaFree(p);
    }
    plain_.clear();
    base_ = nullptr;
    vmm_ = false;
    std::fill(mapped_.begin(), mapped_.end(), (char) 0);
    std::fill(handle_.begin(), handle_.end(), 0ull);
}

void VramPool::destroy() {
    release();
    dev_ = -1;
    gran_ = block_bytes_ = used_ = 0;
    regions_.clear();
    mapped_.clear();
    handle_.clear();
    summary_.clear();
}

int VramPool::mapped_count() const {
    int n = 0;
    for (char m : mapped_) n += m != 0;
    return n;
}

int VramPool::slot_index(int r, const void* p) const {
    const RegionLayout& L = regions_[(size_t) r];
    const uint8_t* q = (const uint8_t*) p;
    if (L.n_slots <= 0 || L.origin == nullptr || q < L.origin) return -1;
    const size_t off = (size_t) (q - L.origin);
    const size_t j = off % block_bytes_ / L.slot_bytes;
    if (j >= (size_t) L.per_block) return -1;   // the block's unused tail
    const long long i = (long long) (off / block_bytes_ * (size_t) L.per_block + j) - L.first_pos;
    return i >= 0 && i < L.n_slots ? (int) i : -1;
}

bool VramPool::linear(int r) const {
    const RegionLayout& L = regions_[(size_t) r];
    return (size_t) L.per_block * L.slot_bytes == block_bytes_ || L.first_pos + L.n_slots <= L.per_block;
}

std::vector<VramPool::Span> VramPool::block_slots(int b) const {
    std::vector<Span> out;
    for (int r = 0; r < region_count(); ++r) {
        const RegionLayout& L = regions_[(size_t) r];
        if (L.n_slots <= 0) continue;
        // the region's slot positions g in [first_pos, first_pos + n); block b holds g in [lo, hi)
        const long long rel = (long long) b - L.first_block;
        if (rel < 0) continue;
        const long long lo = std::max<long long>(rel * L.per_block, L.first_pos);
        const long long hi = std::min<long long>((rel + 1) * L.per_block, (long long) L.first_pos + L.n_slots);
        if (lo < hi) out.push_back({r, (int) (lo - L.first_pos), (int) (hi - lo)});
    }
    return out;
}

VramPool::Result VramPool::unmap(const std::vector<int>& blocks, std::string& err) {
    if (!vmm_) {
        err = "vram pool: not lendable";
        return Result::NotLendable;
    }
    std::vector<char> seen(mapped_.size(), 0);
    for (int b : blocks)
        if (b < 0 || b >= block_count() || !mapped_[(size_t) b] || seen[(size_t) b]++) {
            err = "vram pool: block " + std::to_string(b) + " is not a mapped block of the set";
            return Result::Error;
        }
#if defined(STRATA_VRAM_POOL_HAS_VMM)
    // (cuMemUnmap does not wait for kernels still reading the range: the caller synchronized before this)
    for (int b : blocks) {
        const CUdeviceptr va = (CUdeviceptr) (base_ + (size_t) b * block_bytes_);
        CUresult r = cuMemUnmap(va, block_bytes_);
        if (r != CUDA_SUCCESS) {
            err = std::string("vram pool: cuMemUnmap of block ") + std::to_string(b) + ": " + cu_name(r);
            return Result::Error;   // (this block and the rest stay mapped; those before it are unmapped)
        }
        mapped_[(size_t) b] = 0;
        r = cuMemRelease((CUmemGenericAllocationHandle) handle_[(size_t) b]);
        handle_[(size_t) b] = 0;
        if (r != CUDA_SUCCESS) {
            err = std::string("vram pool: cuMemRelease of block ") + std::to_string(b) + ": " + cu_name(r);
            return Result::Error;
        }
    }
#endif
    return Result::Ok;
}

VramPool::Result VramPool::map(const std::vector<int>& blocks, std::string& err) {
    if (!vmm_) {
        err = "vram pool: not lendable";
        return Result::NotLendable;
    }
    std::vector<char> seen(mapped_.size(), 0);
    for (int b : blocks)
        if (b < 0 || b >= block_count() || mapped_[(size_t) b] || seen[(size_t) b]++) {
            err = "vram pool: block " + std::to_string(b) + " is not an unmapped block of the set";
            return Result::Error;
        }
#if defined(STRATA_VRAM_POOL_HAS_VMM)
    DeviceScope ds(dev_);
    size_t done = 0;
    CUresult r = CUDA_SUCCESS;
    for (; done < blocks.size(); ++done) {
        const int b = blocks[done];
        CUmemGenericAllocationHandle h = 0;
        if ((r = map_block(dev_, (CUdeviceptr) (base_ + (size_t) b * block_bytes_), block_bytes_, h)) != CUDA_SUCCESS)
            break;
        handle_[(size_t) b] = (unsigned long long) h;
        mapped_[(size_t) b] = 1;
    }
    if (r == CUDA_SUCCESS) return Result::Ok;
    // all or nothing: the blocks this call mapped go back, the set stays unmapped
    for (size_t i = 0; i < done; ++i) {
        const int b = blocks[i];
        cuMemUnmap((CUdeviceptr) (base_ + (size_t) b * block_bytes_), block_bytes_);
        cuMemRelease((CUmemGenericAllocationHandle) handle_[(size_t) b]);
        handle_[(size_t) b] = 0;
        mapped_[(size_t) b] = 0;
    }
    if (r == CUDA_ERROR_OUT_OF_MEMORY) {
        err = "vram pool: CUDA" + std::to_string(dev_) + " has no free memory for " + std::to_string(blocks.size()) +
              " blocks yet (block " + std::to_string(done) + " did not allocate); try again later";
        return Result::Busy;
    }
    err = std::string("vram pool: mapping block ") + std::to_string(blocks[done]) + ": " + cu_name(r);
    return Result::Error;
#else
    return Result::NotLendable;
#endif
}

}  // namespace strata::core
