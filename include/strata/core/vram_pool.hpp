// include/strata/core/vram_pool.hpp - one card's expert slots in a VRAM pool whose blocks can be given back to the
// driver and mapped again at the same addresses.
//
// The pool reserves one virtual address range at load (cuMemAddressReserve) and backs it with physical blocks of one
// size (cuMemCreate + cuMemMap + cuMemSetAccess, the owning device only).  The block size is a multiple of the
// device's allocation granularity and holds whole slots: no slot straddles two blocks, so unmapping a block takes
// out whole slots and nothing else.
//
// Layout.  The caller gives its slot groups as regions (one per layer: slot_bytes, n_slots), in address order.  A
// block holds per_block = floor(block_bytes / slot_bytes) slots of one size, back to back from the block's start;
// the rest of the block (less than one slot) is unused.  Consecutive regions of the same slot size share blocks, so
// a region may start inside a block (first_pos) and the unused bytes are at most one partial slot a block.  Slot i of
// region r is at
//     origin(r) + (g / per_block) * block_bytes + (g % per_block) * slot_bytes,   g = first_pos + i
// which is slot_ptr(r, i).  That is base + i * slot_bytes only where linear(r) holds (per_block * slot_bytes ==
// block_bytes, or the region within one block); otherwise the engine addresses slots through slot_ptr / slot_index,
// or the same formula on the device.  Addresses never change: a block mapped back is at the same place.
//
// Lending.  unmap() releases a chosen set of blocks to the driver (the card's free memory rises at once); map()
// backs them again.  cuMemUnmap does not wait for kernels in flight: the caller takes the blocks' slots out of use
// and synchronizes every stream that may touch them BEFORE unmap().  map() all-or-nothing: if the driver has no
// physical memory for one block (another process holds it), the blocks this call mapped are released again, every
// block of the set stays unmapped, and the call answers Busy - retry later.  Not thread-safe: one caller at a time.
//
// Fallback.  Without VMM (a HIP build, a device without CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
// Options::lend false, or STRATA_VRAM_POOL_VMM=0) the same layout lives in one plain cudaMalloc (or, where the
// driver refuses one allocation that large, one per region); lendable() is false and unmap / map answer NotLendable.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

struct VramPoolRegion {
    size_t slot_bytes = 0;   // the slot's size and its pitch inside a block (keep it a multiple of the alignment the
                             // engine needs: a block starts at a granularity boundary)
    int n_slots = 0;
};

class VramPool {
public:
    enum class Result { Ok, Busy, NotLendable, Error };

    struct Options {
        bool lend = true;                          // false: the plain allocation, never lends
        size_t min_block_bytes = (size_t) 16 << 20;   // the block size is chosen in [min, max] (rounded to the
        size_t max_block_bytes = (size_t) 64 << 20;   // granularity, never below the largest slot) to waste least
        size_t block_bytes = 0;                    // > 0: this size instead (rounded up to the granularity)
        const char* tag = "vram pool";             // the load log's prefix ("glm fast", "ds4")
        bool log = true;                           // print the one-line summary at create()
    };

    // a region's place in the pool (see the layout above)
    struct RegionLayout {
        size_t slot_bytes = 0;
        int n_slots = 0;
        int per_block = 0;     // slots a block
        int first_block = 0;   // the block holding slot 0
        int first_pos = 0;     // slot 0's position in that block
        uint8_t* origin = nullptr;   // first_block's address
    };

    // the slots a block holds: region, first slot, count
    struct Span {
        int region = 0, first = 0, count = 0;
    };

    VramPool() = default;
    ~VramPool();
    VramPool(const VramPool&) = delete;
    VramPool& operator=(const VramPool&) = delete;

    // Reserves, maps and lays out the regions on device `dev` (a runtime device index).  The caller's current
    // device is kept.  False with `err` on failure (nothing is left allocated).
    bool create(int dev, const std::vector<VramPoolRegion>& regions, const Options& o, std::string& err);
    // Unmaps and releases every block and frees the reservation (or the plain allocation).  The device must be done
    // with the pool's memory.  The destructor calls it.
    void destroy();
    // Whether device `dev` can lend (a CUDA build, VMM supported), and its allocation granularity (2 MiB where it
    // cannot).  No context is created.
    static bool vmm_supported(int dev, size_t* granularity = nullptr);

    bool valid() const { return block_bytes_ > 0; }
    int device() const { return dev_; }
    bool lendable() const { return vmm_; }
    size_t granularity() const { return gran_; }
    size_t block_bytes() const { return block_bytes_; }
    int block_count() const { return (int) mapped_.size(); }
    size_t bytes() const { return (size_t) block_count() * block_bytes_; }   // the reserved range
    size_t slot_bytes_total() const { return used_; }                         // what the slots occupy of it
    bool mapped(int b) const { return b >= 0 && b < block_count() && mapped_[(size_t) b] != 0; }
    int mapped_count() const;
    const std::string& summary() const { return summary_; }   // the load log's line

    int region_count() const { return (int) regions_.size(); }
    const RegionLayout& region(int r) const { return regions_[(size_t) r]; }
    uint8_t* slot_ptr(int r, int i) const {
        const RegionLayout& L = regions_[(size_t) r];
        const size_t g = (size_t) L.first_pos + (size_t) i;
        return L.origin + (g / (size_t) L.per_block) * block_bytes_ + (g % (size_t) L.per_block) * L.slot_bytes;
    }
    int slot_index(int r, const void* p) const;   // the slot of region r holding address p, -1 none
    bool linear(int r) const;                      // slot_ptr(r, i) == slot_ptr(r, 0) + i * slot_bytes for every i
    uint8_t* block_ptr(int b) const { return base_ + (size_t) b * block_bytes_; }   // (VMM pool)
    std::vector<Span> block_slots(int b) const;

    // Releases blocks `blocks` (each mapped, no repeats) to the driver.  See the header's note on synchronizing.
    Result unmap(const std::vector<int>& blocks, std::string& err);
    // Maps blocks `blocks` (each unmapped, no repeats) back at their addresses.  Busy: no physical memory now, the
    // blocks stay unmapped.
    Result map(const std::vector<int>& blocks, std::string& err);

private:
    void release();   // frees the memory, keeps the layout

    int dev_ = -1;
    bool vmm_ = false;
    size_t gran_ = 0, block_bytes_ = 0, used_ = 0;
    uint8_t* base_ = nullptr;                  // the reservation, or the one plain allocation
    std::vector<void*> plain_;                 // the fallback's allocations
    std::vector<RegionLayout> regions_;
    std::vector<char> mapped_;
    std::vector<unsigned long long> handle_;   // a mapped block's physical allocation (CUmemGenericAllocationHandle)
    std::string summary_;
};

}  // namespace strata::core
