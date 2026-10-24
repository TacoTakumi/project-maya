// src/ds4/ds4_lend.cu - lending DeepSeek's expert pool to another program (see ds4_model.hpp's lend namespace): the
// choice of a card's blocks, the slot tables' changes, and the VramLender over a Site (ds4_model.cu's over the model).
// Host code only: the pool (core::VramPool) does the driver's part.
#include "ds4_model.hpp"
#include "strata/core/vram_pool.hpp"

#include <algorithm>
#include <cstdio>
#include <utility>

namespace strata::ds4::lend {

namespace {

constexpr size_t kMiB = (size_t) 1 << 20;

float heat_of(const Block& b) {
    float h = 0.0f;
    for (const Slot& s : b.slots) h += s.heat;
    return h;
}

}  // namespace

std::vector<int> candidates(const std::vector<Block>& blocks, int n_regions, int keep) {
    const size_t nb = blocks.size();
    std::vector<int> have((size_t) std::max(0, n_regions), 0);   // a region's live slots (not fixed) that stay
    std::vector<char> open(nb, 0);
    auto count = [&](const Block& b) {
        for (const Slot& s : b.slots)
            if (s.live && !s.fixed && s.region >= 0 && s.region < n_regions) ++have[(size_t) s.region];
    };
    for (size_t i = 0; i < nb; ++i) {
        const Block& b = blocks[i];
        if (!b.mapped) continue;
        bool fixed = false;
        for (const Slot& s : b.slots) fixed = fixed || s.fixed;
        if (b.foreign || fixed) count(b);   // (never lent: what it holds stays)
        else open[i] = 1;
    }
    // a region short of its floor keeps open blocks of its live slots until it has it: the one that holds live slots
    // of the most regions still short (one block keeps several floors), then the most of its own, then the lowest
    // index.  The layout and the blocks lent decide, never the heat or which slots hold an expert: the same blocks stay
    // whatever the cache holds, and a lend (of candidates only) leaves the choice for the rest as it was
    for (int r = 0; r < n_regions; ++r) {
        while (have[(size_t) r] < keep) {
            int best = -1, b_short = 0, b_own = 0;
            for (size_t i = 0; i < nb; ++i) {
                if (!open[i]) continue;
                int own = 0;
                std::vector<int> shorts;   // the regions still short it holds live slots of
                for (const Slot& s : blocks[i].slots) {
                    if (!s.live || s.fixed || s.region < 0 || s.region >= n_regions) continue;
                    own += s.region == r;
                    if (have[(size_t) s.region] < keep &&
                        std::find(shorts.begin(), shorts.end(), s.region) == shorts.end())
                        shorts.push_back(s.region);
                }
                const int ns = (int) shorts.size();
                if (own > 0 && (best < 0 || ns > b_short || (ns == b_short && own > b_own))) {
                    best = (int) i;
                    b_short = ns;
                    b_own = own;
                }
            }
            if (best < 0) break;   // (none left: it keeps what it has)
            open[(size_t) best] = 0;
            count(blocks[(size_t) best]);
        }
    }
    std::vector<int> out;
    for (size_t i = 0; i < nb; ++i)
        if (open[i]) out.push_back((int) i);
    return out;
}

Figures figures(const std::vector<Block>& blocks, size_t n_cand, size_t block_bytes) {
    Figures f;
    size_t n = 0;
    for (const Block& b : blocks) n += b.mapped && !b.foreign;
    f.pool = n * block_bytes;
    f.min = (n - std::min(n, n_cand)) * block_bytes;
    f.lendable = core::lendable_bytes(f.pool, 0, f.min, block_bytes);
    return f;
}

std::vector<int> pick(const std::vector<Block>& blocks, const std::vector<int>& cand, int n) {
    int nr = 0;
    for (int c : cand)
        for (const Slot& s : blocks[(size_t) c].slots) nr = std::max(nr, s.region + 1);
    std::vector<int> taken((size_t) nr, 0);   // blocks this lend took from each region
    std::vector<int> left = cand, out;
    std::vector<float> hl;   // their heat, in step with `left`
    for (int c : cand) hl.push_back(heat_of(blocks[(size_t) c]));
    while ((int) out.size() < n && !left.empty()) {
        size_t best = 0;
        int best_t = 0;
        for (size_t i = 0; i < left.size(); ++i) {
            int t = 0;   // the most any of its regions gave so far
            for (const Slot& s : blocks[(size_t) left[i]].slots)
                if (s.region >= 0) t = std::max(t, taken[(size_t) s.region]);
            if (i == 0 || t < best_t || (t == best_t && hl[i] < hl[best]) ||
                (t == best_t && hl[i] == hl[best] && left[i] < left[best])) {
                best = i;
                best_t = t;
            }
        }
        const int b = left[best];
        out.push_back(b);
        std::vector<int> seen;   // (a region once a block)
        for (const Slot& s : blocks[(size_t) b].slots)
            if (s.region >= 0 && std::find(seen.begin(), seen.end(), s.region) == seen.end()) {
                seen.push_back(s.region);
                ++taken[(size_t) s.region];
            }
        left.erase(left.begin() + (std::ptrdiff_t) best);
        hl.erase(hl.begin() + (std::ptrdiff_t) best);
    }
    return out;
}

void slots_out(std::vector<int>& slot, std::vector<int>& out, std::vector<int>& free, int first, int count,
               std::vector<int>& dropped) {
    for (int e = 0; e < (int) slot.size(); ++e) {
        const int s = slot[(size_t) e];
        if (s >= first && s < first + count) {
            slot[(size_t) e] = -1;
            dropped.push_back(e);
        }
    }
    for (int s = first; s < first + count; ++s) {
        ++out[(size_t) s];
        free.erase(std::remove(free.begin(), free.end(), s), free.end());
    }
}

void slots_back(std::vector<int>& out, std::vector<int>& free, int first, int count) {
    for (int s = first; s < first + count; ++s)
        if (out[(size_t) s] > 0 && --out[(size_t) s] == 0) free.push_back(s);
}

std::vector<core::LendCard> Lender::lend_cards() {
    std::vector<core::LendCard> v;
    for (int dev : site_->cards()) {
        core::LendCard c;
        c.dev = dev;
        const core::VramPool* P = site_->pool(dev);
        if (P != nullptr && P->lendable()) {
            std::vector<Block> B;
            int nr = 0;
            site_->view(dev, B, nr);
            const size_t n = candidates(B, nr, keep_).size();
            c.block_mib = P->block_bytes() / kMiB;
            c.lendable_mib = figures(B, n, P->block_bytes()).lendable / kMiB;
        }
        v.push_back(c);
    }
    return v;
}

core::LendResult Lender::lend(int dev, std::size_t mib) {
    core::LendResult res;
    core::VramPool* P = site_->pool(dev);
    if (P == nullptr || !P->lendable() || mib == 0) return res;
    site_->quiesce();   // (the spare and the slot tables still while the blocks are chosen and taken)
    std::vector<Block> B;
    int nr = 0;
    site_->view(dev, B, nr);
    const std::vector<int> cand = candidates(B, nr, keep_);
    const size_t block = P->block_bytes();
    res.max_mib = figures(B, cand.size(), block).lendable / kMiB;
    const size_t n = (mib * kMiB + block - 1) / block;
    if (n > cand.size()) return res;
    const std::vector<int> blocks = pick(B, cand, (int) n);
    std::string err;
    if (!site_->take(dev, blocks, err)) {
        std::fprintf(stderr, "ds4 lend: CUDA%d: %s\n", dev, err.c_str());
        return res;
    }
    if (P->unmap(blocks, err) != core::VramPool::Result::Ok) {   // (the ones it did unmap mapped again; slots back)
        std::fprintf(stderr, "ds4 lend: CUDA%d: %s\n", dev, err.c_str());
        std::vector<int> gone;
        for (int b : blocks)
            if (!P->mapped(b)) gone.push_back(b);
        std::string e2;
        if (!gone.empty() && P->map(gone, e2) != core::VramPool::Result::Ok)
            std::fprintf(stderr, "ds4 lend: CUDA%d: %zu blocks not mapped again: %s\n", dev, gone.size(), e2.c_str());
        else
            site_->put(dev, blocks);
        return res;
    }
    res.ok = true;
    res.mib = n * block / kMiB;
    res.token = next_++;
    res.max_mib = 0;
    out_[res.token] = Out{dev, blocks};
    return res;
}

core::GiveBack Lender::give_back(int dev, std::uint64_t token) {
    const auto it = out_.find(token);
    if (it == out_.end() || it->second.dev != dev) return core::GiveBack::returned;   // (nothing of ours out)
    core::VramPool* P = site_->pool(dev);
    std::string err;
    const core::VramPool::Result r = P->map(it->second.blocks, err);
    if (r != core::VramPool::Result::Ok) {   // (Busy: held by someone; an error: logged, the lease stays all the same)
        if (r != core::VramPool::Result::Busy) std::fprintf(stderr, "ds4 lend: CUDA%d: %s\n", dev, err.c_str());
        return core::GiveBack::busy;
    }
    site_->put(dev, it->second.blocks);
    out_.erase(it);
    return core::GiveBack::returned;
}

}  // namespace strata::ds4::lend
