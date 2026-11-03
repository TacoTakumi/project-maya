// tests/glm_resplit_test.cu - the room check of a whole-card lend (src/core/glm_resplit.cu, resplit_room), on figures
// alone: no model, no GPU.  A marked card passes when a neighbour part's pool, above its own floor, holds the card's
// dense weights, KV and pool floor, and the lent part's RAM tier (pinned anew by the neighbour for the merged tier)
// fits in the RAM free under the headroom; the neighbour with the most pool room is chosen.  A pass sizes the merged
// RAM tier as a first load would: the free RAM under the headroom, bounded by what the merged layers can use and
// never below the two tiers' sum.
//  - a pass: the footprint is the sum of what the part holds on its card (dense, KV, activations, drafter, window,
//    pool), in whole MiB, rounded up;
//  - no room on either neighbour: the reason names the need and each neighbour's shortfall;
//  - a device the split does not use: "not in the split";
//  - the lent part's tier over the free RAM under the headroom: the reason names the tier, the room, the headroom and
//    the shortfall; exactly the room passes, one byte short fails; a part without a tier passes whatever the free RAM;
//  - the merged tier: free RAM large -> the layers' bound (every merged expert outside the merged pool's main slots,
//    plus the spares); free RAM between the sum and that bound -> the free RAM under the headroom; free RAM under the
//    sum but holding the lent tier -> the sum; free RAM unknown (0) -> the sum, the rule skipped;
//  - the neighbour choice: the adjacent part with the most pool room takes the layers, the previous on a tie;
//  - a lone part, and a device two parts share, refuse.
#include "strata/core/glm_model.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using strata::core::RamBound;
using strata::core::ResplitPart;
using strata::core::ResplitRoom;
using strata::core::ram_bound_name;
using strata::core::resplit_room;

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

constexpr uint64_t kMiB = (uint64_t) 1 << 20;
constexpr uint64_t kGiB = (uint64_t) 1 << 30;

bool has(const std::string& s, const char* what) { return s.find(what) != std::string::npos; }

// a five-card split: two 24 GiB cards, a 16 GiB card in the middle, two 16 GiB cards after it
std::vector<ResplitPart> five() {
    std::vector<ResplitPart> p(5);
    for (int i = 0; i < 5; ++i) {
        p[(size_t) i].dev = i;
        p[(size_t) i].l0 = i * 10;
        p[(size_t) i].l1 = i * 10 + 10;
        p[(size_t) i].weights = 2 * kGiB;
        p[(size_t) i].kv = 512 * kMiB;
        p[(size_t) i].act = 100 * kMiB;
        p[(size_t) i].drafter = 0;
        p[(size_t) i].window = 50 * kMiB;
        p[(size_t) i].floor = 1 * kGiB;
        p[(size_t) i].ram = 8 * kGiB;
        // the tier's bound: 20 GiB of routed experts a part, 1 GiB of spares, the pool's main slots 2 GiB under the
        // pool (the tail)
        p[(size_t) i].experts = 20 * kGiB;
        p[(size_t) i].spares = 1 * kGiB;
    }
    p[0].pool = 18 * kGiB;
    p[1].pool = 18 * kGiB;
    p[2].pool = 10 * kGiB;
    p[3].pool = 10 * kGiB;
    p[4].pool = 10 * kGiB;
    for (int i = 0; i < 5; ++i) p[(size_t) i].held = p[(size_t) i].pool - 2 * kGiB;
    p[4].drafter = 1 * kGiB;   // the last part carries the drafter
    return p;
}

// the merged tier's layers' bound when card 2 goes into part 1: the two parts' experts and spares, less the main slots
// the merged pool keeps (part 1's 16 GiB less card 2's dense 2 GiB + KV 512 MiB that move in) = 42 - 13.5 GiB
constexpr uint64_t kCap21 = 42 * kGiB - (16 * kGiB - 2 * kGiB - 512 * kMiB);

void test_pass() {
    const auto p = five();
    // 64 GiB free on the host, a 6 GiB headroom: room enough for any tier here
    const ResplitRoom r = resplit_room(p, 2, 64 * kGiB, 6 * kGiB);
    CHECK(r.ok, "card 2 should pass: %s", r.reason.c_str());
    // the footprint: everything the part holds on its card, in MiB
    const uint64_t want = (2 * kGiB + 512 * kMiB + 100 * kMiB + 0 + 50 * kMiB + 10 * kGiB) / kMiB;
    CHECK(r.whole_mib == want, "footprint %llu, want %llu", (unsigned long long) r.whole_mib, (unsigned long long) want);
    CHECK(r.into == 1, "the neighbour with the most room (part 1, 17 GiB above its floor), got %d", r.into);
    CHECK(r.reason.empty(), "a pass has no reason: %s", r.reason.c_str());
    // the drafter counts in the footprint too (the last part)
    const ResplitRoom l = resplit_room(p, 4, 64 * kGiB, 6 * kGiB);
    CHECK(l.ok && l.into == 3, "card 4 into part 3: %s", l.reason.c_str());
    CHECK(l.whole_mib == want + 1024, "card 4 footprint %llu", (unsigned long long) l.whole_mib);
    // a footprint of a part MiB is rounded up
    auto q = p;
    q[2].act += 1;
    CHECK(resplit_room(q, 2, 64 * kGiB, 6 * kGiB).whole_mib == want + 1, "rounded up to the next MiB");
}

void test_no_room() {
    auto p = five();
    // the pools of both neighbours barely above their floors: card 2 needs 2 GiB + 512 MiB + 1 GiB = 3584 MiB
    p[1].pool = p[1].floor + 2 * kGiB;   // 2048 MiB of room: 1536 short
    p[3].pool = p[3].floor + 1 * kGiB;   // 1024 MiB of room: 2560 short
    const ResplitRoom r = resplit_room(p, 2, 64 * kGiB, 6 * kGiB);
    CHECK(!r.ok, "no room on either neighbour");
    CHECK(r.whole_mib == 0 && r.into < 0, "a fail carries no footprint and no neighbour");
    CHECK(has(r.reason, "3584 MiB"), "the need: %s", r.reason.c_str());
    CHECK(has(r.reason, "CUDA1") && has(r.reason, "2048 MiB") && has(r.reason, "short 1536 MiB"), "neighbour 1: %s",
          r.reason.c_str());
    CHECK(has(r.reason, "CUDA3") && has(r.reason, "1024 MiB") && has(r.reason, "short 2560 MiB"), "neighbour 3: %s",
          r.reason.c_str());
    // a pool under its own floor has no room at all
    p[1].pool = p[1].floor / 2;
    const ResplitRoom u = resplit_room(p, 2, 64 * kGiB, 6 * kGiB);
    CHECK(!u.ok && has(u.reason, "CUDA1 has 0 MiB"), "under the floor: %s", u.reason.c_str());
    // exactly enough passes
    p[1].pool = p[1].floor + 3584 * kMiB;
    CHECK(resplit_room(p, 2, 64 * kGiB, 6 * kGiB).ok, "room equal to the need passes");
}

void test_not_in_split() {
    const auto p = five();
    const ResplitRoom r = resplit_room(p, 7, 64 * kGiB, 6 * kGiB);
    CHECK(!r.ok && r.reason == "not in the split", "device 7: %s", r.reason.c_str());
    const ResplitRoom n = resplit_room(p, -1, 64 * kGiB, 6 * kGiB);
    CHECK(!n.ok && n.reason == "not in the split", "device -1: %s", n.reason.c_str());
    // a device two parts share is not lent whole
    auto q = p;
    q[3].dev = 2;
    const ResplitRoom s = resplit_room(q, 2, 64 * kGiB, 6 * kGiB);
    CHECK(!s.ok && has(s.reason, "2 parts"), "a shared device: %s", s.reason.c_str());
    // one part alone has no neighbour
    std::vector<ResplitPart> one(1, p[0]);
    const ResplitRoom o = resplit_room(one, 0, 64 * kGiB, 6 * kGiB);
    CHECK(!o.ok && has(o.reason, "no neighbour"), "a lone part: %s", o.reason.c_str());
}

void test_ram_headroom() {
    auto p = five();
    // part 2's 8 GiB tier, pinned anew by part 1 for the merged tier; the host has 10 GiB free less a 6 GiB headroom
    // = 4 GiB of room for the new pinning: 4 GiB short
    const ResplitRoom r = resplit_room(p, 2, 10 * kGiB, 6 * kGiB);
    CHECK(!r.ok, "the RAM headroom fail");
    CHECK(has(r.reason, "RAM"), "names RAM: %s", r.reason.c_str());
    CHECK(has(r.reason, "CUDA2") && has(r.reason, "8192 MiB"), "the lent part's tier: %s", r.reason.c_str());
    CHECK(has(r.reason, "4096 MiB free under the headroom"), "the room (4 GiB free under the headroom): %s",
          r.reason.c_str());
    CHECK(has(r.reason, "short 4096 MiB"), "the shortfall: %s", r.reason.c_str());
    CHECK(has(r.reason, "headroom 6144 MiB"), "the headroom figure: %s", r.reason.c_str());
    CHECK(r.ram == 0, "a fail carries no merged tier");
    // exactly the room passes; one byte less fails
    CHECK(resplit_room(p, 2, 14 * kGiB, 6 * kGiB).ok, "8 GiB free above the headroom holds the 8 GiB tier");
    CHECK(!resplit_room(p, 2, 14 * kGiB - 1, 6 * kGiB).ok, "a byte short fails");
    // free RAM under the headroom alone: nothing may be pinned
    CHECK(!resplit_room(p, 2, 5 * kGiB, 6 * kGiB).ok, "free RAM below the headroom");
    // a part without a RAM tier pins nothing new
    p[2].ram = 0;
    CHECK(resplit_room(p, 2, 5 * kGiB, 6 * kGiB).ok, "no tier to pin: passes whatever the free RAM");
    // an unknown free RAM figure (0: the host could not say) is not checked
    p[2].ram = 8 * kGiB;
    CHECK(resplit_room(p, 2, 0, 6 * kGiB).ok, "free RAM unknown: the RAM rule is skipped");
}

void test_merged_tier() {
    const auto p = five();
    // the two tiers' sum, 16 GiB; the layers' bound 28.5 GiB (kCap21)
    const uint64_t sum = 16 * kGiB;
    CHECK(kCap21 == 29184 * kMiB, "the layers' bound of the figures: %llu MiB", (unsigned long long) (kCap21 / kMiB));
    // free RAM large (64 GiB less the 6 GiB headroom = 58 GiB): the layers' bound
    ResplitRoom r = resplit_room(p, 2, 64 * kGiB, 6 * kGiB);
    CHECK(r.ok && r.ram == kCap21, "free RAM large: the layers' bound %llu MiB, got %llu MiB (%s)",
          (unsigned long long) (kCap21 / kMiB), (unsigned long long) (r.ram / kMiB), r.reason.c_str());
    CHECK(r.bound == RamBound::kLayers, "the bound named: %s", ram_bound_name(r.bound));
    // free RAM between the sum and the bound (30 GiB less the headroom = 24 GiB): the free-derived figure
    r = resplit_room(p, 2, 30 * kGiB, 6 * kGiB);
    CHECK(r.ok && r.ram == 24 * kGiB, "free RAM 24 GiB under the headroom: tier %llu MiB", (unsigned long long) (r.ram / kMiB));
    CHECK(r.bound == RamBound::kFree, "the bound named: %s", ram_bound_name(r.bound));
    // exactly the bound: the layers' bound names it
    r = resplit_room(p, 2, kCap21 + 6 * kGiB, 6 * kGiB);
    CHECK(r.ok && r.ram == kCap21 && r.bound == RamBound::kLayers, "free RAM equal to the bound: the layers' bound");
    r = resplit_room(p, 2, kCap21 + 6 * kGiB - 1, 6 * kGiB);
    CHECK(r.ok && r.ram == kCap21 - 1 && r.bound == RamBound::kFree, "a byte under the bound: the free RAM");
    // free RAM under the sum (20 GiB less the headroom = 14 GiB) but holding the lent tier: the sum
    r = resplit_room(p, 2, 20 * kGiB, 6 * kGiB);
    CHECK(r.ok && r.ram == sum, "free RAM 14 GiB under the headroom: the sum, got %llu MiB", (unsigned long long) (r.ram / kMiB));
    CHECK(r.bound == RamBound::kSum, "the bound named: %s", ram_bound_name(r.bound));
    // free RAM below the lent tier plus the headroom (13 GiB: 7 GiB of room for the 8 GiB tier): the fail names it
    r = resplit_room(p, 2, 13 * kGiB, 6 * kGiB);
    CHECK(!r.ok && has(r.reason, "RAM") && has(r.reason, "short 1024 MiB"), "1 GiB short: %s", r.reason.c_str());
    // free RAM unknown (0): the sum, as the skipped rule had it
    r = resplit_room(p, 2, 0, 6 * kGiB);
    CHECK(r.ok && r.ram == sum && r.bound == RamBound::kSum, "free RAM unknown: the sum, got %llu MiB",
          (unsigned long long) (r.ram / kMiB));
    // parts without the bound's figures (an older caller): the sum, whatever the free RAM
    auto q = p;
    for (ResplitPart& x : q) x.experts = x.spares = x.held = 0;
    r = resplit_room(q, 2, 64 * kGiB, 6 * kGiB);
    CHECK(r.ok && r.ram == sum && r.bound == RamBound::kSum, "no bound figures: the sum");
    // the bound follows the neighbour chosen: part 3 (held 8 GiB) takes the layers when it has the most room
    q = p;
    q[1].pool = q[1].floor + 4 * kGiB;
    q[3].pool = q[3].floor + 5 * kGiB;
    r = resplit_room(q, 2, 64 * kGiB, 6 * kGiB);
    const uint64_t cap23 = 42 * kGiB - (8 * kGiB - 2 * kGiB - 512 * kMiB);
    CHECK(r.ok && r.into == 3 && r.ram == cap23, "into part 3: the bound %llu MiB, got %llu MiB",
          (unsigned long long) (cap23 / kMiB), (unsigned long long) (r.ram / kMiB));
    // the lent part's dense weights and KV over the neighbour's main slots: no main slot kept, every expert and spare
    q = p;
    q[1].held = 1 * kGiB;
    r = resplit_room(q, 2, 64 * kGiB, 6 * kGiB);
    CHECK(r.ok && r.ram == 42 * kGiB, "no main slots kept: the bound is every expert and spare, got %llu MiB",
          (unsigned long long) (r.ram / kMiB));
    // the bound names
    CHECK(std::string(ram_bound_name(RamBound::kSum)) != ram_bound_name(RamBound::kFree) &&
              std::string(ram_bound_name(RamBound::kFree)) != ram_bound_name(RamBound::kLayers),
          "three bound names");
}

void test_neighbour_choice() {
    auto p = five();
    // part 3 has more room than part 1: it takes the layers
    p[1].pool = p[1].floor + 4 * kGiB;
    p[3].pool = p[3].floor + 5 * kGiB;
    ResplitRoom r = resplit_room(p, 2, 64 * kGiB, 6 * kGiB);
    CHECK(r.ok && r.into == 3, "the next part has the most room: into %d (%s)", r.into, r.reason.c_str());
    // a tie: the previous part
    p[1].pool = p[3].pool;
    r = resplit_room(p, 2, 64 * kGiB, 6 * kGiB);
    CHECK(r.ok && r.into == 1, "a tie goes to the previous part: into %d", r.into);
    // the most room, even when the other one would also hold the card
    p[1].pool = p[1].floor + 6 * kGiB;
    r = resplit_room(p, 2, 64 * kGiB, 6 * kGiB);
    CHECK(r.ok && r.into == 1, "the most room wins: into %d", r.into);
    // the first part has one neighbour only; the last too
    CHECK(resplit_room(p, 0, 64 * kGiB, 6 * kGiB).into == 1, "card 0 into part 1");
    CHECK(resplit_room(p, 4, 64 * kGiB, 6 * kGiB).into == 3, "card 4 into part 3");
    // the merged tier reads the chosen neighbour's tier: part 3's 20 GiB as pinned plus the lent 8 GiB
    p[3].ram = 20 * kGiB;
    p[1].pool = p[1].floor + 1 * kGiB;   // part 1 out of the running
    r = resplit_room(p, 2, 14 * kGiB, 6 * kGiB);
    CHECK(r.ok && r.into == 3, "part 3 takes it, its 20 GiB tier as pinned: %s", r.reason.c_str());
    CHECK(r.ram == 28 * kGiB && r.bound == RamBound::kSum, "the sum reads part 3's tier: %llu MiB",
          (unsigned long long) (r.ram / kMiB));
    CHECK(!resplit_room(p, 2, 14 * kGiB - 1, 6 * kGiB).ok, "the lent part's 8 GiB tier still must fit the free RAM");
}

}  // namespace

int main() {
    test_pass();
    test_no_room();
    test_not_in_split();
    test_ram_headroom();
    test_merged_tier();
    test_neighbour_choice();
    if (g_fail == 0) std::printf("glm_resplit_test: all passed\n");
    return g_fail == 0 ? 0 : 1;
}
