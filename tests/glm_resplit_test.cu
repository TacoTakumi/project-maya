// tests/glm_resplit_test.cu - the room check of a whole-card lend (src/core/glm_resplit.cu, resplit_room), on figures
// alone: no model, no GPU.  A marked card passes when a neighbour part's pool, above its own floor, holds the card's
// dense weights, KV and pool floor, and the merged RAM tier (the neighbour's as pinned plus the lent part's, pinned
// anew) stays within what the host can pin under the headroom; the neighbour with the most pool room is chosen.
//  - a pass: the footprint is the sum of what the part holds on its card (dense, KV, activations, drafter, window,
//    pool), in whole MiB, rounded up;
//  - no room on either neighbour: the reason names the need and each neighbour's shortfall;
//  - a device the split does not use: "not in the split";
//  - the merged RAM tier over the host's room: the reason names the tiers and the room;
//  - the neighbour choice: the adjacent part with the most pool room takes the layers, the previous on a tie;
//  - a lone part, and a device two parts share, refuse.
#include "strata/core/glm_model.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using strata::core::ResplitPart;
using strata::core::ResplitRoom;
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
    }
    p[0].pool = 18 * kGiB;
    p[1].pool = 18 * kGiB;
    p[2].pool = 10 * kGiB;
    p[3].pool = 10 * kGiB;
    p[4].pool = 10 * kGiB;
    p[4].drafter = 1 * kGiB;   // the last part carries the drafter
    return p;
}

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
    // the merged tier: part 1's 8 GiB as pinned + part 2's 8 GiB pinned anew; the host has 10 GiB free less a 6 GiB
    // headroom = 4 GiB of room for the new pinning: 4 GiB short
    const ResplitRoom r = resplit_room(p, 2, 10 * kGiB, 6 * kGiB);
    CHECK(!r.ok, "the RAM headroom fail");
    CHECK(has(r.reason, "RAM"), "names RAM: %s", r.reason.c_str());
    CHECK(has(r.reason, "16384 MiB"), "the merged tier: %s", r.reason.c_str());
    CHECK(has(r.reason, "12288 MiB"), "the room (8 GiB pinned + 4 GiB free under the headroom): %s", r.reason.c_str());
    CHECK(has(r.reason, "short 4096 MiB"), "the shortfall: %s", r.reason.c_str());
    CHECK(has(r.reason, "headroom 6144 MiB"), "the headroom figure: %s", r.reason.c_str());
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
    // the RAM rule reads the chosen neighbour's tier
    p[3].ram = 20 * kGiB;
    p[1].pool = p[1].floor + 1 * kGiB;   // part 1 out of the running
    r = resplit_room(p, 2, 14 * kGiB, 6 * kGiB);
    CHECK(r.ok && r.into == 3, "part 3 takes it, its 20 GiB tier as pinned: %s", r.reason.c_str());
    CHECK(!resplit_room(p, 2, 14 * kGiB - 1, 6 * kGiB).ok, "the lent part's 8 GiB tier still must fit the free RAM");
}

}  // namespace

int main() {
    test_pass();
    test_no_room();
    test_not_in_split();
    test_ram_headroom();
    test_neighbour_choice();
    if (g_fail == 0) std::printf("glm_resplit_test: all passed\n");
    return g_fail == 0 ? 0 : 1;
}
