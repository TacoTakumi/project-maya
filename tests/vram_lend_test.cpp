// CPU-only checks of the shared lend commands and the lendable rule: a fake engine stands in for a model's expert
// cache, so no HIP/CUDA headers, device queries or allocations.
#include "strata/core/vram_lend.hpp"

#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <vector>

using strata::core::GiveBack;
using strata::core::LendCard;
using strata::core::LendCommands;
using strata::core::LendLineKind;
using strata::core::LendQueue;
using strata::core::LendResult;
using strata::core::VramLender;
using strata::core::lend_line_kind;
using strata::core::lendable_bytes;
#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); return 1; } } while (0)
#define CHECK_REPLY(desk, line, want) do { std::string r_; \
    if (!(desk).handle(line, r_) || r_ != (want)) { \
        std::fprintf(stderr, "line %d: '%s' gave '%s', want '%s'\n", __LINE__, line, r_.c_str(), want); return 1; } \
    } while (0)

namespace {

constexpr size_t M = size_t{1} << 20;

// One card of the fake engine: a pool of whole blocks, the prompt path's borrow and the minimum slots it keeps.  The
// blocks lent leave the pool, as they leave a real cache, so the rule sees the pool after lends.
struct FakeCard {
    int dev;
    size_t pool, borrow, min_slots, block;   // bytes
};

struct FakeEngine : VramLender {
    std::vector<FakeCard> cards;
    std::map<uint64_t, std::pair<int, size_t>> leases;   // token -> (dev, bytes)
    std::set<uint64_t> busy;                             // tokens whose memory someone still holds
    uint64_t next_token = 1000;
    size_t refuse_max = SIZE_MAX;   // below SIZE_MAX: the next lend is refused with this max (a pool that shrank)

    FakeCard* card(int dev) {
        for (FakeCard& c : cards)
            if (c.dev == dev) return &c;
        return nullptr;
    }
    std::vector<LendCard> lend_cards() override {
        std::vector<LendCard> out;
        for (const FakeCard& c : cards)
            out.push_back({c.dev, lendable_bytes(c.pool, c.borrow, c.min_slots, c.block) / M, c.block / M});
        return out;
    }
    LendResult lend(int dev, size_t mib) override {
        LendResult r;
        FakeCard* c = card(dev);
        if (!c) return r;
        if (refuse_max != SIZE_MAX) {
            r.max_mib = refuse_max;
            refuse_max = SIZE_MAX;
            return r;
        }
        c->pool -= mib * M;
        r.ok = true;
        r.mib = mib;
        r.token = next_token++;
        leases[r.token] = {dev, mib * M};
        return r;
    }
    GiveBack give_back(int dev, uint64_t token) override {
        if (busy.count(token)) return GiveBack::busy;
        auto it = leases.find(token);
        if (it == leases.end() || it->second.first != dev) return GiveBack::busy;   // the handler never asks this
        card(dev)->pool += it->second.second;
        leases.erase(it);
        return GiveBack::returned;
    }
};

// "LENT <id> <dev> <mib>\n" -> id (0 when the reply is not a LENT line)
uint64_t lent_id(const std::string& reply, int dev, size_t mib) {
    unsigned long long id = 0;
    int d = -1;
    size_t m = 0;
    char tail = 0;
    if (std::sscanf(reply.c_str(), "LENT %llu %d %zu%c", &id, &d, &m, &tail) != 4 || tail != '\n') return 0;
    return d == dev && m == mib ? (uint64_t) id : 0;
}

}  // namespace

int main() {
    // ---- the lendable rule: pool - prompt borrow - minimum slots, clamped at 0, down to whole blocks (bytes)
    CHECK(lendable_bytes(640 * M, 64 * M, 128 * M, 64 * M) == 448 * M);
    CHECK(lendable_bytes(1000 * M, 250 * M, 300 * M, 100 * M) == 400 * M);   // 450 MiB -> 4 blocks of 100
    CHECK(lendable_bytes(1000 * M, 250 * M, 300 * M, 100 * M - 1) == 4 * (100 * M - 1));
    CHECK(lendable_bytes(10, 0, 0, 3) == 9);
    CHECK(lendable_bytes(192 * M, 64 * M, 128 * M, 64 * M) == 0);   // exactly used up
    CHECK(lendable_bytes(100 * M, 64 * M, 128 * M, 64 * M) == 0);   // short: 0, not a wrapped huge value
    CHECK(lendable_bytes(100 * M, 200 * M, 0, 64 * M) == 0);
    CHECK(lendable_bytes(100 * M, 0, 200 * M, 64 * M) == 0);
    CHECK(lendable_bytes(SIZE_MAX, SIZE_MAX, SIZE_MAX, 64 * M) == 0);
    CHECK(lendable_bytes(255 * M, 64 * M, 128 * M, 64 * M) == 0);   // 63 MiB left: less than one block
    CHECK(lendable_bytes(640 * M, 64 * M, 128 * M, 0) == 0);        // no block size (no lending there): nothing
    // the engine passes the pool after lends: 3 blocks lent from the 640 MiB pool leave 256 MiB lendable
    CHECK(lendable_bytes(640 * M - 192 * M, 64 * M, 128 * M, 64 * M) == 256 * M);

    FakeEngine eng;
    eng.cards = {{0, 640 * M, 64 * M, 128 * M, 64 * M},      // lendable 448 MiB, blocks of 64
                 {2, 1000 * M, 250 * M, 300 * M, 100 * M},   // lendable 400 MiB, blocks of 100
                 {3, 100 * M, 64 * M, 128 * M, 64 * M}};     // lendable 0
    LendCommands desk(eng);

    // ---- LENDABLE: one line per card, then END
    CHECK_REPLY(desk, "LENDABLE", "LENDABLE dev=0 lendable=448 lent=0\n"
                                  "LENDABLE dev=2 lendable=400 lent=0\n"
                                  "LENDABLE dev=3 lendable=0 lent=0\n"
                                  "END\n");
    CHECK_REPLY(desk, "  LENDABLE \r", "LENDABLE dev=0 lendable=448 lent=0\n"
                                       "LENDABLE dev=2 lendable=400 lent=0\n"
                                       "LENDABLE dev=3 lendable=0 lent=0\n"
                                       "END\n");

    // ---- LEND: rounded up to whole blocks (100 -> 2 blocks of 64), the lease named by a fresh id
    std::string r;
    CHECK(desk.handle("LEND 0 100", r));
    const uint64_t a = lent_id(r, 0, 128);
    CHECK(a != 0);
    CHECK(desk.handle("LEND 0 64", r));   // exactly one block: no rounding
    const uint64_t b = lent_id(r, 0, 64);
    CHECK(b != 0 && b != a);
    CHECK(desk.open_leases() == 2);
    CHECK(eng.leases.size() == 2);
    // lendable shrinks by what is out (448 - 192), lent shows it
    CHECK_REPLY(desk, "LENDABLE", "LENDABLE dev=0 lendable=256 lent=192\n"
                                  "LENDABLE dev=2 lendable=400 lent=0\n"
                                  "LENDABLE dev=3 lendable=0 lent=0\n"
                                  "END\n");

    // ---- ERR MAX: more than the card's lendable, with the lendable it has now
    CHECK_REPLY(desk, "LEND 0 1000", "ERR MAX 256\n");
    CHECK_REPLY(desk, "LEND 0 257", "ERR MAX 256\n");
    CHECK_REPLY(desk, "LEND 2 401", "ERR MAX 400\n");
    CHECK_REPLY(desk, "LEND 3 1", "ERR MAX 0\n");
    CHECK(desk.open_leases() == 2);
    // the engine may still refuse (its pool shrank since it reported): its max is the answer
    eng.refuse_max = 64;
    CHECK_REPLY(desk, "LEND 0 128", "ERR MAX 64\n");
    CHECK(desk.open_leases() == 2);

    // ---- RETURN: two leases on one card come back independently, in either order
    const std::string ra = std::to_string(a), rb = std::to_string(b);
    eng.busy.insert(eng.leases.rbegin()->first);   // b's token (the later one): its memory is still held
    CHECK_REPLY(desk, ("RETURN " + rb).c_str(), ("ERR BUSY " + rb + "\n").c_str());
    CHECK(desk.open_leases() == 2);   // a busy lease stays open
    CHECK_REPLY(desk, ("RETURN " + ra).c_str(), ("RETURNED " + ra + "\n").c_str());
    CHECK(desk.open_leases() == 1);
    CHECK_REPLY(desk, "LENDABLE", "LENDABLE dev=0 lendable=384 lent=64\n"
                                  "LENDABLE dev=2 lendable=400 lent=0\n"
                                  "LENDABLE dev=3 lendable=0 lent=0\n"
                                  "END\n");
    CHECK_REPLY(desk, ("RETURN " + ra).c_str(), ("ERR UNKNOWN " + ra + "\n").c_str());   // already returned
    CHECK_REPLY(desk, ("RETURN " + rb).c_str(), ("ERR BUSY " + rb + "\n").c_str());      // still held: retry
    eng.busy.clear();
    CHECK_REPLY(desk, ("RETURN " + rb).c_str(), ("RETURNED " + rb + "\n").c_str());      // later: the same id
    CHECK(desk.open_leases() == 0 && eng.leases.empty());
    CHECK_REPLY(desk, ("RETURN " + rb).c_str(), ("ERR UNKNOWN " + rb + "\n").c_str());

    // two leases on another card, the later one returned first
    CHECK(desk.handle("LEND 2 150", r));   // 2 blocks of 100
    const uint64_t c = lent_id(r, 2, 200);
    CHECK(desk.handle("LEND 2 1", r));     // 1 block
    const uint64_t d = lent_id(r, 2, 100);
    CHECK(c != 0 && d != 0 && c != d && c != a && c != b && d != a && d != b);
    CHECK_REPLY(desk, "LEND 2 101", "ERR MAX 100\n");
    CHECK_REPLY(desk, ("RETURN " + std::to_string(d)).c_str(), ("RETURNED " + std::to_string(d) + "\n").c_str());
    CHECK_REPLY(desk, "LENDABLE", "LENDABLE dev=0 lendable=448 lent=0\n"
                                  "LENDABLE dev=2 lendable=200 lent=200\n"
                                  "LENDABLE dev=3 lendable=0 lent=0\n"
                                  "END\n");
    CHECK_REPLY(desk, ("RETURN " + std::to_string(c)).c_str(), ("RETURNED " + std::to_string(c) + "\n").c_str());
    CHECK(desk.handle("LEND 2 400", r) && lent_id(r, 2, 400) != 0);   // the whole lendable
    CHECK_REPLY(desk, "LEND 2 1", "ERR MAX 0\n");

    // ---- UNKNOWN: ids never handed out (the ids are opaque: any one word is looked up)
    CHECK_REPLY(desk, "RETURN 999999999", "ERR UNKNOWN 999999999\n");
    CHECK_REPLY(desk, "RETURN 0", "ERR UNKNOWN 0\n");
    CHECK_REPLY(desk, "RETURN abc", "ERR UNKNOWN abc\n");

    // ---- malformed lend commands: answered with an ERR, never passed on
    for (const char* bad : {"LEND", "LEND 0", "LEND x 10", "LEND -1 10", "LEND 0 0", "LEND 0 -5", "LEND 0 +5",
                            "LEND 0 1.5", "LEND 0 10MiB", "LEND 0 10 extra", "LEND 0 99999999999999999999999",
                            "LEND 99999999999 10"})
        CHECK_REPLY(desk, bad, "ERR MALFORMED LEND <dev> <mib>\n");
    for (const char* bad : {"RETURN", "RETURN 1 2"}) CHECK_REPLY(desk, bad, "ERR MALFORMED RETURN <id>\n");
    CHECK_REPLY(desk, "LENDABLE 0", "ERR MALFORMED LENDABLE\n");
    CHECK_REPLY(desk, "LEND 5 10", "ERR NODEV 5\n");   // well formed, but no such card
    CHECK(eng.leases.size() == 1);

    // ---- not a lend command: false and no reply, for the engine's own loop to handle
    for (const char* other : {"", "   ", "GEN 1 2 3", "QUIT", "STOP", "LOGP 2 1,2,3", "GENI 1 2", "LENDX 0 10",
                              "lend 0 10", "RETURNED 1", "LENDABLES"}) {
        std::string o = "stale";
        CHECK(!desk.handle(other, o));
        CHECK(o.empty());
    }

    // ---- ids stay unique for the process life, also across another command table
    FakeEngine eng2;
    eng2.cards = {{0, 640 * M, 0, 0, 64 * M}};
    LendCommands desk2(eng2);
    CHECK(desk2.handle("LEND 0 64", r));
    const uint64_t e = lent_id(r, 0, 64);
    CHECK(e != 0 && e != a && e != b && e != c && e != d);
    CHECK_REPLY(desk, ("RETURN " + std::to_string(e)).c_str(), ("ERR UNKNOWN " + std::to_string(e) + "\n").c_str());

    // status(): the same figures the LENDABLE lines show, with the block size
    const auto st = desk.status();
    CHECK(st.size() == 3 && st[1].dev == 2 && st[1].lendable_mib == 0 && st[1].lent_mib == 400 && st[1].block_mib == 100);

    // ---- the control-line routing: which lines are lend commands, and which of them only read
    for (const char* l : {"LENDABLE", "  LENDABLE \r", "LENDABLE 0", "LENDABLE\t"})
        CHECK(lend_line_kind(l) == LendLineKind::lendable);
    for (const char* l : {"LEND 0 64", "LEND", "LEND x 10", " RETURN 7", "RETURN", "RETURN 1 2", "LEND 0 64\r"})
        CHECK(lend_line_kind(l) == LendLineKind::change);
    for (const char* l : {"", "   ", "GEN 1 2 3", "QUIT", "STOP", "STOP 3", "PAUSE keep", "LOGP 2 1,2,3", "GENI 1 2",
                          "LENDX 0 10", "lend 0 10", "RETURNED 1", "LENDABLES", "LENT 1 0 64", "END", "X LEND 0 64"})
        CHECK(lend_line_kind(l) == LendLineKind::none);

    // ---- the queue between a reader thread and the command loop: lend lines only, in order; a LENDABLE is taken
    // ahead of the loop's quiet point only while no LEND or RETURN is in front of it
    LendQueue q;
    std::string l;
    CHECK(!q.waiting() && !q.take(l) && !q.take_lendable(l));
    CHECK(!q.route("GEN 4 rid=1 1,2,3") && !q.route("STOP") && !q.route("") && !q.waiting());
    CHECK(q.route("LENDABLE"));
    CHECK(q.waiting());
    CHECK(q.route("LEND 0 64") && q.route("LENDABLE") && q.route("RETURN 3"));
    CHECK(q.take_lendable(l) && l == "LENDABLE");
    CHECK(!q.take_lendable(l) && q.waiting());   // a LEND in front: the LENDABLE behind it waits too
    CHECK(q.take(l) && l == "LEND 0 64");
    CHECK(q.take_lendable(l) && l == "LENDABLE");
    CHECK(!q.take_lendable(l) && q.waiting());
    CHECK(q.take(l) && l == "RETURN 3");
    CHECK(!q.waiting() && !q.take(l) && !q.take_lendable(l));
    // a reader thread and the loop at once: every lend line arrives once and in order, the others are left out
    {
        LendQueue tq;
        constexpr int kN = 2000;
        std::thread reader([&tq] {
            for (int i = 0; i < kN; ++i) {
                tq.route("GEN 1 rid=" + std::to_string(i) + " 1");
                tq.route(i % 3 == 0 ? "LENDABLE" : "RETURN " + std::to_string(i));
            }
        });
        int next = 0, polls = 0;
        while (next < kN && polls < 100000000) {
            ++polls;
            std::string got;
            if (!tq.take(got)) continue;
            const std::string want = next % 3 == 0 ? "LENDABLE" : "RETURN " + std::to_string(next);
            if (got != want) {
                std::fprintf(stderr, "queue order: got '%s', want '%s'\n", got.c_str(), want.c_str());
                reader.join();
                return 1;
            }
            ++next;
        }
        reader.join();
        CHECK(next == kN && !tq.waiting());
    }

    std::printf("vram_lend_test: all checks passed\n");
    return 0;
}
