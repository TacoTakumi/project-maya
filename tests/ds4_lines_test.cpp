// CPU-only checks of maya-ds4's control-line routing (src/ds4/ds4_lines.hpp): the lend commands go to a queue of their
// own for the main thread, which runs them between the model's calls - the reader never runs them - and STOP, QUIT and
// the requests are routed as before.  No model, no GPU.
#include "ds4_lines.hpp"
#include "strata/core/vram_lend.hpp"

#include <chrono>
#include <cstdio>
#include <future>
#include <string>
#include <vector>
#if !defined(_WIN32)
#include <unistd.h>
#endif

using strata::ds4::Lines;
using strata::ds4::lend_line;
#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

namespace {

// a lender with one card that lends nothing: enough for the handler to answer every lend command
struct NoCards : strata::core::VramLender {
    std::vector<strata::core::LendCard> lend_cards() override { return {{0, 0, 2}}; }
    strata::core::LendResult lend(int, std::size_t) override { return {}; }
    strata::core::GiveBack give_back(int, std::uint64_t) override { return strata::core::GiveBack::returned; }
};

const std::vector<std::string> kLend = {"LEND 0 4096", "LEND 1 2", "RETURN 7", "LENDABLE", "LEND", "RETURN",
                                        "LENDABLE now", "LEND x y z", "  LEND 0 1", "LEND\t0\t1", "RETURN 1 2"};
const std::vector<std::string> kOther = {"", "GEN 4 1,2,3", "GENI 4 pic 1,2", "STOP", "STOP 3", "QUIT", "LENDX 0 1",
                                         "RETURNED 1", "LENT 1 0 2", "lend 0 1", "Lendable", "END", "ERR MAX 0"};

}  // namespace

int main() {
    // the predicate is the handler's own split: every line it routes is a lend command, every other line is not one
    {
        NoCards eng;
        strata::core::LendCommands cmds(eng);
        std::string reply;
        for (const std::string& l : kLend) {
            CHECK(lend_line(l));
            CHECK(cmds.handle(l, reply) && !reply.empty());
        }
        for (const std::string& l : kOther) {
            CHECK(!lend_line(l));
            CHECK(!cmds.handle(l, reply) && reply.empty());
        }
    }
    // the reader's routing: lend lines to their queue, not the request queue, and no stop
    {
        Lines in;
        for (const std::string& l : kLend) in.route(l);
        CHECK(in.q.empty());
        CHECK(!in.stop.load());
        std::string l;
        for (const std::string& want : kLend) {
            CHECK(in.take_lend(l));
            CHECK(l == want);
        }
        CHECK(!in.take_lend(l));
    }
    // STOP, QUIT and the requests as before: STOP only sets stop, QUIT sets it and is queued, a request is queued
    {
        Lines in;
        in.route("GEN 4 1,2");
        CHECK(!in.stop.load());
        in.route("STOP");
        CHECK(in.stop.load());
        in.stop.store(false);
        in.route("STOP 3");
        CHECK(in.stop.load());
        in.stop.store(false);
        in.route("QUIT");
        CHECK(in.stop.load());
        CHECK(in.q.size() == 2 && in.q[0] == "GEN 4 1,2" && in.q[1] == "QUIT");
        std::string l;
        CHECK(!in.take_lend(l));   // (the decode loop's look takes no request)
        CHECK(in.q.size() == 2);
    }
    // a lend while a request decodes: taken between steps, the queued request left for the serve loop; a STOP still
    // seen
    {
        Lines in;
        in.route("GEN 4 1,2");
        in.route("LEND 0 4096");
        in.route("STOP");
        std::string l;
        CHECK(in.take_lend(l) && l == "LEND 0 4096");
        CHECK(in.stop.load());
        CHECK(in.next(l) && l == "GEN 4 1,2");
    }
    // idle: the serve loop's wait wakes for a lend line, and hands it out before a request
    {
        Lines in;
        auto got = std::async(std::launch::async, [&in] {
            std::string l;
            return in.next(l) ? l : std::string("(none)");
        });
        CHECK(got.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout);
        in.route("LENDABLE");
        CHECK(got.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
        CHECK(got.get() == "LENDABLE");
        in.route("GEN 4 1,2");
        in.route("RETURN 7");
        std::string l;
        CHECK(in.next(l) && l == "RETURN 7");
        CHECK(in.next(l) && l == "GEN 4 1,2");
    }
    // stdin's end: the lend lines still queued come out first, then next() is done
    {
        Lines in;
        in.route("LEND 0 2");
        in.finish();
        std::string l;
        CHECK(in.next(l) && l == "LEND 0 2");
        CHECK(!in.next(l));
    }
#if !defined(_WIN32)
    // the reader thread over a real stdin (a pipe): CRs dropped, each line routed as above
    {
        int fd[2];
        CHECK(::pipe(fd) == 0);
        CHECK(::dup2(fd[0], 0) == 0);
        ::close(fd[0]);
        const std::string text = "LEND 0 4096\r\nGEN 4 1,2\nSTOP\nRETURN 1\nLENDABLE";   // (the last line without '\n')
        CHECK(::write(fd[1], text.data(), text.size()) == (ssize_t) text.size());
        ::close(fd[1]);
        Lines in;
        in.start();
        std::string l;
        CHECK(in.next(l) && l == "LEND 0 4096");
        std::vector<std::string> rest;
        while (in.next(l)) rest.push_back(l);
        CHECK(rest.size() == 3);   // (STOP is not a line for the loop)
        size_t gens = 0, lends = 0;
        for (const std::string& r : rest) {
            gens += r == "GEN 4 1,2";
            lends += r == "RETURN 1" || r == "LENDABLE";
        }
        CHECK(gens == 1 && lends == 2);
    }
#endif
    std::printf("ds4_lines_test: ok\n");
    return 0;
}
