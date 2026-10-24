// src/ds4/ds4_lines.hpp - maya-ds4's control lines: stdin read on a thread of its own, each line routed as it comes.
#pragma once

#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "strata/platform/idle_gate.hpp"   // (idle_poke: the service threads asleep after a long quiet wake for a line, #685)
#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#include <cerrno>
#endif

namespace strata::ds4 {

// a lend command (LEND, RETURN, LENDABLE: the first word, split on blanks as core::LendCommands splits it)
inline bool lend_line(const std::string& l) {
    size_t a = 0;
    while (a < l.size() && std::isspace((unsigned char) l[a])) ++a;
    size_t b = a;
    while (b < l.size() && !std::isspace((unsigned char) l[b])) ++b;
    const std::string w = l.substr(a, b - a);
    return w == "LEND" || w == "RETURN" || w == "LENDABLE";
}

// stdin lines on a thread, so a request can see STOP while it decodes.  read(2), not std::cin - glibc's exit()
// flushes stdio and waits on stdin's lock, which getline holds while it waits for input (the GLM serve's pattern).
// The lend commands wait in a queue of their own: the reader never runs them (they touch the model's expert slots), the
// main thread does, between the model's calls - while idle (next() hands them out first) and while a request decodes
// (take_lend between steps), so the request queue's next GEN stays where it is
struct Lines {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> q;
    std::deque<std::string> lend;
    bool eof = false;
    std::atomic<bool> stop{false};
    // --parallel: `STOP <rid>` ends that request alone (stop_rids), a bare STOP, QUIT or the input's end all of them
    std::vector<long long> stop_rids;
    std::atomic<bool> stop_all{false};
    bool stopped(long long rid) {
        if (stop_all.load()) return true;
        std::lock_guard<std::mutex> lk(mu);
        return std::find(stop_rids.begin(), stop_rids.end(), rid) != stop_rids.end();
    }
    void forget(long long rid) {   // (the request is over: a late STOP of it changes nothing)
        std::lock_guard<std::mutex> lk(mu);
        stop_rids.erase(std::remove(stop_rids.begin(), stop_rids.end(), rid), stop_rids.end());
    }
    // one line from the reader (its CR gone): STOP only sets stop; QUIT sets it too, and is queued; a lend command to
    // the lend queue
    void route(const std::string& l) {
        std::lock_guard<std::mutex> lk(mu);
        if (l == "STOP" || l.rfind("STOP ", 0) == 0) {
            stop.store(true);
            if (l == "STOP") stop_all.store(true);
            else stop_rids.push_back(std::atoll(l.c_str() + 5));
            cv.notify_all();
            return;
        }
        if (lend_line(l)) {
            lend.push_back(l);
            cv.notify_all();
            return;
        }
        if (l == "QUIT") stop.store(true), stop_all.store(true);
        q.push_back(l);
        cv.notify_all();
    }
    // stdin is done: the queues stay as they are, and next() returns false once they are empty
    void finish() {
        std::lock_guard<std::mutex> lk(mu);
        eof = true;
        stop.store(true);
        stop_all.store(true);
        cv.notify_all();
    }
    void start() {
        std::thread([this] {
            std::string l, buf;
            char chunk[4096];
            auto getline_fd = [&](std::string& out) -> bool {
                for (;;) {
                    const size_t nlpos = buf.find('\n');
                    if (nlpos != std::string::npos) {
                        out.assign(buf, 0, nlpos);
                        buf.erase(0, nlpos + 1);
                        return true;
                    }
#if defined(_WIN32)
                    const int n = _read(0, chunk, (unsigned) sizeof chunk);
#else
                    const ssize_t n = ::read(0, chunk, sizeof chunk);
                    if (n < 0 && errno == EINTR) continue;
#endif
                    if (n <= 0) {
                        if (buf.empty()) return false;
                        out.swap(buf);
                        buf.clear();
                        return true;
                    }
                    buf.append(chunk, (size_t) n);
                }
            };
            while (getline_fd(l)) {
                strata::platform::idle_poke();   // (the service threads asleep after a long quiet wake for this line)
                if (!l.empty() && l.back() == '\r') l.pop_back();
                route(l);
            }
            finish();
        }).detach();
    }
    // the next line, waiting for one: a lend command first, else the next request (false: stdin is done, both empty)
    bool next(std::string& l) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !lend.empty() || !q.empty() || eof; });
        std::deque<std::string>& from = !lend.empty() ? lend : q;
        if (from.empty()) return false;
        l = std::move(from.front());
        from.pop_front();
        return true;
    }
    // a lend command if one waits, without waiting (the decode loop's look between steps)
    bool take_lend(std::string& l) {
        std::lock_guard<std::mutex> lk(mu);
        if (lend.empty()) return false;
        l = std::move(lend.front());
        lend.pop_front();
        return true;
    }
};

}  // namespace strata::ds4
