// The shared lend commands and the lendable rule (include/strata/core/vram_lend.hpp).
#include "strata/core/vram_lend.hpp"

#include <atomic>
#include <cctype>
#include <charconv>
#include <climits>
#include <sstream>

namespace strata::core {

namespace {

// Lease ids for the whole process: a counter, so an id is never handed out twice, even by another command table or
// after its lease came back.  0 is never an id.
std::atomic<std::uint64_t> g_next_lease_id{1};

// A whole decimal number, digits only (no sign, no blanks, nothing after), that fits `out`.
bool parse_u64(const std::string& s, std::uint64_t& out) {
    if (s.empty()) return false;
    for (char ch : s)
        if (ch < '0' || ch > '9') return false;
    const auto r = std::from_chars(s.data(), s.data() + s.size(), out);
    return r.ec == std::errc() && r.ptr == s.data() + s.size();
}

}  // namespace

std::size_t lendable_bytes(std::size_t pool_bytes, std::size_t prompt_borrow_bytes, std::size_t min_slots_bytes,
                           std::size_t block_bytes) {
    if (block_bytes == 0) return 0;
    // subtracted one at a time so neither step can wrap: short of either is nothing to lend
    if (pool_bytes <= prompt_borrow_bytes) return 0;
    std::size_t left = pool_bytes - prompt_borrow_bytes;
    if (left <= min_slots_bytes) return 0;
    left -= min_slots_bytes;
    return left / block_bytes * block_bytes;
}

LendLineKind lend_line_kind(const std::string& line) {
    // the first word, as handle() splits the line (blanks, a trailing CR among them)
    std::size_t a = 0;
    while (a < line.size() && std::isspace((unsigned char) line[a])) ++a;
    std::size_t b = a;
    while (b < line.size() && !std::isspace((unsigned char) line[b])) ++b;
    const std::string w = line.substr(a, b - a);
    if (w == "LENDABLE") return LendLineKind::lendable;
    if (w == "LEND" || w == "RETURN") return LendLineKind::change;
    return LendLineKind::none;
}

bool LendQueue::route(const std::string& line) {
    if (lend_line_kind(line) == LendLineKind::none) return false;
    std::lock_guard<std::mutex> lk(mu_);
    q_.push_back(line);
    n_.store(q_.size(), std::memory_order_release);
    return true;
}

bool LendQueue::take(std::string& line) {
    if (!waiting()) return false;
    std::lock_guard<std::mutex> lk(mu_);
    if (q_.empty()) return false;
    line = std::move(q_.front());
    q_.pop_front();
    n_.store(q_.size(), std::memory_order_release);
    return true;
}

bool LendQueue::take_lendable(std::string& line) {
    if (!waiting()) return false;
    std::lock_guard<std::mutex> lk(mu_);
    if (q_.empty() || lend_line_kind(q_.front()) != LendLineKind::lendable) return false;
    line = std::move(q_.front());
    q_.pop_front();
    n_.store(q_.size(), std::memory_order_release);
    return true;
}

std::size_t LendCommands::lent_mib(int dev) const {
    std::size_t sum = 0;
    for (const auto& [id, l] : leases_)
        if (l.dev == dev) sum += l.mib;
    return sum;
}

std::vector<LendCardStatus> LendCommands::status() {
    std::vector<LendCardStatus> out;
    for (const LendCard& c : lender_.lend_cards()) out.push_back({c.dev, c.lendable_mib, lent_mib(c.dev), c.block_mib});
    return out;
}

bool LendCommands::handle(const std::string& line, std::string& reply) {
    reply.clear();
    std::istringstream ss(line);   // words split on blanks, so a trailing CR from the client is dropped too
    std::vector<std::string> w;
    for (std::string s; ss >> s;) w.push_back(s);
    if (w.empty()) return false;
    const std::string& cmd = w[0];

    if (cmd == "LENDABLE") {
        if (w.size() != 1) {
            reply = "ERR MALFORMED LENDABLE\n";
            return true;
        }
        for (const LendCardStatus& c : status())
            reply += "LENDABLE dev=" + std::to_string(c.dev) + " lendable=" + std::to_string(c.lendable_mib) +
                     " lent=" + std::to_string(c.lent_mib) + "\n";
        reply += "END\n";
        return true;
    }

    if (cmd == "LEND") {
        std::uint64_t dev64 = 0, mib64 = 0;
        if (w.size() != 3 || !parse_u64(w[1], dev64) || dev64 > (std::uint64_t) INT_MAX || !parse_u64(w[2], mib64) ||
            mib64 == 0 || mib64 > (std::uint64_t) SIZE_MAX) {
            reply = "ERR MALFORMED LEND <dev> <mib>\n";
            return true;
        }
        const int dev = (int) dev64;
        const std::size_t mib = (std::size_t) mib64;
        const std::vector<LendCard> cards = lender_.lend_cards();
        const LendCard* card = nullptr;
        for (const LendCard& c : cards)
            if (c.dev == dev) card = &c;
        if (!card) {
            reply = "ERR NODEV " + std::to_string(dev) + "\n";
            return true;
        }
        // whole blocks only: what the card has, down to blocks (the engine's figure already is), and the request up
        const std::size_t block = card->block_mib;
        const std::size_t max = block ? card->lendable_mib / block * block : 0;
        if (mib > max) {
            reply = "ERR MAX " + std::to_string(max) + "\n";
            return true;
        }
        const std::size_t want = (mib + block - 1) / block * block;   // mib <= max, so this cannot pass max or wrap
        const LendResult r = lender_.lend(dev, want);
        if (!r.ok) {
            reply = "ERR MAX " + std::to_string(r.max_mib) + "\n";
            return true;
        }
        const std::uint64_t id = g_next_lease_id.fetch_add(1);
        leases_[id] = {dev, r.mib, r.token};
        reply = "LENT " + std::to_string(id) + " " + std::to_string(dev) + " " + std::to_string(r.mib) + "\n";
        return true;
    }

    if (cmd == "RETURN") {
        if (w.size() != 2) {
            reply = "ERR MALFORMED RETURN <id>\n";
            return true;
        }
        // an id is opaque to the caller: any word that does not name an open lease is unknown
        std::uint64_t id = 0;
        const auto it = parse_u64(w[1], id) ? leases_.find(id) : leases_.end();
        if (it == leases_.end()) {
            reply = "ERR UNKNOWN " + w[1] + "\n";
            return true;
        }
        if (lender_.give_back(it->second.dev, it->second.token) == GiveBack::busy) {
            reply = "ERR BUSY " + std::to_string(id) + "\n";
            return true;
        }
        leases_.erase(it);
        reply = "RETURNED " + std::to_string(id) + "\n";
        return true;
    }

    return false;
}

}  // namespace strata::core
