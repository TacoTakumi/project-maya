#pragma once
// Lending a card's expert-cache memory to another program on the same machine, and taking it back.  The parts here
// are shared by every engine: the control-line commands and their answers, the table of open leases and the rule for
// how much a card can lend.  An engine supplies only the VramLender interface over its own cache.
//
// The commands (one control line each; the words are case-sensitive, extra blanks and a trailing CR are ignored):
//
//   LEND <dev> <mib>    lend <mib> MiB of card <dev>, rounded UP to whole blocks of that card
//       -> LENT <id> <dev> <mib>    the lease, with the MiB actually lent
//       -> ERR MAX <mib>            more than the card can lend; <mib> is what it can lend now
//       -> ERR NODEV <dev>          <dev> is not a card that lends (well formed, no such card)
//   RETURN <id>         give a lease's memory back to the cache
//       -> RETURNED <id>
//       -> ERR BUSY <id>            someone still holds the memory: the lease stays open, try again later
//       -> ERR UNKNOWN <id>         no open lease has that id (never handed out, or already returned)
//   LENDABLE            what each card can lend now and what it has out
//       -> LENDABLE dev=<n> lendable=<mib> lent=<mib>    one line per card, in the engine's order
//       -> END
//
// A lend command that does not parse is answered, never passed on to the engine's other commands:
//       -> ERR MALFORMED LEND <dev> <mib>   <dev> not a non-negative int, or <mib> missing, zero, signed, not a
//                                           whole number or too large, or a word too many
//       -> ERR MALFORMED RETURN <id>        no id, or a word too many
//       -> ERR MALFORMED LENDABLE           a word after LENDABLE
//
// Lease ids are opaque to the caller (decimal numbers today, never 0) and unique for the life of the process, also
// across several LendCommands.  Maya does not record who borrows: a lease is an id, a card and a size.
//
// The commands come while requests run.  An engine's stdin reader thread passes each line through a LendQueue, and the
// engine's command loop takes them on its own thread: LENDABLE at any line, LEND and RETURN where nothing is in flight.
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace strata::core {

/// The lendable rule: what a card can lend is its cached pool minus what the prompt path borrows from it minus the
/// minimum slots the cache keeps, clamped at 0 and rounded DOWN to whole blocks, so a lend never pushes the prompt
/// path back to reading token by token.  All in bytes; the result is a multiple of `block_bytes` (0 when
/// `block_bytes` is 0: a card with no lend blocks lends nothing).
///
/// `pool_bytes` is the pool as it is NOW, after the blocks already lent have left it: the rule does not subtract the
/// open leases itself, so each lease shrinks what is lendable by its own size and a returned one grows it again.
std::size_t lendable_bytes(std::size_t pool_bytes, std::size_t prompt_borrow_bytes, std::size_t min_slots_bytes,
                           std::size_t block_bytes);

/// One card that can lend, as the engine reports it.  `lendable_mib` is lendable_bytes() in whole MiB (a multiple of
/// `block_mib` when blocks are whole MiB, which every VMM block is); `block_mib` is the lend unit (0: no lending).
struct LendCard {
    int dev = 0;
    std::size_t lendable_mib = 0;
    std::size_t block_mib = 0;
};

/// A lend's outcome.  ok: `mib` lent (at least what was asked) and the engine's own `token` for the blocks taken,
/// which comes back unchanged in give_back().  Refused: `max_mib`, what the card can lend now.
struct LendResult {
    bool ok = false;
    std::size_t mib = 0;
    std::uint64_t token = 0;
    std::size_t max_mib = 0;
};

enum class GiveBack {
    returned,   // the blocks are back in the cache
    busy,       // the memory is still held by someone else: nothing changed, the caller tries again later
};

/// What an engine implements over its cache.  The calls come from the engine's own command loop, one at a time.
class VramLender {
public:
    virtual ~VramLender() = default;
    /// Every card that can lend, in a fixed order (a card with nothing lendable now is still listed).
    virtual std::vector<LendCard> lend_cards() = 0;
    /// Take `mib` MiB on card `dev` out of the cache.  The handler has already rounded `mib` up to whole blocks and
    /// checked it against lend_cards(); the engine may still refuse (its pool changed) with the max it has now.
    virtual LendResult lend(int dev, std::size_t mib) = 0;
    /// Put back the blocks of a lease lend() returned `token` for.  Busy leaves the lease open.
    virtual GiveBack give_back(int dev, std::uint64_t token) = 0;
};

/// A card's state as the LENDABLE lines show it: the engine's figures plus what the open leases hold.
struct LendCardStatus {
    int dev = 0;
    std::size_t lendable_mib = 0;
    std::size_t lent_mib = 0;
    std::size_t block_mib = 0;
};

/// What a control line is to the lend commands, by its first word (split on blanks as LendCommands::handle splits it):
/// not a lend command, LENDABLE (it only reads the engine's figures: answered at any line, also between two tokens of
/// a request), or LEND / RETURN (they change the cache: answered only where nothing is in flight on any card).  A
/// malformed lend command is still one of them: LendCommands answers it.
enum class LendLineKind { none, lendable, change };
LendLineKind lend_line_kind(const std::string& line);

/// The lend command lines waiting for an engine's command loop.  The stdin reader thread routes every line through
/// route(), which keeps the lend commands and leaves the rest to the engine's own queue; the command loop takes them in
/// the order they came.  waiting() is a lock-free look for the loop's hot paths (the decode polls it every token).
class LendQueue {
public:
    /// The reader: a lend command is queued (true); any other line is not (false), for the engine's own commands.
    bool route(const std::string& line);
    /// A lend command waits.
    bool waiting() const { return n_.load(std::memory_order_acquire) > 0; }
    /// The command loop: the next lend command, of either kind (false: none waits).
    bool take(std::string& line);
    /// The next one only while it is a LENDABLE: a LEND or RETURN in front keeps it, and those after it, in the queue
    /// (the answers go out in the order the commands came).
    bool take_lendable(std::string& line);

private:
    std::mutex mu_;
    std::deque<std::string> q_;
    std::atomic<std::size_t> n_{0};
};

/// The lend commands over one engine: holds the open leases and turns a control line into its answer.  Keep one for
/// the life of the engine (the leases live in it).  Not thread-safe: call it from the engine's command loop.
class LendCommands {
public:
    explicit LendCommands(VramLender& lender) : lender_(lender) {}

    /// A lend command (LEND, RETURN, LENDABLE): true, and `reply` holds its answer, one or more lines each ending in
    /// '\n', for the loop to print as it is (std::fputs, then std::fflush).  Anything else: false and `reply` empty,
    /// for the loop's own commands.
    bool handle(const std::string& line, std::string& reply);

    /// Every card's figures, as LENDABLE reports them.
    std::vector<LendCardStatus> status();
    std::size_t open_leases() const { return leases_.size(); }

private:
    struct Lease {
        int dev = 0;
        std::size_t mib = 0;
        std::uint64_t token = 0;
    };
    std::size_t lent_mib(int dev) const;

    VramLender& lender_;
    std::map<std::uint64_t, Lease> leases_;   // by lease id
};

}  // namespace strata::core
