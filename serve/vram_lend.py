"""The /vram routes: another program on the machine borrows part of a card's expert-cache memory from the engine and
gives it back.  The engine owns the leases (its LEND / RETURN / LENDABLE control lines, include/strata/core/
vram_lend.hpp); this module turns the HTTP calls into those lines and their answers into JSON:

    POST /vram/lend {"card": "0000:0a:00.0", "mib": 2048}
        200 {"id": "7", "card": "0000:0a:00.0", "mib": 2048}   mib: what the engine lent (rounded up to its blocks)
        400 {"error": ...}           mib missing, not a whole number (a bool is not one), 0 or negative; a card that is
                                     not a PCI address or not one the engine uses; a body that is not a JSON object
        409 {"max_mib": N}           more than the card can lend now (N: its lendable; 0 for a card that does not lend)
    POST /vram/return {"id": "7"}    (a number is taken too)
        200 {"id": "7", "card": "0000:0a:00.0", "mib": 2048}
        400 {"error": ...}           no id, or one that is not a word
        404 {"error": ...}           the server handed out no open lease with that id (never lent, returned already,
                                     or the engine ended): nothing goes to the engine
        409 {"error": ..., "retry": true}   the memory is still in use: the lease stays open, try again later
    GET /vram/state
        200 {"cards": [{"card", "lendable", "lent"}, ...], "leases": [{"id", "card", "mib"}, ...]}

Every lend and return route answers 501 {"error": ...} for an engine without lending (it refuses the commands, or it
is not a Maya engine), and 503 {"error": ...} when the engine is not running or did not answer; /vram/state then lists
the engine's cards with lendable and lent 0 (no cards at all for an engine that reports none).  The engine takes the
commands between its steps, and one step can read a long prompt: a LEND it answers after the route stopped waiting
(Lender.answer_s, 30 s) still lends, and the server gives that lease back by itself - /vram/state lists it until the
engine has it back.

A card is named by its PCI address, matched whatever its case or spelling (gpu_map.pci_addr: CUDA's, NVML's 8-digit
domain, lspci's without one) and answered as "0000:0a:00.0" - lower case, a 4-digit domain and the function, as lspci
and sysfs spell it.  The engine's VRAM lines give each card's number.  A lease id is opaque (a string; the engines hand
out decimal numbers, from 1 again in each process: the server adds what earlier processes handed out, so an id names one
lease for the server's life).

The engine's lease table is the one that counts: LENDABLE's lent= comes from it.  It does not list the leases one by
one, so the server keeps the ones it handed out, for the engine process that lent them: when the process ends (a
restart, an idle unload, a crash) its leases end with it, and so does this record.

The record changes as the engine's answers come, in the order the engine gave them (Calls below), so /vram/state's
leases always match LENDABLE's lent= figures.

The engine interface used: `lend_command(line, timeout, then) -> [answer lines]` and `lend_post(line, then)` (Calls
below), `vram` (the parsed VRAM lines), `alive()` and `proc` (the running process: a new one is a new lease table)."""
from __future__ import annotations

import collections
import re
import threading

from serve import gpu_map

# The engine's answers to the lend commands: the server takes them off the request's lines (they come back while
# requests run, untagged).  A request's own ERR never reads like these (the words after ERR are upper case here).
ANSWER = re.compile(r"(?:LENT|RETURNED|LENDABLE) |END$|ERR (?:MAX|NODEV|BUSY|UNKNOWN) \S+$|"
                    r"ERR MALFORMED (?:LEND|RETURN|LENDABLE)\b|ERR unsupported command '(?:LEND|RETURN|LENDABLE)'")


def is_answer(line: str) -> bool:
    return bool(ANSWER.match(line.rstrip("\r\n")))


def last_line(line: str) -> bool:
    """An answer ends at its first line, except LENDABLE's: a line a card, then END."""
    return not line.startswith("LENDABLE ")


class Call:
    """One lend command sent to the engine: `got` holds its answer lines once `done` is set (None: the engine ended
    before it answered)."""
    __slots__ = ("got", "done", "then", "waited", "answered")

    def __init__(self, then=None):
        self.got: list[str] | None = []
        self.done = threading.Event()
        self.then = then
        self.waited = True                # a caller waits for the answer (False once it stopped waiting)
        self.answered = False


class Calls:
    """The lend commands one engine process was sent and has not answered yet, in the order sent.  The engine answers
    them in that order (one queue, read by its main thread only) and its answers carry no tag, so an answer is the
    oldest waiting command's - also when its caller stopped waiting: never the command's sent after it.

    A call's `then(lines, waited)` runs on the line reader's thread as its answer comes, before the caller has it
    (waited: a caller still waits): what the answers change happens in the engine's order.  It must not wait."""

    def __init__(self, stdin):
        self._stdin = stdin               # the engine process's input (a restart makes a new table)
        self._lock = threading.Lock()     # the table and the writes: a command is listed before it can be answered
        self._sent: collections.deque[Call] = collections.deque()
        self._open = True

    def send(self, line: str, then=None, waited: bool = True) -> Call | None:
        """`line` to the engine -> its call (None: the engine has ended or does not read its input any more)."""
        call = Call(then)
        call.waited = waited
        with self._lock:
            if not self._open:
                return None
            self._sent.append(call)
            try:
                self._stdin.write(line + "\n")
                self._stdin.flush()
            except (OSError, ValueError):
                self._sent.pop()
                return None
        return call

    def wait(self, call: Call, timeout: float) -> bool:
        """True once the call is answered (or the engine ended: got None); False after `timeout` - its answer still
        goes to its `then` when it comes."""
        if call.done.wait(timeout):
            return True
        with self._lock:
            if not call.answered and not call.done.is_set():
                call.waited = False
                return False
        call.done.wait()                  # (answered meanwhile: its then is running)
        return True

    def answer(self, line: str):
        """An answer line from the engine (its line reader's thread): to the oldest call still waiting."""
        with self._lock:
            if not self._sent:            # (no command waits for one: nothing to give it to)
                return
            call = self._sent[0]
            call.got.append(line.strip())
            if not last_line(line):
                return
            self._sent.popleft()
            call.answered = True
        try:
            if call.then is not None:
                call.then(call.got, call.waited)
        except Exception as e:  # noqa: BLE001 - the line reader goes on whatever an answer's handler does
            print(f"[maya] the lend answer {call.got[0]!r} was not handled: {e}", flush=True)
        finally:
            call.done.set()

    def close(self):
        """The engine's output ended: the calls still waiting have no answer coming (got None)."""
        with self._lock:
            self._open = False
            left = list(self._sent)
            self._sent.clear()
            for call in left:
                call.got = None
                call.done.set()


class NoLending(Exception):
    """The engine refused a lend command as unknown: it cannot lend."""


class Lender:
    """The /vram routes over the service's engine (one for the server's life; the engine may restart under it)."""

    def __init__(self):
        self._lock = threading.Lock()     # the record: held to read or change it, never while the engine answers
        self._owner = None                # the engine process the record belongs to
        self._leases: dict[str, dict] = {}   # id -> {"id", "card", "mib"}, in the order they were lent
        self._engine_id: dict[str, str] = {}  # id -> the engine's id for that lease
        self._owed: dict[str, bool] = {}  # the leases the server gives back itself -> their RETURN is out now
        self._base = 0                    # added to the engine's ids: past every id an earlier engine process had
        self._top = 0                     # the highest id handed out
        self._can: bool | None = None     # the running engine takes the lend commands (None: not known yet)
        self.answer_s = 30.0              # how long a route waits for the engine's answer

    # -- what the routes share
    def _sync(self, engine) -> bool:
        """The record for the engine process running now (empty after a restart or while none runs); True when one
        runs.  Under the lock."""
        alive = getattr(engine, "alive", None)
        running = alive() if alive is not None else True
        owner = getattr(engine, "proc", engine) if running else None
        if owner is not self._owner:
            self._owner, self._leases, self._engine_id, self._owed, self._can = owner, {}, {}, {}, None
            self._base = self._top
        return running

    def _begin(self, engine):
        """A route's start: (True when an engine runs, its process); a lease the server gives back that the engine
        could not take back yet (still in use) is sent again."""
        with self._lock:
            running = self._sync(engine)
            owner = self._owner
            again = [lid for lid, out in self._owed.items() if not out]
            for lid in again:
                self._owed[lid] = True
            again = [(lid, self._engine_id[lid]) for lid in again]
        for lid, eid in again:
            self._post_return(engine, owner, lid, eid)
        return running, owner

    def _record(self, owner, eid: str, addr: str, mib: int) -> dict | None:
        """A lease the engine process `owner` lent -> its record (None: that process has ended, and its leases with
        it).  The engines' ids start again with each process: the server's are theirs plus what earlier processes
        handed out, so an id from before a restart never names a lease of the new process.  Under the lock."""
        if owner is not self._owner:
            return None
        lid = eid
        if eid.isdigit():
            n = self._base + int(eid)
            self._top = max(self._top, n)
            lid = str(n)
        lease = {"id": lid, "card": addr, "mib": mib}
        self._leases[lid], self._engine_id[lid] = lease, eid
        return lease

    def _lent(self, engine, owner, addr: str, got: list[str], waited: bool, box: dict):
        """The engine's answer to a LEND (as it comes): its lease in the record.  One whose caller stopped waiting
        goes back to the engine - nobody will use it; it stays in the record, so /vram/state lists it, until the
        engine has it back."""
        f = got[0].split()
        if f[0] != "LENT" or len(f) != 4 or not f[3].isdigit():
            return
        with self._lock:
            box["lease"] = lease = self._record(owner, f[1], addr, int(f[3]))
            if lease is None or waited:
                return
            self._owed[lease["id"]] = True
        self._post_return(engine, owner, lease["id"], f[1])

    def _post_return(self, engine, owner, lid: str, eid: str):
        try:
            engine.lend_post(f"RETURN {eid}", lambda got, waited: self._returned(owner, lid, got))
        except (RuntimeError, ValueError, OSError):
            pass                          # (the engine ended: the lease with it)

    def _returned(self, owner, lid: str, got: list[str]):
        """The engine's answer to a RETURN (as it comes, also when its caller stopped waiting) in the record."""
        f = got[0].split()
        with self._lock:
            if owner is not self._owner:
                return
            if f[0] == "RETURNED" or f[:2] == ["ERR", "UNKNOWN"]:
                self._leases.pop(lid, None)
                self._engine_id.pop(lid, None)
                self._owed.pop(lid, None)
            elif lid in self._owed:       # still in use: sent again at the next route
                self._owed[lid] = False

    def _listed(self, owner, box: dict):
        """The leases as the engine's LENDABLE answer comes: the record matches its lent= figures."""
        with self._lock:
            box["leases"] = [dict(x) for x in self._leases.values()] if self._owner is owner else []

    @staticmethod
    def _cards(engine) -> list[tuple[int, str]]:
        """(engine card number, PCI address) of each card the engine's VRAM lines name, in its order, each once."""
        out: list[tuple[int, str]] = []
        for r in getattr(engine, "vram", None) or []:
            addr = r.get("addr")
            if addr and isinstance(r.get("dev"), int) and all(d != r["dev"] for d, _ in out):
                out.append((r["dev"], addr))
        return out

    def _ask(self, engine, line: str, then=None) -> list[str]:
        """The engine's answer to `line` (`then`: what it changes, run as it comes - Calls).  ValueError when it does
        not answer in answer_s."""
        got = engine.lend_command(line, timeout=self.answer_s, then=then)
        self._can = not got[0].startswith("ERR unsupported command")
        if not self._can:
            raise NoLending(got[0])
        return got

    def _lends(self, engine) -> bool:
        """The running engine takes the lend commands (a LENDABLE asks it, once a process)."""
        if self._can is None:
            try:
                self._ask(engine, "LENDABLE")
            except NoLending:
                return False
            except (RuntimeError, ValueError, OSError):
                return True
        return self._can is not False

    @staticmethod
    def _unsupported():
        return 501, {"error": "this engine cannot lend its GPU memory"}

    @staticmethod
    def _gone(e):
        return 503, {"error": f"the engine did not answer ({e})"}

    # -- the routes
    def lend(self, engine, req) -> tuple[int, dict]:
        if not isinstance(req, dict):
            return 400, {"error": "the request body must be a JSON object {\"card\": ..., \"mib\": ...}"}
        mib = req.get("mib")
        if not isinstance(mib, int) or isinstance(mib, bool) or mib <= 0:
            return 400, {"error": "mib must be a whole number of MiB above 0"}
        addr = gpu_map.pci_addr(req.get("card"))
        if addr is None:
            return 400, {"error": "card must be a GPU's PCI address, like 0000:0a:00.0"}
        if not hasattr(engine, "lend_command"):
            return self._unsupported()
        dev = next((d for d, a in self._cards(engine) if a == addr), None)
        if dev is None:
            return 400, {"error": f"card {addr} is not one the model uses"}
        running, owner = self._begin(engine)
        if not running:
            return 503, {"error": "the model is not loaded"}
        box: dict = {}
        try:
            got = self._ask(engine, f"LEND {dev} {mib}",
                            lambda got, waited: self._lent(engine, owner, addr, got, waited, box))
        except NoLending:
            return self._unsupported()
        except ValueError as e:           # (the engine is busy with a long step: it lends once it is done)
            return 503, {"error": f"{e}: it may still lend that memory - the server will give it back when it does"}
        except (RuntimeError, OSError) as e:
            return self._gone(e)
        f = got[0].split()
        if f[0] == "LENT" and len(f) == 4 and f[3].isdigit():
            lease = box.get("lease")
            if lease is None:
                return 503, {"error": "the model restarted: what it lent ended with it"}
            return 200, dict(lease)
        if f[:2] == ["ERR", "MAX"] and f[2].isdigit():
            return 409, {"max_mib": int(f[2])}
        if f[:2] == ["ERR", "NODEV"]:                 # a card of the model's that does not lend: it can give 0
            return 409, {"max_mib": 0}
        if f[:2] == ["ERR", "MALFORMED"]:             # (a mib too large for the engine)
            return 400, {"error": f"the engine refused the lend: {got[0]}"}
        return 502, {"error": f"the engine answered {got[0]!r}"}

    def give_back(self, engine, req) -> tuple[int, dict]:
        if not isinstance(req, dict):
            return 400, {"error": "the request body must be a JSON object {\"id\": ...}"}
        lid = req.get("id")
        if isinstance(lid, int) and not isinstance(lid, bool):
            lid = str(lid)
        if not isinstance(lid, str) or not lid or not lid.isascii() or not lid.isprintable() or " " in lid:
            return 400, {"error": "id must be a lease id, as POST /vram/lend answered it"}
        if not hasattr(engine, "lend_command"):
            return self._unsupported()
        running, owner = self._begin(engine)
        if not running:                               # (the leases ended with the engine)
            return 404, {"error": f"no open lease {lid}: the model is not loaded"}
        with self._lock:
            lease = dict(self._leases[lid]) if lid in self._leases and self._owner is owner else None
            eid = self._engine_id.get(lid)
        if lease is None:                             # never lent by this engine process: nothing is sent
            if not self._lends(engine):
                return self._unsupported()
            return 404, {"error": f"no open lease {lid}"}
        try:
            got = self._ask(engine, f"RETURN {eid}", lambda got, waited: self._returned(owner, lid, got))
        except NoLending:
            return self._unsupported()
        except (RuntimeError, ValueError, OSError) as e:
            return self._gone(e)
        f = got[0].split()
        if f[0] == "RETURNED":
            return 200, lease
        if f[:2] == ["ERR", "UNKNOWN"]:
            return 404, {"error": f"no open lease {lid}"}
        if f[:2] == ["ERR", "BUSY"]:
            return 409, {"error": f"lease {lid}'s memory is still in use; it stays lent - try again later",
                         "retry": True}
        if f[:2] == ["ERR", "MALFORMED"]:
            return 400, {"error": f"the engine refused the return: {got[0]}"}
        return 502, {"error": f"the engine answered {got[0]!r}"}

    def state(self, engine) -> tuple[int, dict]:
        cards = self._cards(engine)
        idle = {"cards": [{"card": a, "lendable": 0, "lent": 0} for _, a in cards], "leases": []}
        if not hasattr(engine, "lend_command"):
            return 200, idle
        running, owner = self._begin(engine)
        if not running:
            return 200, idle
        box: dict = {}
        try:
            got = self._ask(engine, "LENDABLE", lambda got, waited: self._listed(owner, box))
        except (NoLending, RuntimeError, ValueError, OSError):
            return 200, idle
        figures = {}
        for line in got:
            f = dict(kv.partition("=")[::2] for kv in line.split()[1:]) if line.startswith("LENDABLE ") else {}
            if all(f.get(k, "").isdigit() for k in ("dev", "lendable", "lent")):
                figures[int(f["dev"])] = (int(f["lendable"]), int(f["lent"]))
        name = dict(cards)
        out = [{"card": name[d], "lendable": figures[d][0], "lent": figures[d][1]} for d in figures if d in name]
        # the model's other cards (no lending there): listed, with nothing to give
        out += [{"card": a, "lendable": 0, "lent": 0} for d, a in cards if d not in figures]
        return 200, {"cards": out, "leases": box.get("leases", [])}
