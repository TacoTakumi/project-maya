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
        404 {"error": ...}           no open lease has that id (never lent, returned already, or the engine ended)
        409 {"error": ..., "retry": true}   the memory is still in use: the lease stays open, try again later
    GET /vram/state
        200 {"cards": [{"card", "lendable", "lent"}, ...], "leases": [{"id", "card", "mib"}, ...]}

Every lend and return route answers 501 {"error": ...} for an engine without lending (it refuses the commands, or it
is not a Maya engine), and 503 {"error": ...} when the engine is not running or did not answer; /vram/state then lists
the engine's cards with lendable and lent 0 (no cards at all for an engine that reports none).

A card is named by its PCI address, matched whatever its case or spelling (gpu_map.pci_addr: CUDA's, NVML's 8-digit
domain, lspci's without one) and answered as "0000:0a:00.0" - lower case, a 4-digit domain and the function, as lspci
and sysfs spell it.  The engine's VRAM lines give each card's number.  A lease id is opaque (a string; the engines hand
out decimal numbers).

The engine's lease table is the one that counts: LENDABLE's lent= comes from it.  It does not list the leases one by
one, so the server keeps the ones it handed out, for the engine process that lent them: when the process ends (a
restart, an idle unload, a crash) its leases end with it, and so does this record.

The engine interface used: `lend_command(line) -> [answer lines]`, `vram` (the parsed VRAM lines), `alive()` and
`proc` (the running process: a new one is a new lease table)."""
from __future__ import annotations

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


class NoLending(Exception):
    """The engine refused a lend command as unknown: it cannot lend."""


class Lender:
    """The /vram routes over the service's engine (one for the server's life; the engine may restart under it)."""

    def __init__(self):
        self._lock = threading.Lock()     # one route at a time: a command and its record change go together
        self._owner = None                # the engine process the record belongs to
        self._leases: dict[str, dict] = {}   # id -> {"id", "card", "mib"}, in the order they were lent

    # -- what the routes share
    def _sync(self, engine) -> bool:
        """The record for the engine process running now (empty after a restart or while none runs); True when one
        runs."""
        alive = getattr(engine, "alive", None)
        running = alive() if alive is not None else True
        owner = getattr(engine, "proc", engine) if running else None
        if owner is not self._owner:
            self._owner, self._leases = owner, {}
        return running

    @staticmethod
    def _cards(engine) -> list[tuple[int, str]]:
        """(engine card number, PCI address) of each card the engine's VRAM lines name, in its order, each once."""
        out: list[tuple[int, str]] = []
        for r in getattr(engine, "vram", None) or []:
            addr = r.get("addr")
            if addr and isinstance(r.get("dev"), int) and all(d != r["dev"] for d, _ in out):
                out.append((r["dev"], addr))
        return out

    @staticmethod
    def _ask(engine, line: str) -> list[str]:
        got = engine.lend_command(line)
        if got and got[0].startswith("ERR unsupported command"):
            raise NoLending(got[0])
        return got

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
        with self._lock:
            if not self._sync(engine):
                return 503, {"error": "the model is not loaded"}
            try:
                got = self._ask(engine, f"LEND {dev} {mib}")
            except NoLending:
                return self._unsupported()
            except (RuntimeError, ValueError, OSError) as e:
                return self._gone(e)
            f = got[0].split()
            if f[0] == "LENT" and len(f) == 4 and f[3].isdigit():
                lease = {"id": f[1], "card": addr, "mib": int(f[3])}
                self._leases[f[1]] = lease
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
        with self._lock:
            if not self._sync(engine):                    # (the leases ended with the engine)
                return 404, {"error": f"no open lease {lid}: the model is not loaded"}
            try:
                got = self._ask(engine, f"RETURN {lid}")
            except NoLending:
                return self._unsupported()
            except (RuntimeError, ValueError, OSError) as e:
                return self._gone(e)
            f = got[0].split()
            if f[0] == "RETURNED":
                return 200, dict(self._leases.pop(lid, None) or {"id": lid})
            if f[:2] == ["ERR", "BUSY"]:
                return 409, {"error": f"lease {lid}'s memory is still in use; it stays lent - try again later",
                             "retry": True}
            if f[:2] == ["ERR", "UNKNOWN"]:
                self._leases.pop(lid, None)
                return 404, {"error": f"no open lease {lid}"}
            if f[:2] == ["ERR", "MALFORMED"]:
                return 400, {"error": f"the engine refused the return: {got[0]}"}
            return 502, {"error": f"the engine answered {got[0]!r}"}

    def state(self, engine) -> tuple[int, dict]:
        cards = self._cards(engine)
        idle = {"cards": [{"card": a, "lendable": 0, "lent": 0} for _, a in cards], "leases": []}
        if not hasattr(engine, "lend_command"):
            return 200, idle
        with self._lock:
            if not self._sync(engine):
                return 200, idle
            try:
                got = self._ask(engine, "LENDABLE")
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
            return 200, {"cards": out, "leases": [dict(x) for x in self._leases.values()]}
