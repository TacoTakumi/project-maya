"""serve/test_vram_lend.py - the /vram routes (serve/vram_lend.py): lend a GPU's expert-cache memory, give it back and
what is lent, over a stand-in engine that speaks the lend commands (LEND / RETURN / LENDABLE) the way the engines do -
also while a request runs.

    python -m unittest serve.test_vram_lend -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# The stand-in engine: three VRAM lines (card 2 does not lend), READY, then GEN (ids + 1000, one token every 30 ms, on a
# thread: the command loop reads on meanwhile), STOP, QUIT and the lend commands with the engines' answers.  Cards lend
# in 64 MiB blocks (a lend is rounded up); card 1 can also be lent whole (LEND 1 ALL: its footprint of 15360 MiB, the
# whole= figure of its LENDABLE line) - never while a lease of either kind is open on it, and while its whole-card lease
# is open its lendable and whole= are 0.  FAKE_LEND=0: an engine without lending (the "unsupported command" ERR);
# FAKE_MULTI=<n>: several conversations (READY ... multi=<n>, the lines tagged with rid); FAKE_BUSY=<n>: the first n
# RETURNs answer ERR BUSY; FAKE_SLOW=<n>: the first n LENDs are answered late, by FAKE_SLOW_S seconds (1.2; the command
# loop waits, as an engine's does while it reads a long prompt: the commands behind them wait too); FAKE_LOG: every
# line it reads, one a line.
STAND_IN = r'''
import os, re, sys, threading, time

lend_on = os.environ.get("FAKE_LEND", "1") == "1"
multi = int(os.environ.get("FAKE_MULTI", "0") or 0)
busy = [int(os.environ.get("FAKE_BUSY", "0") or 0)]
slow = [int(os.environ.get("FAKE_SLOW", "0") or 0)]
slow_s = float(os.environ.get("FAKE_SLOW_S", "1.2") or 1.2)
log = os.environ.get("FAKE_LOG")
pool = {0: 4096, 1: 2048}                            # dev -> what it can lend with nothing out (MiB)
whole = {1: 15360}                                   # dev -> its whole-card footprint (absent: not whole-lendable)
block = 64
leases, next_id = {}, [7]                            # id -> (dev, mib, whole)
lock = threading.Lock()
stop = threading.Event()


def out(s):
    with lock:
        sys.stdout.write(s + "\n")
        sys.stdout.flush()


def lent(dev):
    return sum(m for d, m, _ in leases.values() if d == dev)


def whole_open(dev):
    return any(d == dev and w for d, _, w in leases.values())


def lendable(dev):
    return 0 if whole_open(dev) else pool[dev] - lent(dev)


def gen(f, rid):
    tag = f"{rid} " if rid is not None else ""
    ids = [int(x) for x in f[-1].split(",")][: int(f[1])]
    for i in ids:
        if stop.is_set() and rid is None:
            break
        time.sleep(0.03)
        out(f"T {tag}{i + 1000}")
    out(f"DONE {tag}{len(ids)} {len(ids)} 1.0 1.0 length")


def lend_cmd(f):
    if f[0] == "LENDABLE":
        if len(f) != 1:
            return ["ERR MALFORMED LENDABLE"]
        return [f"LENDABLE dev={d} lendable={lendable(d)} lent={lent(d)} "
                f"whole={0 if whole_open(d) else whole.get(d, 0)}" for d in sorted(pool)] + ["END"]
    if f[0] == "LEND":
        if len(f) != 3 or not f[1].isdigit() or not (f[2] == "ALL" or f[2].isdigit() and int(f[2]) > 0):
            return ["ERR MALFORMED LEND <dev> <mib>"]
        dev = int(f[1])
        if dev not in pool:
            return [f"ERR NODEV {dev}"]
        if f[2] == "ALL":
            if not whole.get(dev) or lent(dev):
                return [f"ERR MAX {lendable(dev)}"]
            mib = whole[dev]
        else:
            mib = -(-int(f[2]) // block) * block
            if mib > lendable(dev):
                return [f"ERR MAX {lendable(dev)}"]
        i = next_id[0]
        next_id[0] += 5
        leases[i] = (dev, mib, f[2] == "ALL")
        return [f"LENT {i} {dev} {mib}"]
    if len(f) != 2:
        return ["ERR MALFORMED RETURN <id>"]
    i = int(f[1]) if f[1].isdigit() else -1
    if i not in leases:
        return [f"ERR UNKNOWN {f[1]}"]
    if busy[0] > 0:
        busy[0] -= 1
        return [f"ERR BUSY {i}"]
    del leases[i]
    return [f"RETURNED {i}"]


out("VRAM dev=0 pci=0000:01:00.0 total=25296896000 free=751828992 weights=6678575104 pool=15536793108 layers=0-30")
out("VRAM dev=1 pci=0000:0A:00.0 total=25296896000 free=751828992 weights=6678575104 pool=15536793108 layers=31-60")
out("VRAM dev=2 pci=0000:0b:00.0 total=16610951168 free=301989888 weights=268435456 pool=0 layers=61-78")
out("READY 4096 stop" + (f" multi={multi}" if multi else ""))
for line in sys.stdin:
    if log:
        with open(log, "a") as fh:
            fh.write(line)
    f = line.split()
    if not f:
        continue
    if f[0] == "QUIT":
        break
    if f[0] == "STOP":
        stop.set()
        continue
    if f[0] == "GEN":
        stop.clear()
        rid = next((int(x[4:]) for x in f if x.startswith("rid=")), None)
        threading.Thread(target=gen, args=(f, rid), daemon=True).start()
        continue
    if f[0] in ("LEND", "RETURN", "LENDABLE"):
        if not lend_on:
            out(f"ERR unsupported command '{f[0]}' (GEN, GENI, STOP and QUIT)")
            continue
        if f[0] == "LEND" and slow[0] > 0:
            slow[0] -= 1
            time.sleep(slow_s)
        for a in lend_cmd(f):
            out(a)
        continue
    out(f"ERR unsupported command '{f[0]}' (GEN, GENI, STOP and QUIT)")
'''


def stand_in(d: Path) -> str:
    exe = d / "engine"
    exe.write_text(f"#!{sys.executable}\n" + STAND_IN)
    exe.chmod(0o755)
    return str(exe)


class LendBase(unittest.TestCase):
    env: dict = {}

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        d = Path(self.dir.name)
        self.log = d / "lines.log"
        env = dict(os.environ, FAKE_LOG=str(self.log), **self.env)
        self.engine = StrataEngine(stand_in(d), [], env=env)
        tok = ByteTokenizer()
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        if self.engine.alive():
            self.engine.close()
        self.dir.cleanup()

    def call(self, method, path, body=None, raw=None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read() or b"null")

    def lend(self, gpu, mib):
        return self.call("POST", "/vram/lend", {"gpu": gpu, "mib": mib})

    def give_back(self, lease_id):
        return self.call("POST", "/vram/return", {"id": lease_id})

    def state(self):
        code, out = self.call("GET", "/vram/state")
        self.assertEqual(code, 200, out)
        return out

    def gpu(self, state, pci):
        return next(c for c in state["gpus"] if c["gpu"] == pci)

    def engine_lines(self, word):
        try:
            return [x.strip() for x in self.log.read_text().splitlines() if x.startswith(word + " ")]
        except OSError:
            return []


class Routes(LendBase):
    def test_lend_answers_the_lease(self):
        code, out = self.lend("0000:0a:00.0", 100)
        self.assertEqual(code, 200, out)
        self.assertEqual(set(out), {"id", "gpu", "mib"})
        self.assertIsInstance(out["id"], str)
        self.assertEqual((out["gpu"], out["mib"]), ("0000:0a:00.0", 128))   # (the engine's blocks: rounded up)
        self.assertEqual(self.engine_lines("LEND"), ["LEND 1 100"])           # the card's number from its VRAM line

    def test_gpu_is_matched_without_case(self):
        for gpu in ("0000:0A:00.0", "0000:0a:00.0", "00000000:0A:00.0"):
            code, out = self.lend(gpu, 64)
            self.assertEqual(code, 200, (gpu, out))
            self.assertEqual(out["gpu"], "0000:0a:00.0")
        self.assertEqual(self.engine_lines("LEND"), ["LEND 1 64"] * 3)

    def test_unknown_or_malformed_gpu_is_400(self):
        for gpu in ("0000:09:00.0", "0000:01:00.1", "gpu0", "", 3, None, ["0000:01:00.0"]):
            code, out = self.lend(gpu, 64)
            self.assertEqual(code, 400, (gpu, out))
            self.assertIn("error", out)
        code, out = self.call("POST", "/vram/lend", {"mib": 64})
        self.assertEqual(code, 400, out)
        self.assertIn('"gpu"', out["error"])
        self.assertEqual(self.engine_lines("LEND"), [])

    def test_the_old_card_field_is_400_naming_gpu(self):
        code, out = self.call("POST", "/vram/lend", {"card": "0000:01:00.0", "mib": 64})
        self.assertEqual(code, 400, out)
        self.assertIn('"gpu"', out["error"])
        self.assertEqual(self.engine_lines("LEND"), [])

    def test_bad_mib_is_400(self):
        for mib in (0, -5, 1.5, 64.0, "100", True, False, None, [64]):
            code, out = self.lend("0000:01:00.0", mib)
            self.assertEqual(code, 400, (mib, out))
            self.assertIn("error", out)
        code, out = self.call("POST", "/vram/lend", {"gpu": "0000:01:00.0"})
        self.assertEqual(code, 400, out)
        self.assertEqual(self.engine_lines("LEND"), [])

    def test_body_that_is_not_an_object_is_400(self):
        for raw in (b"not json", b"[1, 2]", b"7"):
            code, out = self.call("POST", "/vram/lend", raw=raw)
            self.assertEqual(code, 400, (raw, out))
            code, out = self.call("POST", "/vram/return", raw=raw)
            self.assertEqual(code, 400, (raw, out))

    def test_over_lendable_is_409_with_the_max(self):
        lendable = self.gpu(self.state(), "0000:01:00.0")["lendable"]
        self.assertEqual(lendable, 4096)
        code, out = self.lend("0000:01:00.0", lendable + 1)
        self.assertEqual((code, out), (409, {"max_mib": lendable}))
        self.assertEqual(self.lend("0000:01:00.0", 1024)[0], 200)
        code, out = self.lend("0000:01:00.0", 4096)
        self.assertEqual((code, out), (409, {"max_mib": self.gpu(self.state(), "0000:01:00.0")["lendable"]}))
        self.assertEqual(out["max_mib"], 3072)

    def test_a_card_that_does_not_lend_can_give_0(self):
        code, out = self.lend("0000:0B:00.0", 64)
        self.assertEqual((code, out), (409, {"max_mib": 0}))

    def test_state_shape(self):
        st = self.state()
        self.assertEqual(st, {"gpus": [{"gpu": "0000:01:00.0", "lendable": 4096, "lent": 0},
                                        {"gpu": "0000:0a:00.0", "lendable": 2048, "lent": 0, "whole_mib": 15360},
                                        {"gpu": "0000:0b:00.0", "lendable": 0, "lent": 0}],
                              "leases": []})
        code, a = self.lend("0000:01:00.0", 512)
        self.assertEqual(code, 200)
        st = self.state()
        self.assertEqual(st["leases"], [{"id": a["id"], "gpu": "0000:01:00.0", "mib": 512}])
        self.assertEqual(self.gpu(st, "0000:01:00.0"), {"gpu": "0000:01:00.0", "lendable": 3584, "lent": 512})

    def test_two_leases_on_one_card_return_independently(self):
        code1, a = self.lend("0000:01:00.0", 256)
        code2, b = self.lend("0000:01:00.0", 512)
        self.assertEqual((code1, code2), (200, 200))
        self.assertNotEqual(a["id"], b["id"])
        self.assertEqual(self.gpu(self.state(), "0000:01:00.0")["lent"], 768)
        code, out = self.give_back(a["id"])
        self.assertEqual(code, 200, out)
        self.assertEqual(out, {"id": a["id"], "gpu": "0000:01:00.0", "mib": 256})
        st = self.state()
        self.assertEqual(st["leases"], [{"id": b["id"], "gpu": "0000:01:00.0", "mib": 512}])
        self.assertEqual(self.gpu(st, "0000:01:00.0")["lent"], b["mib"])
        self.assertEqual(self.give_back(b["id"])[0], 200)
        st = self.state()
        self.assertEqual((st["leases"], self.gpu(st, "0000:01:00.0")["lent"]), ([], 0))

    def test_a_numeric_id_is_taken_too(self):
        code, a = self.lend("0000:01:00.0", 64)
        self.assertEqual(code, 200)
        self.assertEqual(self.give_back(int(a["id"]))[0], 200)

    def test_unknown_id_is_404(self):
        self.assertEqual(self.give_back("999")[0], 404)
        self.assertEqual(self.give_back("x9")[0], 404)
        code, a = self.lend("0000:01:00.0", 64)
        self.assertEqual(self.give_back(a["id"])[0], 200)
        code, out = self.give_back(a["id"])                      # (already returned)
        self.assertEqual(code, 404, out)
        self.assertIn("error", out)

    def test_an_id_the_server_did_not_hand_out_is_404_and_not_sent(self):
        # a lease the engine has and the server never recorded: the server does not return it
        self.assertEqual(self.engine.lend_command("LEND 0 64"), ["LENT 7 0 64"])
        code, out = self.give_back("7")
        self.assertEqual(code, 404, out)
        self.assertEqual(self.give_back("999")[0], 404)
        self.assertEqual(self.engine_lines("RETURN"), [])
        self.assertEqual(self.gpu(self.state(), "0000:01:00.0")["lent"], 64)

    def test_an_id_from_before_a_restart_is_404(self):
        # the new process hands out the engine's ids from the start again: an old id never names its lease
        code, a = self.lend("0000:01:00.0", 64)
        self.assertEqual(code, 200, a)
        self.engine.restart()
        code, b = self.lend("0000:01:00.0", 128)
        self.assertEqual(code, 200, b)
        self.assertEqual(self.engine_lines("LENT"), [])
        self.assertNotEqual(b["id"], a["id"])
        code, out = self.give_back(a["id"])
        self.assertEqual(code, 404, out)
        self.assertEqual(self.engine_lines("RETURN"), [])
        st = self.state()
        self.assertEqual(st["leases"], [{"id": b["id"], "gpu": "0000:01:00.0", "mib": 128}])
        self.assertEqual(self.gpu(st, "0000:01:00.0")["lent"], 128)
        self.assertEqual(self.give_back(b["id"])[0], 200)
        self.assertEqual(self.engine_lines("RETURN"), ["RETURN 7"])   # (the new process's own id for it)
        self.assertEqual(self.state()["leases"], [])

    def test_bad_id_is_400(self):
        for i in (None, True, 1.5, "", "1 2", "1\nLEND 0 64", ["1"]):
            code, out = self.give_back(i)
            self.assertEqual(code, 400, (i, out))
        self.assertEqual(self.call("POST", "/vram/return", {})[0], 400)
        self.assertEqual(self.engine_lines("RETURN"), [])
        self.assertEqual(self.engine_lines("LEND"), [])

    def test_restart_ends_every_lease(self):
        self.assertEqual(self.lend("0000:01:00.0", 64)[0], 200)
        self.assertEqual(self.lend("0000:0a:00.0", 64)[0], 200)
        self.engine.restart()
        st = self.state()
        self.assertEqual(st["leases"], [])
        self.assertEqual([c["lent"] for c in st["gpus"]], [0, 0, 0])

    def test_engine_not_running(self):
        code, a = self.lend("0000:01:00.0", 64)
        self.engine.proc.kill()                       # (it died: an idle unload or a crash ends it the same way)
        self.engine.proc.wait()
        st = self.state()
        self.assertEqual(st["leases"], [])
        self.assertEqual([(c["lendable"], c["lent"]) for c in st["gpus"]], [(0, 0)] * 3)
        code, out = self.lend("0000:01:00.0", 64)
        self.assertEqual(code, 503, out)
        self.assertEqual(self.give_back(a["id"])[0], 404)         # (it ended with the engine)

    def test_no_other_routes(self):
        self.assertEqual(self.call("GET", "/vram/lend")[0], 404)
        self.assertEqual(self.call("POST", "/vram/state", {})[0], 404)
        self.assertEqual(self.call("POST", "/be" + "rth/lend", {"gpu": "0000:01:00.0", "mib": 64})[0], 404)


class Busy(LendBase):
    env = {"FAKE_BUSY": "1"}

    def test_busy_is_409_and_the_lease_stays(self):
        code, a = self.lend("0000:01:00.0", 128)
        self.assertEqual(code, 200)
        code, out = self.give_back(a["id"])
        self.assertEqual(code, 409, out)
        self.assertIs(out.get("retry"), True)
        self.assertIn("error", out)
        st = self.state()
        self.assertEqual(st["leases"], [{"id": a["id"], "gpu": "0000:01:00.0", "mib": 128}])
        self.assertEqual(self.gpu(st, "0000:01:00.0")["lent"], 128)
        code, out = self.give_back(a["id"])
        self.assertEqual(code, 200, out)
        self.assertEqual(self.state()["leases"], [])


class LateAnswers(LendBase):
    """A LEND the engine answers after the server stopped waiting (an engine takes the lend commands between its
    steps, and one step can read a long prompt): the answer is that LEND's, never the next command's, and its lease
    goes back to the engine."""
    env = {"FAKE_SLOW": "1"}

    def setUp(self):
        super().setUp()
        self.svc.vram_lend.answer_s = 0.5

    def given_back(self):
        """/vram/state once the late lease is back (the engine read RETURN and nothing is listed any more)."""
        end = time.monotonic() + 10
        while time.monotonic() < end:
            if self.engine_lines("RETURN"):
                st = self.state()
                if not st["leases"]:
                    return st
            time.sleep(0.05)
        self.fail(f"the late lease did not go back: {self.engine_lines('RETURN')}, {self.state()}")

    def test_late_lend_is_given_back(self):
        code, out = self.lend("0000:01:00.0", 64)
        self.assertEqual(code, 503, out)
        self.assertIn("give it back", out["error"])
        st = self.given_back()
        self.assertEqual(self.engine_lines("RETURN"), ["RETURN 7"])
        self.assertEqual([c["lent"] for c in st["gpus"]], [0, 0, 0])
        self.assertEqual(self.gpu(st, "0000:01:00.0")["lendable"], 4096)

    def test_retry_after_a_late_lend_holds_one_lease(self):
        self.assertEqual(self.lend("0000:01:00.0", 64)[0], 503)
        self.svc.vram_lend.answer_s = 10
        code, b = self.lend("0000:01:00.0", 128)
        self.assertEqual(code, 200, b)
        self.assertEqual((b["gpu"], b["mib"]), ("0000:01:00.0", 128))
        self.assertNotEqual(b["id"], "7")
        end = time.monotonic() + 10
        while time.monotonic() < end:
            st = self.state()
            if self.engine_lines("RETURN") and len(st["leases"]) == 1:
                break
            time.sleep(0.05)
        self.assertEqual(self.engine_lines("RETURN"), ["RETURN 7"])       # the late one, not the retry's
        self.assertEqual(st["leases"], [b])
        self.assertEqual(self.gpu(st, "0000:01:00.0")["lent"], 128)

    def test_state_after_a_late_lend_reads_the_engine(self):
        self.assertEqual(self.lend("0000:01:00.0", 64)[0], 503)
        self.svc.vram_lend.answer_s = 10
        st = self.state()
        self.assertEqual(self.gpu(st, "0000:0a:00.0"),
                         {"gpu": "0000:0a:00.0", "lendable": 2048, "lent": 0, "whole_mib": 15360})
        first = self.gpu(st, "0000:01:00.0")
        self.assertEqual(first["lendable"] + first["lent"], 4096)
        self.assertEqual(sum(x["mib"] for x in st["leases"]), first["lent"])   # listed while the engine lends it
        st = self.given_back()
        self.assertEqual(self.gpu(st, "0000:01:00.0"), {"gpu": "0000:01:00.0", "lendable": 4096, "lent": 0})


class WholeCard(LendBase):
    """Lending a whole GPU: "all" or a number above its lendable and at most its whole_mib send LEND <dev> ALL; the
    two kinds of lease never share a GPU, and the server refuses the mixes itself, before the engine sees a LEND."""
    WHOLE = 15360
    CARD = "0000:0a:00.0"                             # whole-lendable: lendable 2048, whole_mib 15360
    POOL = "0000:01:00.0"                             # lends from its pool only (lendable 4096)

    def test_state_shows_whole_mib_on_the_whole_lendable_card_only(self):
        st = self.state()
        self.assertEqual(self.gpu(st, self.CARD), {"gpu": self.CARD, "lendable": 2048, "lent": 0,
                                                   "whole_mib": self.WHOLE})
        for pci in (self.POOL, "0000:0b:00.0"):
            self.assertNotIn("whole_mib", self.gpu(st, pci), pci)

    def check_whole_lend(self, mib):
        code, out = self.lend(self.CARD, mib)
        self.assertEqual(code, 200, (mib, out))
        self.assertEqual(set(out), {"id", "gpu", "mib"})
        self.assertEqual((out["gpu"], out["mib"]), (self.CARD, self.WHOLE))
        self.assertEqual(self.engine_lines("LEND"), ["LEND 1 ALL"])
        st = self.state()
        self.assertEqual(self.gpu(st, self.CARD), {"gpu": self.CARD, "lendable": 0, "lent": self.WHOLE})
        self.assertEqual(st["leases"], [{"id": out["id"], "gpu": self.CARD, "mib": self.WHOLE}])
        self.assertEqual(self.gpu(st, self.POOL), {"gpu": self.POOL, "lendable": 4096, "lent": 0})
        code, back = self.give_back(out["id"])
        self.assertEqual(code, 200, back)
        self.assertEqual(back, out)
        st = self.state()
        self.assertEqual(st["leases"], [])
        self.assertEqual(self.gpu(st, self.CARD), {"gpu": self.CARD, "lendable": 2048, "lent": 0,
                                                   "whole_mib": self.WHOLE})

    def test_all_lends_the_whole_card(self):
        self.check_whole_lend("all")

    def test_a_number_above_lendable_lends_the_whole_card(self):
        self.check_whole_lend(2049)

    def test_whole_mib_itself_lends_the_whole_card(self):
        self.check_whole_lend(self.WHOLE)

    def test_lendable_itself_is_a_pool_lend(self):
        code, out = self.lend(self.CARD, 2048)
        self.assertEqual(code, 200, out)
        self.assertEqual(out["mib"], 2048)
        self.assertEqual(self.engine_lines("LEND"), ["LEND 1 2048"])

    def test_all_on_a_card_without_whole_mib_is_409_with_its_lendable(self):
        before = self.state()
        code, out = self.lend(self.POOL, "all")
        self.assertEqual((code, out), (409, {"max_mib": 4096}))
        code, out = self.lend("0000:0b:00.0", "all")
        self.assertEqual((code, out), (409, {"max_mib": 0}))
        self.assertEqual(self.engine_lines("LEND"), [])
        self.assertEqual(self.state(), before)

    def test_a_number_above_whole_mib_is_409_with_whole_mib(self):
        before = self.state()
        code, out = self.lend(self.CARD, self.WHOLE + 1)
        self.assertEqual((code, out), (409, {"max_mib": self.WHOLE}))
        code, out = self.lend(self.CARD, 10 ** 9)
        self.assertEqual((code, out), (409, {"max_mib": self.WHOLE}))
        self.assertEqual(self.engine_lines("LEND"), [])
        self.assertEqual(self.state(), before)

    def test_bad_mib_is_400(self):
        for mib in (True, "ALL ", 0, -1, "All", "all ", " all", "ALL", "2049", 2049.0):
            code, out = self.lend(self.CARD, mib)
            self.assertEqual(code, 400, (mib, out))
            self.assertIn("error", out)
        self.assertEqual(self.engine_lines("LEND"), [])

    def test_whole_lend_is_refused_while_a_pool_lease_is_open(self):
        code, a = self.lend(self.CARD, 64)
        self.assertEqual(code, 200, a)
        before = self.state()
        self.assertEqual(self.gpu(before, self.CARD), {"gpu": self.CARD, "lendable": 1984, "lent": 64,
                                                       "whole_mib": self.WHOLE})
        for mib in ("all", 2049, self.WHOLE):
            code, out = self.lend(self.CARD, mib)
            self.assertEqual((code, out), (409, {"max_mib": 1984}), mib)
        self.assertEqual(self.engine_lines("LEND"), ["LEND 1 64"])          # the engine saw no whole lend
        self.assertEqual(self.state(), before)
        self.assertEqual(self.give_back(a["id"])[0], 200)
        self.assertEqual(self.lend(self.CARD, "all")[0], 200)              # free again: the whole card goes

    def test_pool_lend_is_refused_while_the_whole_lease_is_open(self):
        code, a = self.lend(self.CARD, "all")
        self.assertEqual(code, 200, a)
        before = self.state()
        self.assertEqual(self.gpu(before, self.CARD), {"gpu": self.CARD, "lendable": 0, "lent": self.WHOLE})
        for mib in (1024, 64, 2049, "all"):
            code, out = self.lend(self.CARD, mib)
            self.assertEqual((code, out), (409, {"max_mib": 0}), mib)
        self.assertEqual(self.engine_lines("LEND"), ["LEND 1 ALL"])         # nothing more went to the engine
        self.assertEqual(self.state(), before)
        self.assertEqual(self.lend(self.POOL, 64)[0], 200)                 # another card lends as before
        self.assertEqual(self.give_back(a["id"])[0], 200)
        self.assertEqual(self.lend(self.CARD, 1024)[0], 200)


class LateWholeAnswers(LendBase):
    """The engine answers a LEND "45 s" late: past a pool lend's 30 s wait (503, and the server gives the late lease
    back) but inside a whole-card lend's 180 s (200).  The seconds are scaled: the stand-in answers after 1.2 s and the
    two waits are set to 0.8 s and 4.8 s - the same ratios as 45 s against 30 s and 180 s."""
    env = {"FAKE_SLOW": "1", "FAKE_SLOW_S": "1.2"}

    def setUp(self):
        super().setUp()
        self.svc.vram_lend.answer_s = 0.8
        self.svc.vram_lend.whole_answer_s = 4.8

    def returned(self, n=1):
        end = time.monotonic() + 10
        while time.monotonic() < end:
            if len(self.engine_lines("RETURN")) >= n and not self.state()["leases"]:
                return
            time.sleep(0.05)
        self.fail(f"the late lease did not go back: {self.engine_lines('RETURN')}, {self.state()}")

    def test_a_pool_lend_answered_at_45_s_is_503_and_given_back(self):
        code, out = self.lend("0000:0a:00.0", 64)
        self.assertEqual(code, 503, out)
        self.assertIn("give it back", out["error"])
        self.returned()
        self.assertEqual(self.engine_lines("RETURN"), ["RETURN 7"])
        self.assertEqual(self.gpu(self.state(), "0000:0a:00.0"),
                         {"gpu": "0000:0a:00.0", "lendable": 2048, "lent": 0, "whole_mib": 15360})

    def test_a_whole_lend_answered_at_45_s_is_200(self):
        t0 = time.monotonic()
        code, out = self.lend("0000:0a:00.0", "all")
        self.assertEqual(code, 200, out)
        self.assertGreater(time.monotonic() - t0, 1.0)
        self.assertEqual((out["gpu"], out["mib"]), ("0000:0a:00.0", 15360))
        self.assertEqual(self.engine_lines("LEND"), ["LEND 1 ALL"])
        self.assertEqual(self.engine_lines("RETURN"), [])
        st = self.state()
        self.assertEqual(st["leases"], [{"id": out["id"], "gpu": "0000:0a:00.0", "mib": 15360}])
        self.assertEqual(self.gpu(st, "0000:0a:00.0"), {"gpu": "0000:0a:00.0", "lendable": 0, "lent": 15360})
        self.assertEqual(self.give_back(out["id"])[0], 200)

    def test_a_whole_lend_past_its_own_wait_is_given_back(self):
        self.svc.vram_lend.whole_answer_s = 0.5
        code, out = self.lend("0000:0a:00.0", "all")
        self.assertEqual(code, 503, out)
        self.assertIn("give it back", out["error"])
        self.returned()
        self.assertEqual(self.engine_lines("RETURN"), ["RETURN 7"])
        self.assertEqual(self.gpu(self.state(), "0000:0a:00.0"),
                         {"gpu": "0000:0a:00.0", "lendable": 2048, "lent": 0, "whole_mib": 15360})

    def test_the_default_waits(self):
        from serve import vram_lend
        self.assertEqual((vram_lend.POOL_WAIT_S, vram_lend.WHOLE_WAIT_S), (30.0, 180.0))
        lender = vram_lend.Lender()
        self.assertEqual((lender.answer_s, lender.whole_answer_s), (30.0, 180.0))


class WhileRequestsRun(LendBase):
    """A lend, a return and the state while a request runs: not behind the request queue, and the answers stay out of
    the request's own lines."""

    def run_request(self, n=60):
        ids = list(range(1, n + 1))
        got = {}

        def one():
            got["toks"] = [t for t in self.engine.generate(ids, n, {}, threading.Event()) if t is not None]
        th = threading.Thread(target=one)
        th.start()
        return ids, got, th

    def check(self):
        self.svc.fifo.acquire()                       # (every place: the request queue is held by "requests")
        try:
            ids, got, th = self.run_request()
            time.sleep(0.2)
            t0 = time.monotonic()
            code, a = self.lend("0000:01:00.0", 64)
            self.assertEqual(code, 200, a)
            code, out = self.lend("0000:01:00.0", 99999)
            self.assertEqual(code, 409, out)
            st = self.state()
            self.assertEqual(st["leases"], [{"id": a["id"], "gpu": "0000:01:00.0", "mib": 64}])
            self.assertEqual(self.give_back(a["id"])[0], 200)
            self.assertEqual(self.give_back(a["id"])[0], 404)
            self.assertLess(time.monotonic() - t0, 1.5)
            self.assertTrue(th.is_alive())            # (all of it while the request ran)
            th.join(30)
            self.assertEqual(got.get("toks"), [i + 1000 for i in ids])
        finally:
            self.svc.fifo.release()

    def test_one_conversation(self):
        self.check()


class WhileRequestsRunMulti(WhileRequestsRun):
    env = {"FAKE_MULTI": "2"}

    def test_one_conversation(self):
        self.assertEqual(self.engine.multi, 2)
        self.check()


class Unsupported(LendBase):
    env = {"FAKE_LEND": "0"}

    def test_engine_without_lending(self):
        code, out = self.lend("0000:01:00.0", 64)
        self.assertEqual(code, 501, out)
        self.assertIn("error", out)
        self.assertEqual(self.give_back("7")[0], 501)
        self.assertEqual(self.state(), {"gpus": [{"gpu": "0000:01:00.0", "lendable": 0, "lent": 0},
                                                  {"gpu": "0000:0a:00.0", "lendable": 0, "lent": 0},
                                                  {"gpu": "0000:0b:00.0", "lendable": 0, "lent": 0}],
                                        "leases": []})
        # the engine's ERR lines for a request are still the request's
        ids = [1, 2, 3]
        self.assertEqual([t for t in self.engine.generate(ids, 3, {}, threading.Event()) if t is not None],
                         [1001, 1002, 1003])


class NotAMayaEngine(unittest.TestCase):
    def test_mock_engine(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "hi"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            def call(method, path, body=None):
                req = urllib.request.Request(base + path, method=method,
                                             data=json.dumps(body).encode() if body is not None else None)
                try:
                    with urllib.request.urlopen(req, timeout=30) as r:
                        return r.status, json.loads(r.read())
                except urllib.error.HTTPError as e:
                    with e:
                        return e.code, json.loads(e.read())
            self.assertEqual(call("GET", "/vram/state"), (200, {"gpus": [], "leases": []}))
            self.assertEqual(call("POST", "/vram/lend", {"gpu": "0000:01:00.0", "mib": 64})[0], 501)
            self.assertEqual(call("POST", "/vram/return", {"id": "7"})[0], 501)
        finally:
            httpd.shutdown()
            httpd.server_close()


class Answers(unittest.TestCase):
    """Which engine lines are lend answers (taken off the request's lines) and which are not."""

    def test_is_answer(self):
        from serve import vram_lend
        for line in ("LENT 7 0 128\n", "RETURNED 7\n", "LENDABLE dev=0 lendable=4096 lent=0\n", "END\n",
                     "LENDABLE dev=1 lendable=2048 lent=0 whole=15360\n",
                     "ERR MAX 4096\n", "ERR NODEV 3\n", "ERR BUSY 7\n", "ERR UNKNOWN x9\n",
                     "ERR MALFORMED LEND <dev> <mib>\n", "ERR MALFORMED RETURN <id>\n", "ERR MALFORMED LENDABLE\n",
                     "ERR unsupported command 'LEND' (GEN, GENI, STOP and QUIT)\n",
                     "ERR unsupported command 'LENDABLE' (GEN, GENI, LOGP, STOP and QUIT)\n"):
            self.assertTrue(vram_lend.is_answer(line), line)
        for line in ("T 5\n", "DONE 3 3 1.0 1.0 length\n", "ERR non-finite\n", "ERR 3 non-finite\n", "RELENT 123\n",
                     "ERR unsupported command 'LOGP' (GEN, GENI, STOP and QUIT)\n", "ENDED\n",
                     "ERR malformed GEN line\n", "ERR 4 all 2 conversation states are busy\n"):
            self.assertFalse(vram_lend.is_answer(line), line)


class SourceScan(unittest.TestCase):
    def test_no_foreign_route_names(self):
        word = "/be" + "rth"
        for name in ("server.py", "vram_lend.py"):
            text = (ROOT / "serve" / name).read_text(encoding="utf-8")
            self.assertNotIn(word, text.lower(), name)

    def test_vram_json_names_no_card(self):
        # the /vram JSON names a GPU "gpu" (and the list "gpus"): no "card" key in a dict literal or a lookup
        import re
        key = re.compile(r"""["']cards?["']\s*:|\[\s*["']cards?["']\s*\]|\.get\(\s*["']cards?["']""")
        for name in ("vram_lend.py", "server.py"):
            text = (ROOT / "serve" / name).read_text(encoding="utf-8")
            self.assertEqual(key.findall(text), [], name)


if __name__ == "__main__":
    unittest.main()
