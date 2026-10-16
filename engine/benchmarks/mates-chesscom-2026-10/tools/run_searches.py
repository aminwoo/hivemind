"""Search every dumped position with hivemind and append results to JSONL.

Usage: python run_searches.py positions.json results.jsonl ENGINE MODEL MOVETIME_MS WORKERS
Resumable: positions already in results.jsonl are skipped.
"""
import json
import queue
import subprocess
import sys
import threading
import time

positions_path, results_path, engine, model, movetime, workers = sys.argv[1:7]
movetime, workers = int(movetime), int(workers)
positions = json.load(open(positions_path))


def key(p):
    return f"{p['game']}:{p['ply']}:{p['team']}"


done = set()
try:
    for line in open(results_path):
        done.add(key(json.loads(line)))
except FileNotFoundError:
    pass
todo = [p for p in positions if key(p) not in done]
print(f"{len(todo)} of {len(positions)} searches to run", flush=True)
jobs = queue.Queue()
for p in todo:
    jobs.put(p)
write_lock = threading.Lock()


class Engine:
    def __init__(self):
        self.proc = subprocess.Popen(
            [engine, "--model", model], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.send("uci")
        self.read_until("uciok", 1800)
        self.send("isready")
        self.read_until("readyok", 1800)

    def send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def read_until(self, token, timeout):
        lines, start = [], time.time()
        while time.time() - start < timeout:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine exited")
            line = line.rstrip("\n")
            lines.append(line)
            if line.startswith(token):
                return lines
        raise TimeoutError(token)

    def search(self, p):
        self.send("ucinewgame")
        for name, value in (("Team", p["team"]), ("MultiPV", 1), ("TimeModel", "clock"),
                            ("TimeLeadMs", p["leadMs"]), ("TeamMoveTimeMs", 100),
                            ("OpponentMoveTimeMs", 100)):
            self.send(f"setoption name {name} value {value}")
        moves = " ".join(p["moves"])
        self.send(f"position {p['position']}" + (f" moves {moves}" if moves else ""))
        self.send("isready")
        self.read_until("readyok", 60)
        started = time.time()
        self.send(f"go movetime {movetime}")
        lines = self.read_until("bestmove", movetime / 1000 + 120)
        elapsed = round((time.time() - started) * 1000)
        # The engine keeps thinking ahead after bestmove; stop it.
        self.send("stop")
        infos = [l for l in lines if l.startswith("info depth")]
        last = infos[-1] if infos else ""
        tok = last.split()
        result = {"elapsed_ms": elapsed, "bestmove": lines[-1].split()[1] if len(lines[-1].split()) > 1 else None,
                  "info": last, "strings": [l for l in lines if l.startswith("info string")][-6:]}
        if "score" in tok:
            i = tok.index("score")
            result["score_kind"], result["score"] = tok[i + 1], int(tok[i + 2])
        for name in ("depth", "nodes", "time"):
            if name in tok:
                result[name] = int(tok[tok.index(name) + 1])
        if "pv" in tok:
            result["pv"] = tok[tok.index("pv") + 1:]
        return result


def worker(index):
    eng = Engine()
    while True:
        try:
            p = jobs.get_nowait()
        except queue.Empty:
            break
        try:
            r = eng.search(p)
        except Exception as error:  # restart the engine and move on
            r = {"error": repr(error)}
            try:
                eng.proc.kill()
            except Exception:
                pass
            eng = Engine()
        out = {k: p[k] for k in ("game", "partner", "ply", "team", "leadMs", "fen", "position", "moves", "nextMove")}
        out.update(r)
        with write_lock:
            with open(results_path, "a") as f:
                f.write(json.dumps(out) + "\n")
            done.add(key(p))
            if len(done) % 25 == 0:
                print(f"{len(done)}/{len(positions)} done", flush=True)
    eng.send("quit")


threads = [threading.Thread(target=worker, args=(i,)) for i in range(workers)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print("finished", flush=True)
