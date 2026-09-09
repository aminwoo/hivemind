#!/usr/bin/env python3
"""Drive the Hivemind UCI engine, waiting for each response instead of
firing every command at once. `go` is asynchronous: piping `quit` straight
after it aborts the search before a single node is expanded."""
import subprocess, sys, time, threading, queue, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--position", default="startpos")
    ap.add_argument("--go", default="go movetime 3000")
    ap.add_argument("--setoption", action="append", default=[])
    ap.add_argument("--timeout", type=float, default=180.0)
    a = ap.parse_args()

    p = subprocess.Popen([a.bin, "--model", a.model],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    q = queue.Queue()
    threading.Thread(target=lambda: [q.put(l.rstrip("\n")) for l in p.stdout],
                     daemon=True).start()

    def send(cmd):
        print(f">>> {cmd}", flush=True)
        p.stdin.write(cmd + "\n"); p.stdin.flush()

    def wait_for(token, timeout):
        end = time.time() + timeout
        lines = []
        while time.time() < end:
            try:
                line = q.get(timeout=0.2)
            except queue.Empty:
                continue
            lines.append(line)
            print(line, flush=True)
            if line.startswith(token):
                return lines, True
        return lines, False

    send("uci");     wait_for("uciok", 60)
    for opt in a.setoption:
        send(f"setoption name {opt}")
    send("isready"); wait_for("readyok", 60)
    send(f"position {a.position}")
    send("isready"); wait_for("readyok", 60)

    t0 = time.time()
    send(a.go)
    lines, ok = wait_for("bestmove", a.timeout)
    elapsed = time.time() - t0
    send("quit")
    try: p.wait(timeout=10)
    except subprocess.TimeoutExpired: p.kill()

    best = next((l for l in lines if l.startswith("bestmove")), None)
    print(f"\n=== {elapsed:.2f}s -> {best or 'NO BESTMOVE'} ===")
    return 0 if (ok and best and "(none)" not in best) else 1

sys.exit(main())
