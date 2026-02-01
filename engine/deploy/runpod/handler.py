#!/usr/bin/env python3
"""
Hivemind Bughouse Engine - RunPod Serverless Handler

Handles inference requests for the bughouse chess engine.
Receives board positions, runs MCTS search, returns best moves.

UCI Protocol Summary for Hivemind:
- FEN format: "fenA|fenB" (two boards separated by |)
- Move format: "1e2e4" (board number prefix: 1=A, 2=B)
- Bestmove format: "(moveA,moveB)" or "(e2e4,pass)"
- Options: Hash, MultiPV, Team (white/black), Mode (sit/go)
"""

import os
import subprocess
import json
import re
import time
import threading
import queue
from typing import Dict, Any, Optional, List

import runpod

# Configuration
ENGINE_PATH = os.environ.get('ENGINE_PATH', '/app/build/hivemind')
NETWORKS_DIR = os.environ.get('NETWORKS_DIR', '/app/networks')
DEFAULT_NODES = 800
DEFAULT_MOVE_TIME_MS = 1000
MAX_MOVE_TIME_MS = 30000  # 30 second max


class HivemindEngine:
    """Manages a Hivemind UCI engine session for bughouse."""
    
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.initialized = False
        self.output_queue: queue.Queue = queue.Queue()
        self.reader_thread: Optional[threading.Thread] = None
        self.running = False
        
    def _reader_worker(self):
        """Background thread to read engine output."""
        while self.running and self.process and self.process.stdout:
            try:
                line = self.process.stdout.readline()
                if line:
                    line = line.strip()
                    print(f"<<< {line}", flush=True)
                    self.output_queue.put(line)
                elif self.process.poll() is not None:
                    # Process exited
                    break
            except Exception as e:
                print(f"Reader error: {e}", flush=True)
                break
        
    def start(self):
        """Start the engine process."""
        if self.process is not None and self.process.poll() is None:
            return  # Already running
            
        print(f"Starting engine: {ENGINE_PATH}")
        self.process = subprocess.Popen(
            [ENGINE_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,
            cwd="/app"  # Run from /app where networks/ directory is located
        )
        
        # Start reader thread
        self.running = True
        self.reader_thread = threading.Thread(target=self._reader_worker, daemon=True)
        self.reader_thread.start()
        
        # Wait a moment for process to start
        time.sleep(0.1)
        
        # Initialize UCI
        self._send("uci")
        response = self._wait_for("uciok", timeout=60)
        print(f"UCI response: {response[:3]}...")
        
        self._send("isready")
        self._wait_for("readyok", timeout=60)
        
        self.initialized = True
        print("Engine initialized successfully")
        
    def stop(self):
        """Stop the engine process."""
        if self.process:
            try:
                self.running = False
                self._send("quit")
                self.process.wait(timeout=5)
            except:
                self.process.kill()
            self.process = None
            self.initialized = False
            
    def _send(self, cmd: str):
        """Send a command to the engine."""
        if self.process and self.process.stdin:
            print(f">>> {cmd}", flush=True)
            self.process.stdin.write(cmd + "\n")
            self.process.stdin.flush()
            
    def _read_line(self, timeout: float = 10) -> Optional[str]:
        """Read a line from engine output with timeout."""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        
    def _wait_for(self, expected: str, timeout: float = 60) -> List[str]:
        """Wait for a specific response, collecting all lines."""
        lines = []
        start = time.time()
        while time.time() - start < timeout:
            line = self._read_line(timeout=1)
            if line:
                lines.append(line)
                if expected in line:
                    return lines
        raise TimeoutError(f"Timeout waiting for '{expected}'")
        
    def set_option(self, name: str, value: str):
        """Set a UCI option."""
        self._send(f"setoption name {name} value {value}")
        
    def new_game(self):
        """Reset for a new game."""
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok", timeout=30)
        
    def set_position(self, fen: str, moves: List[str] = None):
        """
        Set the board position.
        
        Args:
            fen: Bughouse FEN in format "fenA|fenB"
            moves: Optional list of moves in format "1e2e4" (board prefix + move)
        """
        if fen == "startpos":
            cmd = "position startpos"
        else:
            cmd = f"position fen {fen}"
            
        if moves:
            cmd += " moves " + " ".join(moves)
            
        self._send(cmd)
        
    def go(self, movetime: int = None, nodes: int = None) -> Dict[str, Any]:
        """
        Run search and return results.
        
        Hivemind uses "go movetime <ms>" format.
        
        Returns:
            Dict with:
            - bestmove: "(moveA,moveB)" format
            - info: List of info strings
            - eval: Centipawn evaluation (if available)
            - nodes: Nodes searched
            - depth: Search depth
            - time_ms: Time taken
        """
        if movetime:
            self._send(f"go movetime {movetime}")
        elif nodes:
            # Hivemind currently uses movetime, estimate time from nodes
            # Rough estimate: 1000 nodes/second
            estimated_time = max(100, nodes)  # At least 100ms
            self._send(f"go movetime {estimated_time}")
        else:
            self._send(f"go movetime {DEFAULT_MOVE_TIME_MS}")
            
        # Calculate timeout based on search time
        search_time = movetime if movetime else (nodes if nodes else DEFAULT_MOVE_TIME_MS)
        timeout = (search_time / 1000) + 30  # Add 30 second buffer
        
        # Collect output until bestmove
        info_lines = []
        result = {
            "bestmove": None,
            "info": [],
            "eval": None,
            "nodes": None,
            "depth": None,
            "time_ms": None
        }
        
        start = time.time()
        while time.time() - start < timeout:
            line = self._read_line(timeout=1)
            if not line:
                continue
                
            if line.startswith("info"):
                info_lines.append(line)
                result["info"].append(line)
                
                # Parse info line
                self._parse_info(line, result)
                
            elif line.startswith("bestmove"):
                # Parse bestmove: "bestmove (e2e4,d2d4)"
                match = re.search(r'bestmove\s+(\([^)]+\)|\S+)', line)
                if match:
                    result["bestmove"] = match.group(1)
                    
                result["time_ms"] = int((time.time() - start) * 1000)
                return result
                
        raise TimeoutError("Timeout waiting for bestmove")
        
    def _parse_info(self, line: str, result: Dict):
        """Parse UCI info line and update result dict."""
        parts = line.split()
        
        for i, part in enumerate(parts):
            if part == "depth" and i + 1 < len(parts):
                try:
                    result["depth"] = int(parts[i + 1])
                except ValueError:
                    pass
                    
            elif part == "nodes" and i + 1 < len(parts):
                try:
                    result["nodes"] = int(parts[i + 1])
                except ValueError:
                    pass
                    
            elif part == "cp" and i + 1 < len(parts):
                try:
                    result["eval"] = int(parts[i + 1]) / 100.0
                except ValueError:
                    pass
                    
            elif part == "mate" and i + 1 < len(parts):
                try:
                    mate_in = int(parts[i + 1])
                    # Convert mate to large eval
                    result["eval"] = 100.0 if mate_in > 0 else -100.0
                    result["mate"] = mate_in
                except ValueError:
                    pass


# Global engine instance (reused across warm requests)
engine = HivemindEngine()


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler.
    
    Input format:
    {
        "input": {
            "fen": "rnbqkbnr/...|rnbqkbnr/...",  # Bughouse FEN (boardA|boardB)
            "movetime": 2500,  # Search time in milliseconds
            "moves": "1e2e4 1e7e5",  # Optional: space-separated moves with board prefix
            "mode": "go",  # "go" (normal) or "sit" (time advantage)
            "team": "white",  # Which team we're playing as
            "multipv": false  # Whether to return multiple PVs
        }
    }
    
    Output format:
    {
        "bestmove": "(e2e4,d2d4)",  # Joint action
        "moveA": "e2e4",  # Board A move
        "moveB": "d2d4",  # Board B move  
        "eval": 0.15,  # Evaluation in pawns
        "depth": 12,
        "nodes": 5000,
        "time_ms": 1000,
        "info": [...]  # Raw UCI info strings
    }
    """
    global engine
    
    try:
        job_input = job.get("input", {})
        action = job_input.get("action", "move")
        
        # Ensure engine is running
        if not engine.initialized:
            engine.start()
            
        # Handle different actions
        if action == "newgame":
            engine.new_game()
            return {"status": "ok", "message": "New game started"}
            
        # Default action is "move" - no need for explicit action field
        fen = job_input.get("fen")
        if not fen:
            return {"error": "Missing 'fen' parameter"}
            
        # Set team if provided
        team = job_input.get("team", "white")
        engine.set_option("Team", team)
        
        # Set mode if provided
        mode = job_input.get("mode", "go")
        engine.set_option("Mode", mode)
        
        # Set MultiPV if requested
        multipv = job_input.get("multipv", False)
        if multipv:
            engine.set_option("MultiPV", "5")
        else:
            engine.set_option("MultiPV", "1")
            
        # Set position - moves can be string or list
        moves = job_input.get("moves", "")
        if isinstance(moves, str) and moves:
            moves = moves.split()
        engine.set_position(fen, moves if moves else None)
        
        # Get search parameters
        movetime = job_input.get("movetime", 0)
            
        # Clamp for safety
        if movetime > MAX_MOVE_TIME_MS:
            movetime = MAX_MOVE_TIME_MS
        if not movetime:
            movetime = DEFAULT_MOVE_TIME_MS
            
        # Run search
        result = engine.go(movetime=movetime)
        
        # Parse joint action into individual moves
        bestmove = result.get("bestmove", "(none)")
        moveA, moveB = parse_joint_action(bestmove)
        
        return {
            "bestmove": bestmove,
            "moveA": moveA,
            "moveB": moveB,
            "eval": result.get("eval"),
            "depth": result.get("depth"),
            "nodes": result.get("nodes"),
            "time_ms": result.get("time_ms"),
            "mate": result.get("mate"),
            "info": result.get("info", [])
        }
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        # Try to restart engine on error
        try:
            engine.stop()
        except:
            pass
            
        return {"error": str(e)}


def parse_joint_action(bestmove: str) -> tuple:
    """
    Parse joint action format "(moveA,moveB)" into individual moves.
    
    Examples:
        "(e2e4,d2d4)" -> ("e2e4", "d2d4")
        "(e2e4,pass)" -> ("e2e4", None)
        "(pass,d2d4)" -> (None, "d2d4")
        "(none)" -> (None, None)
    """
    if not bestmove or bestmove == "(none)":
        return (None, None)
        
    # Remove parentheses
    inner = bestmove.strip("()")
    
    if "," not in inner:
        return (None, None)
        
    parts = inner.split(",")
    if len(parts) != 2:
        return (None, None)
        
    moveA = parts[0].strip() if parts[0].strip() != "pass" else None
    moveB = parts[1].strip() if parts[1].strip() != "pass" else None
    
    return (moveA, moveB)


# For local testing
if __name__ == "__main__":
    import sys
    
    # Test the handler locally
    test_job = {
        "input": {
            "action": "move",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1|rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "movetime": 1000,
            "team": "white"
        }
    }
    
    print("Testing Hivemind RunPod handler...")
    print(f"Engine path: {ENGINE_PATH}")
    print(f"Networks dir: {NETWORKS_DIR}")
    print()
    
    result = handler(test_job)
    print()
    print("Result:")
    print(json.dumps(result, indent=2))
else:
    # RunPod serverless mode
    runpod.serverless.start({"handler": handler})
