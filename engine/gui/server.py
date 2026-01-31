#!/usr/bin/env python3
"""
Hivemind Bughouse GUI Server

A Flask-based web server that serves the bughouse visualization GUI,
provides real-time game state, and can start/stop evaluation games.

Usage:
    python server.py [--port PORT] [--engine-path PATH]
"""

import argparse
import json
import os
import signal
import subprocess
import threading
import time
from glob import glob
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS

app = Flask(__name__, static_folder='.')
CORS(app)

# Configuration
STATE_FILE = './game_state.json'
ENGINE_PATH = '../build/hivemind'
NETWORKS_DIR = '../networks'

# Engine process management
engine_process = None
engine_lock = threading.Lock()

# Last known state (for caching)
last_state = {
    'fenA': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
    'fenB': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
    'sideToMove': 'w',
    'ply': 0,
    'gameNumber': 0,
    'totalGames': 0,
    'whiteTeam': 'Player 1',
    'blackTeam': 'Player 2',
    'result': 'ongoing',
    'moves': [],
    'player1Wins': 0,
    'player1Losses': 0,
    'draws': 0,
    'timestamp': 0,
    'engineRunning': False
}

last_modified = 0


def find_onnx_models():
    """Find all ONNX models in the networks directory."""
    models = []
    networks_path = Path(NETWORKS_DIR)
    if networks_path.exists():
        for onnx_file in sorted(networks_path.glob('*.onnx'), reverse=True):
            models.append({
                'name': onnx_file.stem,
                'path': str(onnx_file.absolute())
            })
    return models


def read_game_state():
    """Read the current game state from the JSON file."""
    global last_state, last_modified
    
    try:
        state_path = Path(STATE_FILE)
        if not state_path.exists():
            state = last_state.copy()
            state['engineRunning'] = is_engine_running()
            return state
        
        # Check if file was modified
        mtime = state_path.stat().st_mtime
        if mtime <= last_modified:
            state = last_state.copy()
            state['engineRunning'] = is_engine_running()
            return state
        
        with open(state_path, 'r') as f:
            new_state = json.load(f)
            new_state['engineRunning'] = is_engine_running()
            last_state = new_state
            last_modified = mtime
            return new_state
            
    except (json.JSONDecodeError, IOError) as e:
        state = last_state.copy()
        state['engineRunning'] = is_engine_running()
        return state


def is_engine_running():
    """Check if the engine process is running."""
    global engine_process
    with engine_lock:
        if engine_process is None:
            return False
        return engine_process.poll() is None


def start_engine(config):
    """Start the engine with the given configuration."""
    global engine_process
    
    with engine_lock:
        # Kill existing process if running
        if engine_process is not None and engine_process.poll() is None:
            engine_process.terminate()
            try:
                engine_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                engine_process.kill()
        
        # Clear previous game state
        state_path = Path(STATE_FILE)
        if state_path.exists():
            state_path.unlink()
        
        # Build command
        cmd = [ENGINE_PATH]
        
        eval_type = config.get('evalType', 'eval')
        cmd.append(eval_type)
        
        if eval_type == 'eval':
            cmd.extend(['--new', config.get('newModel', '')])
            cmd.extend(['--old', config.get('oldModel', '')])
        else:  # param-eval
            cmd.extend(['--model', config.get('model', '')])
            
            # Player 1 settings
            if config.get('p1Name'):
                cmd.extend(['--p1-name', config['p1Name']])
            if config.get('p1Nodes'):
                cmd.extend(['--p1-nodes', str(config['p1Nodes'])])
            if config.get('p1Time'):
                cmd.extend(['--p1-time', str(config['p1Time'])])
            if config.get('p1Cpuct'):
                cmd.extend(['--p1-cpuct', str(config['p1Cpuct'])])
                
            # Player 2 settings
            if config.get('p2Name'):
                cmd.extend(['--p2-name', config['p2Name']])
            if config.get('p2Nodes'):
                cmd.extend(['--p2-nodes', str(config['p2Nodes'])])
            if config.get('p2Time'):
                cmd.extend(['--p2-time', str(config['p2Time'])])
            if config.get('p2Cpuct'):
                cmd.extend(['--p2-cpuct', str(config['p2Cpuct'])])
        
        # Common settings
        cmd.extend(['--games', str(config.get('numGames', 10))])
        
        if config.get('moveTimeMs'):
            cmd.extend(['--time', str(config['moveTimeMs'])])
        elif config.get('nodesPerMove'):
            cmd.extend(['--nodes', str(config['nodesPerMove'])])
            
        if config.get('temperature'):
            cmd.extend(['--temperature', str(config['temperature'])])
        if config.get('tempMoves'):
            cmd.extend(['--temp-moves', str(config['tempMoves'])])
            
        cmd.append('--gui')
        cmd.extend(['--gui-path', str(Path(STATE_FILE).absolute())])
        
        if config.get('verbose'):
            cmd.append('--verbose')
        
        print(f"Starting engine: {' '.join(cmd)}")
        
        # Start process
        engine_process = subprocess.Popen(
            cmd,
            cwd=str(Path(ENGINE_PATH).parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Start output reader thread
        def read_output():
            for line in engine_process.stdout:
                print(f"[ENGINE] {line}", end='')
        
        threading.Thread(target=read_output, daemon=True).start()
        
        return True


def stop_engine():
    """Stop the running engine process."""
    global engine_process
    
    with engine_lock:
        if engine_process is not None:
            engine_process.terminate()
            try:
                engine_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                engine_process.kill()
            engine_process = None
            return True
    return False


@app.route('/')
def index():
    """Serve the main GUI page."""
    return send_file('index.html')


@app.route('/<path:filename>')
def static_files(filename):
    """Serve static files."""
    return send_from_directory('.', filename)


@app.route('/api/game-state')
def get_game_state():
    """API endpoint to get current game state."""
    state = read_game_state()
    return jsonify(state)


@app.route('/api/models')
def get_models():
    """API endpoint to get available ONNX models."""
    models = find_onnx_models()
    return jsonify({'models': models})


@app.route('/api/start', methods=['POST'])
def api_start():
    """API endpoint to start the engine."""
    config = request.json
    if not config:
        return jsonify({'error': 'No configuration provided'}), 400
    
    try:
        start_engine(config)
        return jsonify({'status': 'started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stop', methods=['POST'])
def api_stop():
    """API endpoint to stop the engine."""
    stopped = stop_engine()
    return jsonify({'status': 'stopped' if stopped else 'not_running'})


@app.route('/api/status')
def api_status():
    """API endpoint to get engine status."""
    return jsonify({
        'running': is_engine_running(),
        'timestamp': time.time()
    })


@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'timestamp': time.time()})


def main():
    parser = argparse.ArgumentParser(description='HiveMind Bughouse GUI Server')
    parser.add_argument('--port', type=int, default=8080,
                        help='Port to run the server on (default: 8080)')
    parser.add_argument('--state-file', type=str, default='./game_state.json',
                        help='Path to the game state JSON file')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--engine', type=str, default='../build/hivemind',
                        help='Path to the hivemind engine executable')
    parser.add_argument('--networks', type=str, default='../networks',
                        help='Path to the networks directory')
    args = parser.parse_args()
    
    global STATE_FILE, ENGINE_PATH, NETWORKS_DIR
    STATE_FILE = args.state_file
    ENGINE_PATH = args.engine
    NETWORKS_DIR = args.networks
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           🐝 Hivemind Bughouse GUI Server 🐝                 ║
╠══════════════════════════════════════════════════════════════╣
║  Open your browser to: http://localhost:{args.port:<5}               ║
║  Engine: {args.engine:<50} ║
║  Networks: {args.networks:<48} ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Handle shutdown gracefully
    def signal_handler(sig, frame):
        print("\nShutting down...")
        stop_engine()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the server
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
