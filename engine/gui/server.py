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
import re
import signal
import subprocess
import threading
import time
from glob import glob
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS
import chess
import chess.variant

app = Flask(__name__, static_folder='.')
CORS(app)

MOVE_PREFIX_RE = re.compile(r'^\d+([ABab])\.$')
TAG_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')
RESULT_TOKENS = {'1-0', '0-1', '1/2-1/2', '*'}
PARSER_VERSION = 'bughouse-parser-2026-08-01-v3'

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


def _clean_single_fen(fen: str) -> str:
    """Normalize a single-board FEN that may have minor formatting quirks."""
    parts = fen.strip().split()
    if not parts:
        raise ValueError('Empty FEN section encountered')
    parts[0] = parts[0].rstrip('/')
    return ' '.join(parts)


def _initial_bughouse_fens(headers: dict[str, str]) -> tuple[str, str]:
    """Extract initial board FENs from headers or use default start positions."""
    if headers.get('SetUp') == '1' and headers.get('FEN'):
        fen_parts = headers['FEN'].split('|')
        if len(fen_parts) != 2:
            raise ValueError('Expected two FENs separated by "|" in PGN FEN tag')
        return _clean_single_fen(fen_parts[0]), _clean_single_fen(fen_parts[1])

    start = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1'
    return start, start


def _extract_headers_and_movetext(raw_pgn: str) -> tuple[dict[str, str], str]:
    """Parse PGN headers and return remaining move text."""
    headers: dict[str, str] = {}
    lines = raw_pgn.splitlines()
    move_start_idx = 0

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        tag_match = TAG_RE.match(stripped)
        if not tag_match:
            move_start_idx = idx
            break
        headers[tag_match.group(1)] = tag_match.group(2)
        move_start_idx = idx + 1

    move_text = ' '.join(line.strip() for line in lines[move_start_idx:])
    return headers, move_text


def _sanitize_move_token(token: str) -> str:
    """Normalize Chess.com bughouse SAN quirks into python-chess-parseable SAN."""
    token = token.strip()
    token = token.rstrip('.')

    # Chess.com encodes some knight drops as $146@square.
    knight_drop = re.match(r'^\$146@([a-h][1-8])([+#]?)$', token)
    if knight_drop:
        return f"N@{knight_drop.group(1)}{knight_drop.group(2)}"

    # Some exports omit the dropped piece symbol (e.g. "@e4").
    # Keep it as-is and resolve against legal moves later.
    if token.startswith('@'):
        return token

    # Safety fallback: if a malformed token like "6@e4" appears,
    # still interpret it as a pawn drop on e4.
    mangled_drop = re.match(r'^\d+@([a-h][1-8])([+#]?)$', token)
    if mangled_drop:
        return f"P@{mangled_drop.group(1)}{mangled_drop.group(2)}"

    # Strip any remaining NAG prefix like $1, $5, etc.
    token = re.sub(r'^\$\d+', '', token)
    return token


def _resolve_bare_drop_san(
    board: chess.variant.CrazyhouseBoard,
    san: str,
) -> str | None:
    """Resolve bare drop notation like "@e4" to an explicit pawn drop "P@e4"."""
    match = re.match(r'^@([a-h][1-8])([+#]?)$', san.strip())
    if not match:
        return None
    return f"P@{match.group(1)}{match.group(2)}"


def _recover_legal_move_from_san(
    board: chess.variant.CrazyhouseBoard,
    san: str,
) -> chess.Move | None:
    """Best-effort SAN recovery for slightly malformed/ambiguous tokens."""
    token = san.strip()
    if not token:
        return None

    # Normalize check/mate suffixes and common castling notation variants.
    core = token.rstrip('+#')
    core = core.replace('0-0-0', 'O-O-O').replace('0-0', 'O-O')

    target_match = re.search(r'([a-h][1-8])', core)
    target_sq = chess.parse_square(target_match.group(1)) if target_match else None

    piece_type = None
    if core and core[0] in 'KQRBN':
        piece_type = {
            'K': chess.KING,
            'Q': chess.QUEEN,
            'R': chess.ROOK,
            'B': chess.BISHOP,
            'N': chess.KNIGHT,
        }[core[0]]

    wants_capture = 'x' in core

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    candidates = legal_moves

    if target_sq is not None:
        candidates = [m for m in candidates if m.to_square == target_sq]
    if piece_type is not None:
        candidates = [m for m in candidates if board.piece_type_at(m.from_square) == piece_type]
    if wants_capture:
        capture_candidates = [m for m in candidates if board.is_capture(m)]
        if capture_candidates:
            candidates = capture_candidates

    if len(candidates) == 1:
        return candidates[0]

    # Final fallback: compare simplified SAN strings against legal SAN output.
    def simplify(s: str) -> str:
        return s.replace('+', '').replace('#', '').replace('x', '').strip()

    simple_core = simplify(core)
    matching = []
    for move in legal_moves:
        legal_san = board.san(move)
        if simplify(legal_san) == simple_core:
            matching.append(move)

    if len(matching) == 1:
        return matching[0]

    return None


def _parse_bughouse_moves(move_text: str) -> list[tuple[str, str]]:
    """Convert interleaved board-prefixed move text into ordered (board, san) tuples."""
    cleaned = re.sub(r'\{[^}]*\}', ' ', move_text)
    cleaned = re.sub(r'\([^)]*\)', ' ', cleaned)
    # Remove standalone NAG tokens like "$1" while preserving "$146@e4".
    # Token boundaries prevent regex backtracking from partially stripping "$146@e4".
    cleaned = re.sub(r'(?<!\S)\$\d+\b(?!@[a-h][1-8])', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    tokens = cleaned.split(' ') if cleaned else []
    moves: list[tuple[str, str]] = []
    pending_board = None

    for raw_token in tokens:
        token = raw_token.strip()
        if not token:
            continue

        normalized_result = token.rstrip('.')
        if normalized_result in RESULT_TOKENS:
            break

        prefix_match = MOVE_PREFIX_RE.match(token)
        if prefix_match:
            pending_board = prefix_match.group(1).upper()
            continue

        if pending_board is None:
            continue

        san = _sanitize_move_token(token)
        if not san:
            continue

        moves.append((pending_board, san))
        pending_board = None

    return moves


def _pocket_counts(board: chess.variant.CrazyhouseBoard) -> dict[str, dict[str, int]]:
    """Return pocket counts for both colors using piece letters expected by the UI."""
    piece_order = [
        (chess.PAWN, 'p'),
        (chess.KNIGHT, 'n'),
        (chess.BISHOP, 'b'),
        (chess.ROOK, 'r'),
        (chess.QUEEN, 'q'),
    ]
    return {
        'w': {letter: board.pockets[chess.WHITE].count(piece) for piece, letter in piece_order},
        'b': {letter: board.pockets[chess.BLACK].count(piece) for piece, letter in piece_order},
    }


def _apply_bughouse_move(
    source_board: chess.variant.CrazyhouseBoard,
    partner_board: chess.variant.CrazyhouseBoard,
    move: chess.Move,
) -> None:
    """Apply a move and transfer captured piece to partner board's opposite color pocket."""
    capturer_color = source_board.turn
    partner_color = not capturer_color

    before_src = {
        piece: source_board.pockets[capturer_color].count(piece)
        for piece in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    }

    is_capture = source_board.is_capture(move)
    source_board.push(move)

    if not is_capture:
        return

    gained_piece = None
    for piece in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        after_count = source_board.pockets[capturer_color].count(piece)
        if after_count > before_src[piece]:
            gained_piece = piece
            break

    if gained_piece is None:
        return

    # Undo Crazyhouse local-pocket capture and transfer to the bughouse partner.
    source_board.pockets[capturer_color].remove(gained_piece)
    partner_board.pockets[partner_color].add(gained_piece)


def _snapshot_timeline(
    timeline: list[dict],
    board_a: chess.variant.CrazyhouseBoard,
    board_b: chess.variant.CrazyhouseBoard,
    move_no: int,
    board_name: str,
    san: str,
) -> None:
    fen_a = board_a.fen()
    fen_b = board_b.fen()
    timeline.append({
        'index': len(timeline),
        'ply': move_no,
        'board': board_name,
        'san': san,
        'fenA': fen_a,
        'fenB': fen_b,
        'turnA': 'w' if board_a.turn == chess.WHITE else 'b',
        'turnB': 'w' if board_b.turn == chess.WHITE else 'b',
        'pocketsA': _pocket_counts(board_a),
        'pocketsB': _pocket_counts(board_b),
        'combinedFen': f'{fen_a} | {fen_b}',
    })


def parse_bughouse_pgn(raw_pgn: str) -> dict:
    """Parse Chess.com-style bughouse PGN and produce replay timeline snapshots."""
    headers, move_text = _extract_headers_and_movetext(raw_pgn)
    fen_a, fen_b = _initial_bughouse_fens(headers)

    board_a = chess.variant.CrazyhouseBoard(fen_a)
    board_b = chess.variant.CrazyhouseBoard(fen_b)
    warnings: list[str] = []

    timeline: list[dict] = []
    _snapshot_timeline(timeline, board_a, board_b, 0, '-', 'start')

    for move_no, (board_name, san) in enumerate(_parse_bughouse_moves(move_text), start=1):
        board = board_a if board_name == 'A' else board_b
        partner = board_b if board_name == 'A' else board_a
        san_for_timeline = san
        effective_board_name = board_name
        san_to_parse = _resolve_bare_drop_san(board, san) or san

        try:
            move = board.parse_san(san_to_parse)
            san_for_timeline = san_to_parse
        except Exception as exc:
            recovered = _recover_legal_move_from_san(board, san)
            if recovered is not None:
                san_for_timeline = board.san(recovered)
                warnings.append(
                    f"Recovered move {move_no} on board {board_name}: '{san}' -> '{san_for_timeline}'"
                )
                move = recovered
            else:
                # Some exports may contain mislabeled board prefixes; try the opposite board.
                alt_board_name = 'B' if board_name == 'A' else 'A'
                alt_board = board_b if board_name == 'A' else board_a
                alt_partner = board_a if board_name == 'A' else board_b
                alt_san_to_parse = _resolve_bare_drop_san(alt_board, san) or san

                try:
                    alt_move = alt_board.parse_san(alt_san_to_parse)
                    board = alt_board
                    partner = alt_partner
                    move = alt_move
                    effective_board_name = alt_board_name
                    san_for_timeline = alt_san_to_parse
                    warnings.append(
                        f"Recovered move {move_no} by board swap: '{board_name}:{san}' parsed on board {alt_board_name}"
                    )
                except Exception:
                    alt_recovered = _recover_legal_move_from_san(alt_board, san)
                    if alt_recovered is not None:
                        board = alt_board
                        partner = alt_partner
                        move = alt_recovered
                        effective_board_name = alt_board_name
                        san_for_timeline = board.san(alt_recovered)
                        warnings.append(
                            f"Recovered move {move_no} by board swap + SAN recovery: '{board_name}:{san}' -> '{effective_board_name}:{san_for_timeline}'"
                        )
                    else:
                        warnings.append(
                            f"Stopped at move {move_no} ({board_name}: {san}): {exc}"
                        )
                        break

        _apply_bughouse_move(board, partner, move)
        _snapshot_timeline(timeline, board_a, board_b, move_no, effective_board_name, san_for_timeline)

    return {
        'headers': headers,
        'timeline': timeline,
        'moveCount': max(0, len(timeline) - 1),
        'initialCombinedFen': timeline[0]['combinedFen'],
        'finalCombinedFen': timeline[-1]['combinedFen'],
        'warnings': warnings,
    }


@app.route('/')
def index():
    """Serve the main GUI page."""
    return send_file('index.html')


@app.route('/viewer')
def viewer():
    """Serve the bughouse PGN viewer page."""
    return send_file('bughouse_viewer.html')


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
    return jsonify({
        'status': 'ok',
        'timestamp': time.time(),
        'parserVersion': PARSER_VERSION,
    })


@app.route('/api/parse-bughouse', methods=['POST'])
def api_parse_bughouse():
    """API endpoint to parse bughouse PGN into a replay timeline."""
    payload = request.json or {}
    pgn = payload.get('pgn', '')

    if not isinstance(pgn, str) or not pgn.strip():
        return jsonify({'error': 'Missing or empty PGN text'}), 400

    try:
        parsed = parse_bughouse_pgn(pgn)
        parsed['parserVersion'] = PARSER_VERSION
        return jsonify(parsed)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400


def main():
    parser = argparse.ArgumentParser(description='Hivemind Bughouse GUI Server')
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
