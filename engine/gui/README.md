# Hivemind Bughouse GUI

A web-based graphical interface for running and viewing bughouse model evaluations in real-time.

## Features

- **Interactive Control Panel**: Set all evaluation parameters directly in the browser
- **One-Click Start/Stop**: Launch evaluations without command-line
- **Real-time Visualization**: Both boards displayed side by side
- **Piece Pockets**: Shows captured pieces available for drops
- **Move History**: Color-coded by board with latest move highlighting
- **Tournament Statistics**: Wins, losses, draws, win rate, Elo estimate
- **Auto-Model Discovery**: Automatically finds ONNX models in the networks folder
- **Two Evaluation Modes**:
  - Model vs Model: Compare two different neural networks
  - Parameter Test: Same model with different search parameters

## Quick Start

### 1. Install Python dependencies

```bash
cd gui
pip install -r requirements.txt
```

### 2. Start the GUI server

```bash
python server.py --port 8080
```

Or use the convenience script:

```bash
./start_gui.sh
```

### 3. Open your browser

Navigate to `http://localhost:8080`

### 4. Configure and Play

1. Select evaluation mode (Model vs Model or Parameter Test)
2. Choose models from the dropdown (auto-detected from `networks/` folder)
3. Set search parameters (time per move, nodes, temperature, etc.)
4. Click **▶ Start Evaluation**
5. Watch the games in real-time!

## Evaluation Modes

### Model vs Model

Compare two different neural network models:

- **New Model (Challenger)**: The model you want to test
- **Old Model (Baseline)**: The reference model

### Parameter Test

Test the same model with different search parameters:

- Configure Player 1 and Player 2 separately
- Compare different CPUCT values, node counts, etc.
- Useful for hyperparameter tuning

## Command Line Options

### GUI Server (`server.py`)

| Option         | Default           | Description                      |
| -------------- | ----------------- | -------------------------------- |
| `--port`       | 8080              | Port to run the server on        |
| `--host`       | 0.0.0.0           | Host to bind to                  |
| `--engine`     | ../build/hivemind | Path to the engine executable    |
| `--networks`   | ../networks       | Path to the networks directory   |
| `--state-file` | ./game_state.json | Path to the game state JSON file |

## How It Works

1. The Flask server provides a web interface with controls
2. When you click "Start", the server spawns the engine process
3. The engine writes game state to a JSON file after each move
4. The browser polls the server every 200ms for updates
5. Click "Stop" to terminate the engine

## Architecture

```
┌─────────────────┐                    ┌─────────────────┐
│  Browser        │◄──── HTTP ────────►│  Flask Server   │
│  (index.html)   │                    │  (server.py)    │
└─────────────────┘                    └────────┬────────┘
                                                │ spawns
                                                ▼
                                       ┌─────────────────┐
                                       │  C++ Engine     │
                                       │  (hivemind)     │
                                       └────────┬────────┘
                                                │ writes
                                                ▼
                                       ┌─────────────────┐
                                       │ game_state.json │
                                       └─────────────────┘
```

## Troubleshooting

### "No models found" in dropdowns

- Make sure ONNX models are in the `networks/` directory
- Check that the `--networks` path is correct

### Engine fails to start

- Verify the engine is built: `cd ../build && ninja`
- Check the `--engine` path points to the correct executable
- Look at the terminal for error messages from the engine

### Boards not updating

- Check browser console for errors
- Verify the engine is actually running (check terminal output)
- Try clicking the Refresh button

### Slow performance

- The GUI polls at 5 FPS (200ms interval)
- Reduce number of games for quicker turnaround
- Use time-based search instead of node-based for more predictable game length
