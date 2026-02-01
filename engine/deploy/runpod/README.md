# Hivemind RunPod Serverless Deployment

Deploy the Hivemind bughouse engine as a serverless GPU endpoint on RunPod.

## Prerequisites

- Docker installed locally
- Docker Hub account (or other container registry)
- RunPod account with API key

## Quick Start

### 1. Build and Push Docker Image

```bash
# Set your Docker username
export DOCKER_USERNAME=yourusername

# Build and tag the image
./build_and_push.sh latest

# Push to Docker Hub
docker login
docker push $DOCKER_USERNAME/hivemind-engine:latest
```

### 2. Upload Model to RunPod

Upload your ONNX model to RunPod's network storage or a public URL:

- Go to RunPod Console → Network Volumes
- Create a volume and upload your `.onnx` model file
- Or host on S3/GCS with a public URL

### 3. Create Serverless Endpoint

1. Go to [RunPod Serverless Console](https://www.runpod.io/console/serverless)
2. Click "New Endpoint"
3. Configure:
   - **Container Image**: `yourusername/hivemind-engine:latest`
   - **GPU Type**: RTX 4090 (24GB) recommended
   - **Min Workers**: 0 (scale to zero)
   - **Max Workers**: 5 (adjust based on traffic)
   - **Idle Timeout**: 30 seconds
   - **Environment Variables**:
     - `MODEL_URL`: URL to download ONNX model (optional)
   - **Network Volume**: Mount your volume at `/app/networks`

4. Click "Deploy"

### 4. Test the Endpoint

```bash
# Get your endpoint ID and API key from RunPod console
ENDPOINT_ID="your-endpoint-id"
RUNPOD_API_KEY="your-api-key"

# Test the endpoint
curl -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "action": "move",
      "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1|rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "nodes": 800
    }
  }'
```

## API Reference

### Request Format

```json
{
  "input": {
    "action": "move",
    "fen": "...|...",
    "moves": ["e2e4", "e7e5"],
    "nodes": 800,
    "movetime": 1000,
    "options": {
      "Hash": 128,
      "MultiPV": 1
    }
  }
}
```

### Actions

| Action    | Description                     |
| --------- | ------------------------------- |
| `move`    | Get best move for position      |
| `analyze` | Analyze position (same as move) |
| `newgame` | Reset engine state              |
| `stop`    | Stop current search             |

### Parameters

| Parameter  | Type   | Default  | Description                                   |
| ---------- | ------ | -------- | --------------------------------------------- |
| `fen`      | string | required | Bughouse FEN (both boards separated by `\|`)  |
| `moves`    | array  | []       | Moves from starting position                  |
| `nodes`    | int    | 800      | MCTS nodes to search                          |
| `movetime` | int    | 0        | Search time in milliseconds (overrides nodes) |
| `options`  | object | {}       | UCI options to set                            |

### Response Format

```json
{
  "bestmove": "(e2e4,d2d4)",
  "ponder": "(e7e5,d7d5)",
  "info": ["info depth 10 nodes 800 ..."],
  "eval": 0.15,
  "nodes": 800,
  "time": 234
}
```

## Cost Optimization

- **Scale to Zero**: Set min workers to 0 for lowest cost
- **Idle Timeout**: Set to 30-60 seconds to keep warm between moves
- **GPU Selection**: RTX 4090 offers best price/performance for inference
- **Batch Size**: Engine uses batch size 8 by default

## Troubleshooting

### Cold Start Too Slow

- Increase min workers to 1 for faster response
- Use smaller model or lower precision (FP16)

### Out of Memory

- Reduce batch size with `--batch` flag
- Use smaller GPU or reduce model size

### Model Not Found

- Check network volume is mounted at `/app/networks`
- Verify ONNX file exists in the volume
