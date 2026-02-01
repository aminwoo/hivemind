# Hivemind Deployment Guide

Deploy the Hivemind bughouse chess engine serverlessly.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Browser                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Vercel (Frontend + API)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ index.html  │  │ /api/move   │  │ /api/health             │  │
│  │ (GUI)       │  │ (proxy)     │  │ (status)                │  │
│  └─────────────┘  └──────┬──────┘  └─────────────────────────┘  │
└──────────────────────────┼──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  RunPod Serverless (GPU)                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Docker Container                                            ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐ ││
│  │  │ handler.py   │→ │ hivemind     │→ │ TensorRT Engine   │ ││
│  │  │ (RunPod SDK) │  │ (C++ Engine) │  │ (Neural Network)  │ ││
│  │  └──────────────┘  └──────────────┘  └───────────────────┘ ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Deployment Steps

### Step 1: Deploy to RunPod

1. **Build Docker image**:

   ```bash
   cd deploy/runpod
   chmod +x build_and_push.sh
   ./build_and_push.sh latest
   ```

2. **Push to Docker Hub**:

   ```bash
   docker login
   docker push yourusername/hivemind-engine:latest
   ```

3. **Upload model to RunPod Network Volume**:
   - Go to RunPod Console → Network Volumes
   - Create volume, upload your `.onnx` model

4. **Create Serverless Endpoint**:
   - Go to RunPod Console → Serverless
   - Create endpoint with your Docker image
   - Mount network volume at `/app/networks`
   - Note your Endpoint ID

5. **Get API Key**:
   - Go to Settings → API Keys
   - Create or copy your API key

### Step 2: Deploy to Vercel

1. **Install Vercel CLI**:

   ```bash
   npm install -g vercel
   ```

2. **Set secrets**:

   ```bash
   cd deploy/vercel
   vercel secrets add runpod-endpoint-id "your-endpoint-id"
   vercel secrets add runpod-api-key "your-api-key"
   ```

3. **Deploy**:

   ```bash
   vercel --prod
   ```

4. **Access your site**:
   - Vercel will provide a URL like `https://hivemind-xxx.vercel.app`

## Configuration

### RunPod Endpoint Settings

| Setting           | Recommended Value |
| ----------------- | ----------------- |
| GPU Type          | RTX 4090 (24GB)   |
| Min Workers       | 0 (scale to zero) |
| Max Workers       | 3-5               |
| Idle Timeout      | 30 seconds        |
| Execution Timeout | 60 seconds        |

### Cost Estimation

| Component       | Cost                        |
| --------------- | --------------------------- |
| Vercel          | Free tier (100GB bandwidth) |
| RunPod (idle)   | $0 (scale to zero)          |
| RunPod (active) | ~$0.44/hr (RTX 4090)        |

**Per-move cost**: ~$0.0001-0.001 depending on search depth

## Testing

### Test RunPod Endpoint

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "action": "move",
      "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1|rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "nodes": 100
    }
  }'
```

### Test Vercel API

```bash
curl -X POST "https://your-site.vercel.app/api/move" \
  -H "Content-Type: application/json" \
  -d '{"fen": "...|...", "nodes": 100}'
```

## Troubleshooting

### Cold Start Latency

- First request after idle may take 30-60 seconds
- Keep min workers at 1 for lower latency (adds cost)

### Out of Memory

- Reduce batch size in engine
- Use smaller model

### Model Not Loading

- Check network volume is mounted correctly
- Verify ONNX file exists and is valid

## Files

```
deploy/
├── runpod/
│   ├── Dockerfile          # Container with engine + TensorRT
│   ├── handler.py          # RunPod serverless handler
│   ├── build_and_push.sh   # Build script
│   └── README.md
└── vercel/
    ├── api/
    │   ├── move.py         # Move request proxy
    │   └── health.py       # Health check
    ├── public/
    │   └── index.html      # GUI
    ├── vercel.json         # Vercel config
    ├── requirements.txt
    └── README.md
```
