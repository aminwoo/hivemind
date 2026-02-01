# Hivemind Vercel Deployment

Deploy the Hivemind bughouse GUI to Vercel with serverless API routes.

## Prerequisites

- [Vercel CLI](https://vercel.com/cli) installed (`npm i -g vercel`)
- Vercel account
- RunPod endpoint already deployed (see `../runpod/README.md`)

## Quick Start

### 1. Install Vercel CLI

```bash
npm install -g vercel
```

### 2. Login to Vercel

```bash
vercel login
```

### 3. Set Environment Variables

Set your RunPod credentials as Vercel secrets:

```bash
# Add secrets (you'll be prompted for values)
vercel secrets add runpod-endpoint-id
vercel secrets add runpod-api-key
```

### 4. Deploy

```bash
cd deploy/vercel
vercel
```

For production:

```bash
vercel --prod
```

## Project Structure

```
vercel/
├── api/
│   ├── move.py      # Proxy to RunPod for move requests
│   └── health.py    # Health check endpoint
├── public/
│   └── index.html   # Bughouse GUI
├── vercel.json      # Vercel configuration
└── README.md
```

## API Endpoints

### POST /api/move

Request engine move for a position.

**Request:**

```json
{
  "fen": "rnbqkbnr/...|rnbqkbnr/...",
  "nodes": 800,
  "movetime": 1000
}
```

**Response:**

```json
{
  "bestmove": "(e2e4,d2d4)",
  "eval": 0.15,
  "nodes": 800,
  "time": 234
}
```

### GET /api/health

Check endpoint health and worker status.

**Response:**

```json
{
  "status": "ok",
  "workers": {
    "ready": 1,
    "running": 0,
    "idle": 1
  }
}
```

## Environment Variables

| Variable             | Description                        |
| -------------------- | ---------------------------------- |
| `RUNPOD_ENDPOINT_ID` | Your RunPod serverless endpoint ID |
| `RUNPOD_API_KEY`     | Your RunPod API key                |

## Custom Domain

To use a custom domain:

1. Go to Vercel Dashboard → Your Project → Settings → Domains
2. Add your domain (e.g., `bughouse.yourdomain.com`)
3. Configure DNS as instructed

## Development

Run locally with Vercel CLI:

```bash
vercel dev
```

This will start a local server at `http://localhost:3000`.

## Troubleshooting

### "Endpoint unavailable" error

- Check RunPod endpoint is deployed and running
- Verify environment variables are set correctly
- Check RunPod console for worker status

### Slow first move

- RunPod scales from zero, first request may take 30-60 seconds
- Consider keeping min workers at 1 for faster response

### CORS errors

- API routes include CORS headers
- Ensure you're using `/api/` routes, not direct RunPod calls
