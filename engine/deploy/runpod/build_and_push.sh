#!/bin/bash
# Build and push the Hivemind engine Docker image to Docker Hub
# Usage: ./build_and_push.sh [tag]

set -e

# Configuration
DOCKER_USERNAME="aminwoo"
IMAGE_NAME="hivemind-engine"
TAG="${1:-latest}"
FULL_IMAGE="${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"

echo "Building Docker image: ${FULL_IMAGE}"

# Navigate to engine root
cd "$(dirname "$0")/../.."

# Build the image
docker build -f deploy/runpod/Dockerfile -t "${FULL_IMAGE}" .

echo "Image built successfully!"
echo ""
echo "To push to Docker Hub:"
echo "  docker login"
echo "  docker push ${FULL_IMAGE}"
echo ""
echo "To test locally:"
echo "  docker run --gpus all -it ${FULL_IMAGE}"
echo ""
echo "To deploy to RunPod:"
echo "  1. Go to https://www.runpod.io/console/serverless"
echo "  2. Create a new Serverless Endpoint"
echo "  3. Use Docker image: ${FULL_IMAGE}"
echo "  4. Select GPU type (RTX 4090 recommended)"
echo "  5. Set min/max workers as needed"
