#!/bin/bash

# Exit on error
set -e

if [ -z "$1" ] || { [ "$1" != "wan22_i2v" ] && [ "$1" != "wan22_t2v" ] && [ "$1" != "z_image_turbo" ] && [ "$1" != "qwen_image" ] && [ "$1" != "qwen_image_edit" ]; }; then
    echo "Usage:"
    echo "  ./scripts/deploy_cog_inference.sh model"
    echo " model: One of wan22_i2v, wan22_t2v, z_image_turbo, qwen_image, or qwen_image_edit"
    exit 1
fi

ensure_command() {
    local cmd="$1"
    local install_cmd="$2"

    if ! command -v "$cmd" &> /dev/null; then
        echo "$cmd not found. Installing..."
        eval "$install_cmd" || {
            echo "Failed to install $cmd."
            exit 1
        }
        echo "$cmd installed successfully."
    else
        echo "$cmd is already installed."
    fi
}

OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m | sed 's/x86_64/amd64/' | sed 's/aarch64/arm64/' | sed 's/arm.*/arm/')

# Ensure the required commands are available
ensure_command "cog" "sudo curl -L https://github.com/replicate/cog/releases/latest/download/cog_${OS}_${ARCH} -o /usr/local/bin/cog && sudo chmod +x /usr/local/bin/cog"
ensure_command "yq" "sudo curl -L https://github.com/mikefarah/yq/releases/latest/download/yq_${OS}_${ARCH} -o /usr/local/bin/yq && sudo chmod +x /usr/local/bin/yq"

# Merge cog.yaml templates to get the final yaml config
yq eval-all 'select(fileIndex == 0) *+ select(fileIndex == 1)' ./apps/cog_apps/cog.template.yaml "./apps/cog_apps/$1/cog.yaml" > ./cog.yaml
mv ./apps/cog_apps/$1/app.py app.py

rm -rf src notebooks apps # We remove the local turbogen code, the pip installed version should be used

# Extract the registry image name from the generated cog.yaml
IMAGE_NAME=$(yq eval '.image' ./cog.yaml)
if [ -z "$IMAGE_NAME" ] || [ "$IMAGE_NAME" = "null" ]; then
    echo "Error: The final cog.yaml does not define an 'image:' field."
    exit 1
fi

echo "Target Replicate registry image: $IMAGE_NAME"

echo "Building local image via Cog..."
cog build -t "$IMAGE_NAME"

# Setup cleanup trap to stop background container on script exit/error
CONTAINER_NAME="cog_warmup_$(date +%s)"
cleanup() {
    if docker ps -a --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}$"; then
        echo "Stopping and cleaning up temporary container..."
        docker stop "$CONTAINER_NAME" &>/dev/null || true
        docker rm "$CONTAINER_NAME" &>/dev/null || true
    fi
}
trap cleanup EXIT

echo "Starting container..."
docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus all \
  -p 5000:5000 \
  # By default
  -e HF_HUB_OFFLINE=0 \
  "$IMAGE_NAME"

echo "Running warmup code..."
python scripts/warmup_cog_app.py --model "$1" --port 5000 --timeout 300

echo "Stopping container..."
docker stop "$CONTAINER_NAME"

echo "Committing runtime modifications (all cached JIT kernels) to image..."
docker commit "$CONTAINER_NAME" "$IMAGE_NAME"

# Clean up container explicitly to avoid triggering the trap
docker rm "$CONTAINER_NAME"
trap - EXIT

cog login
echo "Pushing pre-warmed image to Replicate..."
docker push "$IMAGE_NAME"

echo "✅ Deploy finished successfully 🚀"