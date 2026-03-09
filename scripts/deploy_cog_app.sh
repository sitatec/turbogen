#!/bin/bash

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

# Ensure the required commands are available
ensure_command "cog" "sudo curl -L https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m) -o /usr/local/bin/cog && sudo chmod +x /usr/local/bin/cog" && \
ensure_command "yq" "sudo curl -L https://github.com/mikefarah/yq/releases/latest/download/yq_$(uname -s)_$(uname -m) -o /usr/local/bin/yq && sudo chmod +x /usr/local/bin/yq"

yq eval-all 'select(fileIndex == 0) *+ select(fileIndex == 1)' ./apps/cog_apps/cog.template.yaml "./apps/cog_apps/$1/cog.yaml" > ./cog.yaml
mv ./apps/cog_apps/$1/app.py app.py

rm -rf src notebooks apps # We remove the local turbogen code, the pip installed version should be used

cog run script/warmup_cog_app

cog login

cog push