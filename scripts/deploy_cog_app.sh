#!/bin/bash

if [ -z "$1" ] || { [ "$1" != "qwen" ] && [ "$1" != "wan" ] && [ "$1" != "zimage" ] && [ "$1" != "qwen_img" ] && [ "$1" != "qwen_img_edit" ]; }; then
    echo "Usage:"
    echo "  ./scripts/deploy_cog_inference.sh model"
    echo " model: One of qwen (for both image and edit) or wan or zimage or qwen_img or qwen_img_edit"
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

if [ "$1" == "qwen" ]; then
    cog_app_dir="./apps/qwen_image_and_edit"
elif [ "$1" == "wan" ]; then
    cog_app_dir="./apps/wan22_a14b"
elif [ "$1" == "zimage" ]; then
    cog_app_dir="./apps/zimage"
elif [ "$1" == "qwen_img" ]; then
    cog_app_dir="./apps/qwen_image"
elif [ "$1" == "qwen_img_edit" ]; then
    cog_app_dir="./apps/qwen_image_edit"
fi

# Ensure the required commands are available
ensure_command "cog" "sudo curl -L https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m) -o /usr/local/bin/cog && sudo chmod +x /usr/local/bin/cog" && \
ensure_command "yq" "sudo curl -L https://github.com/mikefarah/yq/releases/latest/download/yq_$(uname -s)_$(uname -m) -o /usr/local/bin/yq && sudo chmod +x /usr/local/bin/yq"

yq eval-all 'select(fileIndex == 0) *+ select(fileIndex == 1)' ./apps/cog_apps/cog.template.yaml "./apps/cog_apps/$1/cog.yaml" > ./cog.yaml
mv ./apps/cog_apps/$1/app.py app.py

cog push