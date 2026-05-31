#!/bin/bash

WEIGHTS_EXIST=false

if [ -d "./weights" ]; then
    echo "Found ./weights directory. Moving to /tmp/weights to prevent deletion..."
    rm -rf /tmp/weights
    mv ./weights /tmp/weights
    WEIGHTS_EXIST=true
fi

# Run the git commands
echo "Running git update commands..."
git reset --hard && git clean -fdx && git pull --rebase
GIT_STATUS=$?

if [ "$WEIGHTS_EXIST" = true ]; then
    echo "Restoring weights from /tmp/weights to ./weights..."
    mv /tmp/weights ./weights
fi

exit $GIT_STATUS