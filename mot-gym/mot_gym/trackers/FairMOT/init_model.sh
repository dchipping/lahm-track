#!/bin/bash

SCRIPT_PATH=$(realpath $0)
SCRIPT_DIR=$(dirname $SCRIPT_PATH)

# cd into DCNv2 folder
cd "$SCRIPT_DIR/models/networks/DCNv2"

# Run build
./make.sh