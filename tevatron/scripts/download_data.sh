#!/usr/bin/env bash

# ==============================================================================
# Configuration
# ==============================================================================
DATA_CONFIG_PATH="configs/syn_omni.yaml"
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

# ==============================================================================
# Main Execution
# ==============================================================================
cmd="python -m tevatron.retriever.driver.load_data_mm \
  --data_root \"$DATA_ROOT\" \
  --train_yaml \"$DATA_CONFIG_PATH\" \
  --dataset_download True"

echo "$cmd"
eval "$cmd"
