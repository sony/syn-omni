#!/usr/bin/env bash
set -euo pipefail

echo "==> Environment"
echo "conda location: $(which conda || true)"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
echo ""


# ------------------------------------------------------------------------------
# WAVE-only evaluation script.
# This script is isolated from eval_1gpu.sh to avoid impacting other model runs.
# ------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
BATCH_SIZE=1
MODALITIES=("image" "video" "audio" "audiovisual")
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

WAVE_MODEL_PATH=/home/appuser/omni/hf_checkpoints/tsinghua-ee__WAVE-7B
WAVE_PROCESSOR_PATH=/home/appuser/omni/hf_checkpoints/tsinghua-ee__WAVE-7B
WAVE_BEATS_PATH="${WAVE_BEATS_PATH:-}"
WAVE_BEATS_ONLY="${WAVE_BEATS_ONLY:-false}"

BASE_OUTPUT_PATH="$OUTPUT_BASEDIR/WAVE-7B"
mkdir -p "$BASE_OUTPUT_PATH"

if [[ -n "$WAVE_BEATS_PATH" ]]; then
  WAVE_BEATS_ARGS="--wave_use_beats true --wave_beats_path \"$WAVE_BEATS_PATH\" --wave_beats_only \"$WAVE_BEATS_ONLY\""
  echo "Using BEATs checkpoint: $WAVE_BEATS_PATH"
else
  WAVE_BEATS_ARGS="--wave_use_beats false"
  echo "WARNING: WAVE_BEATS_PATH is empty; this is NOT strict WAVE README reproduction."
fi

for MODALITY in "${MODALITIES[@]}"; do
  DATA_CONFIG_PATH="experiments/public/eval/$MODALITY.yaml"
  OUTPUT_PATH="$BASE_OUTPUT_PATH/$MODALITY/"
  mkdir -p "$OUTPUT_PATH"

  cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --nproc_per_node=8 --master_port=2278 --max_restarts=0 eval_wave.py \
    --lora False \
    --pooling mean \
    --normalize true \
    --per_device_eval_batch_size $BATCH_SIZE \
    --model_backbone wave \
    --model_name \"$WAVE_MODEL_PATH\" \
    --processor_name \"$WAVE_PROCESSOR_PATH\" \
    --apply_native_resolution true \
    --wave_train_classify true \
    --wave_classify_type all_layer \
    --wave_pred_embeds true \
    $WAVE_BEATS_ARGS \
    --dataset_config \"$DATA_CONFIG_PATH\" \
    --encode_output_path \"$OUTPUT_PATH\" \
    --data_basedir \"$DATA_BASEDIR\""

  echo "-------------------------------------------------"
  echo "Modality: $MODALITY"
  echo "Output: $OUTPUT_PATH"
  eval "$cmd"
done

echo "WAVE-only evaluation completed."
