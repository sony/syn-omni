#!/usr/bin/env bash

# ==============================================================================
# Configuration
# ==============================================================================
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
BATCH_SIZE=96
MODEL_NAME="Tevatron/Qwen2.5-Omni-7B-Thinker"
TOKENIZER_NAME="Qwen/Qwen2.5-Omni-7B"
DATA_CONFIG_PATH="configs/syn_omni.yaml"
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

OUTPUT_PATH="$OUTPUT_BASEDIR/uni-omni-7b"
mkdir -p "$OUTPUT_PATH"

export FORCE_QWENVL_VIDEO_READER=decord

# ==============================================================================
# Main Execution
# ==============================================================================
cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --nproc_per_node=8 --master_port=$((10000 + RANDOM % 50000)) \
  -m tevatron.retriever.driver.train_mm \
  --output_dir \"$OUTPUT_PATH\" \
  --save_steps 50 --save_total_limit 10 \
  --model_name_or_path \"$MODEL_NAME\" \
  --tokenizer_name \"$TOKENIZER_NAME\" \
  --lora --lora_r 16 --lora_alpha 32 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --train_yaml \"$DATA_CONFIG_PATH\" \
  --data_root \"$DATA_ROOT\" \
  --bf16 --tf32 True \
  --pooling last --append_eos_token \
  --normalize --temperature 0.02 \
  --per_device_train_batch_size $BATCH_SIZE \
  --dataloader_drop_last True \
  --max_text_len 512 \
  --apply_native_resolution True \
  --num_train_epochs 1 \
  --logging_steps 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 --lr_scheduler_type cosine --warmup_ratio 0.04 \
  --max_grad_norm 1.0 --weight_decay 0.01 \
  --dataloader_num_workers 4 \
  --adam_beta1 0.9 --adam_beta2 0.999 --adam_epsilon 1e-8 \
  --grad_cache True --gc_q_chunk_size 8 --gc_p_chunk_size 8 \
  --report_to none"

echo "$cmd"
eval "$cmd"
