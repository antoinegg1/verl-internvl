#!/usr/bin/env bash
set -euo pipefail
model_path="${your_model_path}"
export BASE_IMAGE_DIR="${your_base_image_dir}"
data_json="${your_data_json}"
output_dir="${your_output_dir}"
prompt_template=qwen3
box_remap=keep

echo "model: $model_path"


nohup python -m vllm.entrypoints.openai.api_server \
  --model "$model_path" \
  --host 0.0.0.0 \
  --trust-remote-code \
  --port 8000 \
  --data-parallel-size 1 \
  --tensor-parallel-size 8 \
  --dtype auto \
  --gpu-memory-utilization 0.9 \
  > /var/log/vllm_api.log 2>&1 &


echo "waiting endpoint..."
until curl -sf http://127.0.0.1:8000/v1/models > /dev/null; do
  tail -n 10  /var/log/vllm_api.log
  sleep 2
done
echo "endpoint ready"

python scripts/vllm_infer.py \
    --model "$model_path" \
    --data_json "$data_json" \
    --output_dir "$output_dir" \
    --endpoint "http://127.0.0.1:8000" \
    --max_tokens 4096 \
    --concurrency 64 \
    --prompt_template $prompt_template \
    --box_remap $box_remap


pkill -9 "VLLM" -f