#!/usr/bin/env bash
set -euo pipefail
# export HF_ENDPOINT=https://hf-mirror.com
model_path="${1:?provide model}"
echo "model: $model_path"


HF_ENDPOINT=https://hf-mirror.com nohup python -m vllm.entrypoints.openai.api_server \
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

export BASE_IMAGE_DIR="${BASE_IMAGE_DIR:-/storage/openpsi/data}"

prompt_template=qwen3
box_remap=keep
model_name="$(basename "$model_path")"

if [[ "$model_path" == *"amodal"* ]]; then
  root="/storage/openpsi/data/amodal_result"
elif [[ "$model_path" == *"qwen"* ]]; then
  root="/storage/openpsi/data/qwen_result"
else
  root="/storage/openpsi/data/grounding_sft_v1_result"
fi

suffix=""
if [[ "$model_path" =~ (^|[^0-9])([0-9]+)[bB]([^0-9]|$) ]]; then
  case "${BASH_REMATCH[2]}" in
    2|4|8) suffix="_${BASH_REMATCH[2]}B" ;;
  esac
fi
result_dir="${root}/${model_name}${suffix}_throughout"
echo "Result dir: $result_dir"
data_name_list=("refcoco_test_sample")
# data_name_list=("refcoco_testA")
# data_name_list=("8b_complex_prompt_failure_cases_aligned")
# data_name_list=("refcoco_testA" "refcoco_testB" "refcoco+_testA" "refcoco+_testB" "refcoco+_val" "refcocog_val" "refcocog_test" "refcoco_val")
data_dir=grounding_sft_v1

for data_name in "${data_name_list[@]}"; do
    echo "Processing dataset: $data_name"
    data_json="/storage/openpsi/data/${data_dir}/${data_name}.jsonl"
    output_dir="${result_dir}/${data_name}_result"
    mkdir -p "$output_dir"
    echo $prompt_template
    echo $box_remap
    python scripts/vllm_infer.py \
        --model "$model_path" \
        --data_json "$data_json" \
        --output_dir "$output_dir" \
        --endpoint "http://127.0.0.1:8000" \
        --max_tokens 4096 \
        --concurrency 64 \
        --prompt_template $prompt_template \
        --box_remap $box_remap

done

pkill -9 "VLLM" -f


#  HF_ENDPOINT=https://hf-mirror.com nohup python -m vllm.entrypoints.openai.api_server \
#   --model /storage/openpsi/models/Qwen3-VL-235B-A22B-Instruct \
#   --host 0.0.0.0 \
#   --port 8000 \
#   --trust-remote-code \
#   --data-parallel-size 1 \
#   --tensor-parallel-size 8 \
#   --dtype auto \
#   --gpu-memory-utilization 0.8 \
#   > /var/log/vllm_api.log 2>&1 &