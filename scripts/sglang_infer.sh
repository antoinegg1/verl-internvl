model_path=${1:?provide model_path}
echo "model: $model_path"

# nohup python -m sglang.launch_server \
#   --model-path "$model_path" \
#   --host 0.0.0.0 --port 30000 \
#   --nnodes 1 --node-rank 0 \
#   --dp-size 8 \
#   --tp-size 1 \
#   --dtype auto \
#   --mem-fraction-static 0.8 \
#   > /var/log/sglang_node0.log 2>&1 &

# echo "waiting endpoint..."

# while ! curl -s http://127.0.0.1:30000 > /dev/null; do
#     sleep 2
# done
# echo "endpoint ready"

export BASE_IMAGE_DIR="/storage/openpsi/data/" 

model_name="$(basename "$model_path")"
if [[ "$model_path" =~ (^|[^0-9])8[bB]([^0-9]|$) ]]; then
    result_dir="/storage/openpsi/data/grounding_sft_v1_result/${model_name}_8B"
else
    result_dir="/storage/openpsi/data/grounding_sft_v1_result/${model_name}"
fi
if [[ "$model_path" == *"amodal"* ]]; then
    # result_dir="${result_dir}_amodal"
    result_dir="/storage/openpsi/data/amodal_result/${model_name}"
fi
echo "Result dir: $result_dir"
# mkdir -p "$result_dir" 
# data_name_list=("refcoco_testA" "refcoco_testB" "refcoco+_testA" "refcoco+_testB" "refcoco+_val" "refcocog_val" "refcocog_test" "refcoco_val")
# "refcoco_testA" "refcoco_testB" "refcoco+_testA" "refcoco+_testB"
# data_name_list=( "stage2_thinking_with_text_sft_train_10.23_241b_iou_added_withflip_part1" "stage2_thinking_with_text_sft_train_10.23_241b_iou_added_withflip_part2" "stage2_thinking_with_text_sft_train_10.23_241b_iou_added_withflip_part3" "stage2_thinking_with_text_sft_train_10.23_241b_iou_added_withflip_part4")
# "refcoco_testA" 
data_name_list=( "refcoco_train_v4_gpt_10.26_withflip_2_RL" )
# data_name_list=("refcoco_train_v5_gpt_candidate_10.28_withflip_RL" )
data_dir=grounding_sft_v1
# data_dir=grounding_cot_v3_train_rl

for data_name in "${data_name_list[@]}"; do
    echo "Processing dataset: $data_name"
    data_json="/storage/openpsi/data/${data_dir}/${data_name}.jsonl"
    output_dir="${result_dir}/${data_name}_result"
    mkdir -p "$output_dir"
    python scripts/sglang_infer.py \
    --model_path $model_path \
    --data_json $data_json \
    --output_dir $output_dir \
    --endpoint "http://127.0.0.1:30000" 

done
pkill -9 "sglang" -f
# nohup python -m sglang.launch_server \
#   --model-path  /storage/openpsi/models/InternVL3_5-241B-A28B  \
#   --host 0.0.0.0 --port 30000 \
#   --nnodes 2 --node-rank 1 \
#   --dist-init-addr 33.180.160.150:50000 \
#   --tp-size 16 \
#   --dtype auto \
#   --mem-fraction-static 0.7 \
#   > /var/log/sglang_node1.log 2>&1 &

# nohup python -m sglang.launch_server \
#   --model-path  /storage/openpsi/models/Qwen2.5-VL-72B-Instruct/  \
#   --host 0.0.0.0 --port 30000 \
#   --nnodes 1 --node-rank 0 \
#   --dp-size 1 \
#   --tp-size 8 \
#   --dtype auto \
#   --mem-fraction-static 0.8 \
#   > /var/log/sglang_node0.log 2>&1 & 