export BASE_IMAGE_DIR="/storage/openpsi/data/" 
model_path=${1:-"/storage/openpsi/models/InternVL3_8B_Grounding_CoT_Text_SFT_20251016"}

echo "Using model path: $model_path"
model_name="$(basename "$model_path")"
if [[ "$model_path" =~ (^|[^0-9])8[bB]([^0-9]|$) ]]; then
    result_dir="/storage/openpsi/data/grounding_sft_v1_result/${model_name}_8B"
else
    result_dir="/storage/openpsi/data/grounding_sft_v1_result/${model_name}"
fi
if [[ "$model_path" == *"amodal"* ]]; then
    result_dir="${result_dir}_amodal"
fi
echo "Result dir: $result_dir"
mkdir -p "$result_dir" 
# data_name_list=( "refcoco_testA" "refcoco_testB" "refcoco+_testA" "refcoco+_testB" "refcoco+_val" "refcocog_val" "refcocog_test" "refcoco_val")
# "refcoco_testA" "refcoco_testB" "refcoco+_testA" "refcoco+_testB"
# data_name_list=( "refcoco_trainv4" )
# "refcoco_testA" 
data_name_list=( "amodal_eval_v3_10.22_eval_filtered_with_modal_bbox" )
for data_name in "${data_name_list[@]}"; do
    echo "Processing dataset: $data_name"
    data_json="/storage/openpsi/data/grounding_sft_v1/${data_name}.jsonl"
    output_dir="${result_dir}/${data_name}_result"
    mkdir -p "$output_dir"
    python scripts/sglang_infer.py \
    --model_path $model_path \
    --data_json $data_json \
    --output_dir $output_dir \
    --endpoint "http://127.0.0.1:30000" 

done

# nohup python -m sglang.launch_server \
#   --model-path  /storage/openpsi/models/internvl3_5_1b_v7_2   \
#   --host 0.0.0.0 --port 30000 \
#   --nnodes 2 --node-rank 0 \
#   --dist-init-addr 33.180.160.252:50000 \
#   --dp-size 16 \
#   --dtype auto \
#   --mem-fraction-static 0.7 \
#   > /var/log/sglang_node0.log 2>&1 &

# nohup python -m sglang.launch_server \
#   --model-path  /storage/openpsi/models/internvl3_5_1b_amodaling_rl/trial1_global_step_80 \
#   --host 0.0.0.0 --port 30000 \
#   --nnodes 1 --node-rank 0 \
#   --dp-size 8 \
#   --tp-size 1 \
#   --dtype auto \
#   --mem-fraction-static 0.8 \
#   > /var/log/sglang_node0.log 2>&1 & 