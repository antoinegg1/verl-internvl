
OUTPUT_DIR="your_output_directory"
INPUT_JSON="your_input_directory/refcoco-train-refbox-addflip570K_1109.jsonl"
    
python examples/data_preprocess/grounding.py \
    --input_json $INPUT_JSON \
    --output_dir $OUTPUT_DIR \
    --format qwen \
    --iou-key "your_iou_key" \
    --num_workers 64