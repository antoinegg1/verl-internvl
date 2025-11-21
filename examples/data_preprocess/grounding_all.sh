
OUTPUT_DIR="your_output_directory"
INPUT_JSON="your_input_directory/refcoco-train-refbox-addflip570K_1109.jsonl"
    
python examples/data_preprocess/grounding_all.py \
    --input_json $INPUT_JSON \
    --output_dir $OUTPUT_DIR \
    --format qwen \
    --num_workers 64
