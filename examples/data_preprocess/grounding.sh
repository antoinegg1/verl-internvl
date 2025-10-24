INPUT_DIR=/storage/openpsi/data/grounding_sft_v1/
OUTPUT_DIR=/storage/openpsi/data/grounding_sft_v1_preprocessed/
TRAIN_FILE=(stage2_thinking_with_text_sft_train_10_24.jsonl)
VALID_FILE=( "refcoco_testA.jsonl" "refcoco_testB.jsonl" "refcoco+_testA.jsonl" "refcoco+_testB.jsonl" "refcoco+_val.jsonl" "refcocog_val.jsonl" "refcocog_test.jsonl" "refcoco_val.jsonl")
python examples/data_preprocess/grounding.py \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --train_files ${TRAIN_FILE[@]} \
    --test_files ${VALID_FILE[@]} \
    --num_workers 64