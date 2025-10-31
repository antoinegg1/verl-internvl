INPUT_DIR=/storage/openpsi/data/grounding_cot_v3_train_rl/
OUTPUT_DIR=/storage/openpsi/data/grounding_cot_v3_train_rl_preprocessed/
TRAIN_FILE=(refcoco_train_v5_gpt_candidate_10.28_withflip_RL_1030.jsonl)
VALID_FILE=(amodal_val_v5/amodal_train_v5_filtered_val_with_modal_bbox.jsonl )
# TRAIN_FILE=(stage2_thinking_with_text_sft_train_10_24_3.jsonl)
# VALID_FILE=( "refcoco_testA.jsonl" "refcoco_testB.jsonl" "refcoco+_testA.jsonl" "refcoco+_testB.jsonl" "refcoco+_val.jsonl" "refcocog_val.jsonl" "refcocog_test.jsonl" "refcoco_val.jsonl")
python examples/data_preprocess/grounding.py \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --train_files ${TRAIN_FILE[@]} \
    --test_files ${VALID_FILE[@]} \
    --num_workers 64