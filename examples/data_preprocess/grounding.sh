# INPUT_DIR=/storage/openpsi/data/grounding_sft_v1/
INPUT_DIR=/storage/openpsi/data/amodal_data
OUTPUT_DIR=/storage/openpsi/data/amodal_data_preprocessed/
# OUTPUT_DIR=/storage/openpsi/data/grounding_sft_v1_preprocessed/
# TRAIN_FILE=("refcoco-train-refbox-addflip570K_1109.jsonl")
# INPUT_DIR=/storage/openpsi/data/object365_vanilla_grounding_train/
# TRAIN_FILE=("object365_train_v1_for_changye_with_8b_iou.jsonl")
TRAIN_FILE=(amodal_val_v5/amodal_train_v5_filtered_val_with_modal_bbox.jsonl)
VALID_FILE=(amodal_val_v5/amodal_train_v5_filtered_val_with_modal_bbox.jsonl )
# TRAIN_FILE=(stage2_thinking_with_text_sft_train_10_24_3.jsonl)
# VALID_FILE=( "refcoco_testA.jsonl" "refcoco_testB.jsonl" "refcoco+_testA.jsonl" "refcoco+_testB.jsonl" "refcoco+_val.jsonl" "refcocog_val.jsonl" "refcocog_test.jsonl" "refcoco_val.jsonl")

python examples/data_preprocess/grounding.py \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --train_files ${TRAIN_FILE[@]} \
    --test_files ${VALID_FILE[@]} \
    --format qwen \
    --iou-key "qwen3_4b_thinking_model_iou" \
    --bucket-specs  "[(0.50, 0.60, 3966),(0.60, 0.70, 7932),(0.70, 0.80, 7932),(0.80, 0.90, 7932),(0.90, 0.98, 11898)]" \
    --num_workers 64