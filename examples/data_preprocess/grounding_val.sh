#!/usr/bin/env bash
set -euo pipefail

BASE_IMG_PATH="${BASE_IMG_PATH:-/storage/openpsi/data}"
export BASE_IMG_PATH

INPUT_DIR="$your_input_directory"
OUTPUT_DIR="$your_output_directory"

VALID_FILES=(
  "refcoco_testA.jsonl"
  "refcoco_testB.jsonl"
  "refcoco+_testA.jsonl"
  "refcoco+_testB.jsonl"
  "refcoco+_val.jsonl"
  "refcocog_val.jsonl"
  "refcocog_test.jsonl"
  "refcoco_val.jsonl"
)

python examples/data_preprocess/grounding_val.py \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --test_files "${VALID_FILES[@]}" \
  --format qwen \
  --num_workers 64
