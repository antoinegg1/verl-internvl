#!/bin/bash
# RefCOCO Evaluation Script for Qwen3-VL Models
# Usage: bash run_refcoco_eval.sh [MODEL_NAME] [DATASET] [MAX_SAMPLES]

set -e

# ============================================================================
# Configuration
# ============================================================================

# Available models (cached in HF)
# Can also use custom paths like /path/to/checkpoint-xxxx
declare -A MODELS=(
    ["2b-directbox-sft"]="/storage/openpsi/models/qwen3-vl-2b-direct_box-sft"
    ["2b-thinking"]="Qwen/Qwen3-VL-2B-Thinking"
    ["4b-instruct"]="Qwen/Qwen3-VL-4B-Instruct"
    ["4b-thinking"]="Qwen/Qwen3-VL-4B-Thinking"
    ["8b-instruct"]="Qwen/Qwen3-VL-8B-Instruct"
    ["8b-thinking"]="Qwen/Qwen3-VL-8B-Thinking"
    # SFT models - shortcuts
    ["4b-instruct-sft"]="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/gzhan/Qwen3-VL/qwen-vl-finetune/output/4b_instruct_grounding_sft/checkpoint-4494"
    ["4b-thinking-sft"]="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/gzhan/Qwen3-VL/qwen-vl-finetune/output/4b_thinking_grounding_sft/checkpoint-4494"
)

# Available datasets
declare -A DATASETS=(
    ["refcoco-testA"]="refcoco_testA"
    ["refcoco-val"]="refcoco_val"
    ["refcoco-testB"]="refcoco_testB"
    ["refcoco+val"]="refcoco+_val"
    ["refcoco+testA"]="refcoco+_testA"
    ["refcoco+testB"]="refcoco+_testB"
    ["refcocog-val"]="refcocog_val"
    ["refcocog-test"]="refcocog_test"
    ["refcoco-all"]="refcoco_val refcoco_testA refcoco_testB"
    ["refcoco+-all"]="refcoco+_val refcoco+_testA refcoco+_testB"
    ["refcocog-all"]="refcocog_val refcocog_test"
    ["all"]="refcoco_val refcoco_testA refcoco_testB refcoco+_val refcoco+_testA refcoco+_testB refcocog_val refcocog_test"
)

# Default settings
DEFAULT_MODEL="2b-directbox-sft"
DEFAULT_DATASET="all"
DEFAULT_MAX_SAMPLES=""  # empty means evaluate all samples
DEFAULT_GPUS=8  # Number of GPUs to use
DEFAULT_OUT_DIR="/storage/openpsi/models/grounding_model/"
DEFAULT_BATCH_SIZE=32  # Batch size per GPU

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Distributed training settings
MASTER_PORT=${MASTER_PORT:-12345}
export MASTER_PORT=${MASTER_PORT}

# ============================================================================
# Helper Functions
# ============================================================================

function print_usage() {
    cat << EOF
RefCOCO Evaluation for Qwen3-VL (Multi-GPU Support)

Usage: 
    bash $0 [MODEL] [DATASET] [MAX_SAMPLES] [GPUS] [OUT_DIR] [BATCH_SIZE]

Arguments:
    MODEL       : Model to evaluate (default: ${DEFAULT_MODEL})
                  Can be a model name key or a full path to checkpoint
    DATASET     : Dataset split(s) to evaluate (default: ${DEFAULT_DATASET})
    MAX_SAMPLES : Maximum samples per split (default: all samples)
    GPUS        : Number of GPUs to use (default: ${DEFAULT_GPUS})
    OUT_DIR     : Output directory for results (default: ${DEFAULT_OUT_DIR})
    BATCH_SIZE  : Batch size per GPU (default: ${DEFAULT_BATCH_SIZE})

Available Models:
$(for key in "${!MODELS[@]}"; do echo "    - $key : ${MODELS[$key]}"; done | sort)

Available Datasets:
$(for key in "${!DATASETS[@]}"; do echo "    - $key : ${DATASETS[$key]}"; done | sort)

Examples:
    # Evaluate 2B Instruct on all RefCOCO datasets with 8 GPUs, batch_size=4
    bash $0 2b-instruct all "" 8 "" 4

    # Evaluate SFT model (using shortcut key)
    bash $0 4b-instruct-sft all "" 4 ./results_sft 4

    # Evaluate SFT model (using full path)
    bash $0 /lustre/fsw/.../checkpoint-4494 all "" 4 ./results 4

    # Evaluate 4B Instruct on RefCOCO val split with 4 GPUs
    bash $0 4b-instruct refcoco-val "" 4

    # Evaluate 2B Thinking on first 100 samples with 2 GPUs
    bash $0 2b-thinking all 100 2

    # Quick test with 10 samples on single GPU
    bash $0 2b-instruct refcoco-val 10 1

    # Higher batch size for better GPU utilization
    bash $0 8b-instruct all "" 8 /path/to/results 8
EOF
}

function print_header() {
    echo "============================================"
    echo "$1"
    echo "============================================"
}

# ============================================================================
# Parse Arguments
# ============================================================================

MODEL_KEY="${1:-$DEFAULT_MODEL}"
DATASET_KEY="${2:-$DEFAULT_DATASET}"
MAX_SAMPLES="${3:-$DEFAULT_MAX_SAMPLES}"
GPUS="${4:-$DEFAULT_GPUS}"
OUT_DIR="${5:-$DEFAULT_OUT_DIR}"
BATCH_SIZE="${6:-$DEFAULT_BATCH_SIZE}"

# Show usage if requested
if [[ "$MODEL_KEY" == "-h" ]] || [[ "$MODEL_KEY" == "--help" ]]; then
    print_usage
    exit 0
fi

# Determine model path
# If MODEL_KEY exists in MODELS dict, use it; otherwise treat as a path
if [[ -v MODELS[$MODEL_KEY] ]]; then
    MODEL_PATH="${MODELS[$MODEL_KEY]}"
    MODEL_NAME="$MODEL_KEY"
elif [[ -d "$MODEL_KEY" ]] || [[ "$MODEL_KEY" == *"/"* ]]; then
    # Treat as a path (directory or HF model name with /)
    MODEL_PATH="$MODEL_KEY"
    MODEL_NAME="$(basename $MODEL_KEY)"
    echo "Using custom model path: $MODEL_PATH"
else
    echo "Error: Unknown model '$MODEL_KEY' and not a valid path"
    echo ""
    print_usage
    exit 1
fi

# Validate dataset
if [[ ! -v DATASETS[$DATASET_KEY] ]]; then
    echo "Error: Unknown dataset '$DATASET_KEY'"
    echo ""
    print_usage
    exit 1
fi

DATASET_LIST="${DATASETS[$DATASET_KEY]}"

# ============================================================================
# Run Evaluation
# ============================================================================

print_header "Qwen3-VL RefCOCO Evaluation (Multi-GPU)"
echo "Model      : $MODEL_NAME"
echo "Model Path : $MODEL_PATH"
echo "Dataset(s) : $DATASET_KEY ($DATASET_LIST)"
echo "Max Samples: ${MAX_SAMPLES:-All}"
echo "GPUs       : $GPUS"
echo "Batch Size : $BATCH_SIZE"
echo "Output Dir : $OUT_DIR"
echo ""

# Convert to absolute path if relative
if [[ ! "$OUT_DIR" = /* ]]; then
    OUT_DIR="$(pwd)/$OUT_DIR"
fi

# Create output directory
mkdir -p "$OUT_DIR"
echo "Results will be saved to: $OUT_DIR"
echo ""

# Build command with torchrun for distributed evaluation
CMD="torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=$GPUS \
    --master_port=$MASTER_PORT \
    $SCRIPT_DIR/eval_refcoco.py \
    --model-path \"$MODEL_PATH\" \
    --datasets $DATASET_LIST \
    --out-dir \"$OUT_DIR\" \
    --batch-size $BATCH_SIZE"

# Add max-samples if specified
if [[ -n "$MAX_SAMPLES" ]]; then
    CMD="$CMD --max-samples $MAX_SAMPLES"
fi

# Add save-predictions for small evaluations
if [[ -n "$MAX_SAMPLES" ]] && [[ "$MAX_SAMPLES" -le 100 ]]; then
    CMD="$CMD --save-predictions"
    echo "Note: Saving detailed predictions (鈮�100 samples)"
    echo ""
fi

# Print command
print_header "Running Evaluation"
echo "Command:"
echo "$CMD"
echo ""

# Execute
eval $CMD

# ============================================================================
# Summary
# ============================================================================

echo ""
print_header "Evaluation Complete"
echo "Results saved to: $OUT_DIR"
echo ""
echo "To view results:"
echo "  ls -lht $OUT_DIR/"
echo ""
echo "To compare models:"
echo "  cat $OUT_DIR/*_summary_*.txt"
echo ""
