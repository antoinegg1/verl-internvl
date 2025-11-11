#!/usr/bin/env bash
set -euo pipefail
set -x

export RAY_MASTER_PORT=6379
export RAY_DASHBOARD_PORT=8265
export MASTER_PORT=34235
export TF_CPP_MIN_LOG_LEVEL=3

# Ray 临时目录
export RAY_TMPDIR=/dev/shm/ray_wwy
rm -rf "${RAY_TMPDIR}"
mkdir -p "${RAY_TMPDIR}"

# 任务名
export PROJECT_NAME="qwen3_2b_grounding_rl"
if echo "$PROJECT_NAME" | grep -q "qwen"; then
    export PYTHONPATH=/storage/openpsi/users/lichangye.lcy/sglang/python
    echo "PYTHONPATH set for qwen project."
else
    echo "PROJECT_NAME does not contain 'qwen'. No PYTHONPATH set."
fi
TASK_NAME="trial3_cot"
echo "TASK_NAME: $TASK_NAME"
echo "PROJECT_NAME: $PROJECT_NAME"
unset ROCR_VISIBLE_DEVICES || true
unset HIP_VISIBLE_DEVICES || true
export OUTPUT_PATH="/storage/openpsi/models/${PROJECT_NAME}/${TASK_NAME}"
export JOBLOG="${OUTPUT_PATH}/training.log"
RAY_WORKING_DIR="${OUTPUT_PATH}/ray_working_dir"

mkdir -p "${OUTPUT_PATH}"  "${RAY_WORKING_DIR}"

# export TRITON_CACHE_DIR="/dev/shm/triton_wwy/"
# export VLLM_CACHE_ROOT="/dev/shm/vllmca_wwy/"

# 让下游程序自动找到本机 head
export RAY_ADDRESS="127.0.0.1:${RAY_MASTER_PORT}"

# 确保干净启动；忽略未运行时的报错
ray stop --force || true

echo "start ray head" &>> "${JOBLOG}"

# 启动本机 head（单机 8 卡）
ray start \
  --head \
  --port="${RAY_MASTER_PORT}" \
  --dashboard-host=0.0.0.0 \
  --dashboard-port="${RAY_DASHBOARD_PORT}" \
  --temp-dir="${RAY_TMPDIR}" \
  --num-gpus=8

echo "Ray head started on ${RAY_ADDRESS}" &>> "${JOBLOG}"

# 可选：展示集群状态
sleep 3
ray status || true
ray list nodes || true
echo "RAY_ADDRESS=${RAY_ADDRESS}"
echo "Dashboard: http://$(hostname -I | awk '{print $1}'):${RAY_DASHBOARD_PORT}"

# Dynamically compute the number of nodes and total GPU engines
NUM_GPUS_PER_NODE=8
MICRO_TRAIN_BATCH_SIZE=32
MICRO_ROLLOUT_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=256
N_SAMPLES_PER_PROMPT=16
TENSOR_PARALLEL=1
SEQUENCE_PARALLEL=1
PPO_MINI_BATCH_SIZE=256
WORLD_SIZE=1

NPROC_PER_NODE=8
use_dynamic_bsz=True
# /storage/openpsi/data/grounding_sft_v1_preprocessed/test_mixed.parquet
#/storage/openpsi/data/amodal_data_preprocessed/val_amodal_v5.parquet
#/storage/openpsi/data/amodal_data_preprocessed/train_amodal_v5.parquet 
# ===== Submit the ray job =====
ray job submit --address=${RAY_ADDRESS} \
    -- python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/storage/openpsi/data/grounding_sft_v1_preprocessed/train_grounding_qwen2bcot_110K.parquet  \
    data.val_files=/storage/openpsi/data/grounding_sft_v1_preprocessed/test_grounding_qwen.parquet \
    data.train_batch_size=${ROLLOUT_BATCH_SIZE} \
    data.max_prompt_length=16384\
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=16 \
    data.truncation='error' \
    data.image_key=images \
    data.trust_remote_code=True \
    reward_model.enable=False \
    data.reward_fn_key=data_source \
    custom_reward_function.path=/storage/openpsi/users/lichangye.lcy/VeRL_InternVL/verl/utils/reward_score/grounding.py \
    custom_reward_function.name=compute_score \
    +custom_reward_function.reward_kwargs.reward_type=mix \
    +custom_reward_function.reward_kwargs.alpha=0.5 \
    +custom_reward_function.reward_kwargs.threshold=0.5 \
    actor_rollout_ref.model.path=/storage/openpsi/models/qwen3_2b_thinking_grounding_sft_4450itr \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps=30 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=25600 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_TRAIN_BATCH_SIZE} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SEQUENCE_PARALLEL} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_ROLLOUT_BATCH_SIZE} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_PARALLEL} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=${N_SAMPLES_PER_PROMPT} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_TRAIN_BATCH_SIZE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${SEQUENCE_PARALLEL} \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.default_local_dir=${OUTPUT_PATH} \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${TASK_NAME} \
    trainer.n_gpus_per_node=${NPROC_PER_NODE} \
    trainer.nnodes=${WORLD_SIZE} \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.val_before_train=True \
    trainer.rollout_data_dir=${OUTPUT_PATH}/rollouts \
    trainer.total_epochs=3 2>&1 | tee ${JOBLOG}

  # trainer.resume_mode=resume_path \
  # trainer.resume_from_path=/storage/openpsi/models/internvl3_8b_grounding_rl/trial1_cot_v3/global_step_100 \
    #   trainer.resume_mode=resume_path \
    # trainer.resume_from_path=/storage/openpsi/models/qwen3_2b_grounding_rl/trial2_cot/global_step_60 \