#!/usr/bin/env bash
set -euo pipefail
set -x

# ===== 端口与日志等 =====
export RAY_MASTER_PORT=26379
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
# 任务名与输出目录
export PROJECT_NAME="internvl3_8b_grounding_rl"
TASK_NAME="trial5_8B_dapo_lesssft"
unset ROCR_VISIBLE_DEVICES || true
unset HIP_VISIBLE_DEVICES || true
export OUTPUT_PATH="${OUTPUT_PATH:-/storage/openpsi/models/${PROJECT_NAME}/${TASK_NAME}}"
export JOBLOG="${JOBLOG:-${OUTPUT_PATH}/traininnvitopg.log}"
RAY_WORKING_DIR="${RAY_WORKING_DIR:-${OUTPUT_PATH}/ray_working_dir}"
mkdir -p "${OUTPUT_PATH}" "${RAY_WORKING_DIR}"

export RAY_ADDRESS=33.180.166.157:26379

# 可选查看集群状态（不影响已运行的 head）
ray status || true
ray list nodes || true
echo "RAY_ADDRESS=${RAY_ADDRESS}"
echo "Dashboard: http://$(hostname -I | awk '{print $1}'):${RAY_DASHBOARD_PORT}"
echo "Cleaning up existing Ray jobs on ${RAY_ADDRESS}..."
# 列出所有 job（排除表头），逐个删除
ray job list --address="${RAY_ADDRESS}" | awk 'NR>2 {print $1}' | while read -r jid; do
  if [[ -n "$jid" && "$jid" != "----" ]]; then
    echo "Deleting Ray job: $jid"
    ray job delete --address="${RAY_ADDRESS}" "$jid" || true
  fi
done
echo "All previous jobs cleared."
# ===== 训练/推理超参（可通过环境变量覆盖）=====
export NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
export MICRO_TRAIN_BATCH_SIZE="${MICRO_TRAIN_BATCH_SIZE:-16}"
export MICRO_ROLLOUT_BATCH_SIZE="${MICRO_ROLLOUT_BATCH_SIZE:-16}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
export TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
export SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-1}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-256}"
# WORLD_SIZE=节点数量（含 head）。多机自行设置，如： WORLD_SIZE=4 ./submit_job.sh
export WORLD_SIZE="${WORLD_SIZE:-2}"

export NPROC_PER_NODE="${NPROC_PER_NODE:-${NUM_GPUS_PER_NODE}}"
export use_dynamic_bsz="${use_dynamic_bsz:-True}"
clip_ratio_low=0.2
clip_ratio_high=0.28
enable_filter_groups=True
filter_groups_metric=seq_reward
max_num_gen_batches=10
RUNTIME_ENV_JSON=$(jq -nc \
  --arg hf "$HF_ENDPOINT" \
  --arg wb "$WANDB_BASE_URL" \
  --arg key "$WANDB_API_KEY" \
  '{env_vars: {
      HF_ENDPOINT: $hf,
      WANDB_BASE_URL: $wb,
      WANDB_API_KEY: $key,
      ROCR_VISIBLE_DEVICES: "",
      HIP_VISIBLE_DEVICES: "",
      VLLM_ALLREDUCE_USE_SYMM_MEM: "0",
  }}')

#[/storage/openpsi/data/grounding_sft_v1_preprocessed/train_cot_v2_rl_flip_for8b_37K.parquet,/storage/openpsi/data/grounding_sft_v1_preprocessed/object365_train10K.parquet]
# ===== 提交 Ray Job =====
ray job submit --address="${RAY_ADDRESS}" \
  --runtime-env-json="$RUNTIME_ENV_JSON" \
  -- python3 -m recipe.dapo.main_dapo \
  algorithm.adv_estimator=grpo \
  data.train_files=/storage/openpsi/data/grounding_sft_v1_preprocessed/train_grounding_8b_lesssft_cot_100K.parquet \
  data.val_files=/storage/openpsi/data/grounding_sft_v1_preprocessed/test_mixed.parquet \
  data.train_batch_size="${ROLLOUT_BATCH_SIZE}" \
  data.max_prompt_length=4096 \
  data.gen_batch_size=768 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.filter_overlong_prompts_workers=8 \
  data.truncation='error' \
  data.image_key=images \
  data.trust_remote_code=True \
  data.shuffle=True \
  reward_model.enable=False \
  data.reward_fn_key=data_source \
  custom_reward_function.path=/storage/openpsi/users/lichangye.lcy/VeRL_InternVL/verl/utils/reward_score/grounding.py \
  custom_reward_function.name=compute_score \
  +custom_reward_function.reward_kwargs.alpha=0.5 \
  +custom_reward_function.reward_kwargs.threshold=0.5 \
  +custom_reward_function.reward_kwargs.reward_type=mix \
  actor_rollout_ref.model.path=/storage/openpsi/models/intern-8b-grounding-cot-sft-step50 \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.warmup_style=cosine \
  actor_rollout_ref.actor.optim.lr_warmup_steps=30 \
  actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
  actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.use_dynamic_bsz="${use_dynamic_bsz}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${use_dynamic_bsz}" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${use_dynamic_bsz}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True  \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True  \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size="${SEQUENCE_PARALLEL}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${MICRO_ROLLOUT_BATCH_SIZE}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TENSOR_PARALLEL}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.dtype=float16 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.enforce_eager=False  \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.n="${N_SAMPLES_PER_PROMPT}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${MICRO_TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.ulysses_sequence_parallel_size="${SEQUENCE_PARALLEL}" \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  algorithm.filter_groups.enable=${enable_filter_groups} \
  algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
  algorithm.filter_groups.metric=${filter_groups_metric} \
  trainer.critic_warmup=0.05 \
  trainer.default_local_dir="${OUTPUT_PATH}" \
  trainer.logger=['console','wandb'] \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${TASK_NAME}" \
  trainer.n_gpus_per_node="${NPROC_PER_NODE}" \
  trainer.nnodes="${WORLD_SIZE}" \
  trainer.save_freq=10 \
  trainer.test_freq=5 \
  trainer.resume_mode=auto \
  trainer.val_before_train=True \
  trainer.rollout_data_dir="${OUTPUT_PATH}/rollouts" \
  trainer.total_epochs=5 2>&1 | tee "${JOBLOG}"

  # trainer.resume_mode=resume_path \
  # trainer.resume_from_path=/storage/openpsi/models/internvl3_8b_grounding_rl/trial1_cot_v3/global_step_100 \