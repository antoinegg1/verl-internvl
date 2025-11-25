# 需要跑的 steps（只改这里）
steps=(260 400 450)

# 固定不变的参数
REF_1B="${your_InternVL3-1B_path}"
REF_2B="${your_InternVL3-2B_path}"
REF_8B="${your_InternVL3-8B_path}"
REF=${REF_1B}

for step in "${steps[@]}"; do
  DST="${your_output_dir}"
  SRC="/storage/openpsi/models/qwen3_4b_grounding_rl/trial2_cot_resume/global_step_${step}/actor"

  # 只有当 REF 不包含 qwen 时才做 cp 到 SRC
  if [[ "${REF}" != *qwen* ]]; then
    cp "${REF}"/*json  "${SRC}/huggingface/"
    cp "${REF}"/*py    "${SRC}/huggingface/"
  fi

  HF_ENDPOINT=https://hf-mirror.com python scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir "${SRC}" \
    --target_dir "${DST}"

  # 只有当 REF 不包含 qwen 时才做清理和回填
  if [[ "${REF}" != *qwen* ]]; then
    rm -f "${DST}"/*json
    rm -f "${DST}"/*py
    rm -f "${DST}"/*jinja

    cp "${REF}"/*json  "${DST}/"
    cp "${REF}"/*py    "${DST}/"
    cp "${REF}"/*jinja "${DST}/"
  fi

  echo "done: global_step_${step}"
done
