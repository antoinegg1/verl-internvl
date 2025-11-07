# 需要跑的 steps（只改这里）
steps=(310 520)

# 固定不变的参数
REF_1B="/storage/openpsi/models/InternVL3-1B"
REF_8B="/storage/openpsi/models/InternVL3-8B"
REF_2B="/storage/openpsi/models/grounding_model/internvl3_2b_v5"
REF_QWEN_2B="/storage/openpsi/models/qwen3-vl-2b-direct_box-sft"
REF_QWEN_4B="/storage/openpsi/models/qwen3-vl-4b-direct_box-sft"
REF_QWEN_8B="/storage/openpsi/models/qwen3-vl-8b-direct_box"
REF=${REF_2B}

for step in "${steps[@]}"; do
  DST="/storage/openpsi/models/internvl3_2b_grounding_rl/trial1_directbbox_global_step_${step}"
  SRC="/storage/openpsi/models/internvl3_2b_grounding_rl/trial1_directbbox/global_step_${step}/actor"

  cp ${REF}/*json  ${SRC}/huggingface/
  cp ${REF}/*py    ${SRC}/huggingface/

  HF_ENDPOINT=https://hf-mirror.com python scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir ${SRC} \
    --target_dir ${DST}

  rm -f ${DST}/*json
  rm -f ${DST}/*py
  rm -f ${DST}/*jinja

  cp ${REF}/*json  ${DST}/
  cp ${REF}/*py    ${DST}/
  cp ${REF}/*jinja  ${DST}/

  echo "done: global_step_${step}"
done
