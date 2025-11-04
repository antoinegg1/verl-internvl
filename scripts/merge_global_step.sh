# 需要跑的 steps（只改这里）
steps=(820 900 1000)

# 固定不变的参数
REF_1B="/storage/openpsi/models/InternVL3-1B"
REF_8B="/storage/openpsi/models/InternVL3-8B"
REF=${REF_1B}

for step in "${steps[@]}"; do
  DST="/storage/openpsi/models/internvl3_1b_grounding_rl/trial7_caption_global_step_${step}"
  SRC="/storage/openpsi/models/internvl3_1b_grounding_rl/trial7_caption/global_step_${step}/actor"

  mkdir -p "${SRC}/huggingface" "${DST}"

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

  echo "done: global_step_${step}"
done
