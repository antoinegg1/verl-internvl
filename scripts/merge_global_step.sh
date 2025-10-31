#model merger
DST=/storage/openpsi/models/internvl3_1b_amodaling_rl/trial1_global_step_700
REF_1B=" /storage/openpsi/models/InternVL3-1B"
REF_8B=" /storage/openpsi/models/InternVL3-8B"
REF=${REF_1B}
SRC=/storage/openpsi/models/internvl3_1b_amodaling_rl/trial1/global_step_700/actor

cp ${REF}/*json  ${SRC}/huggingface/
cp ${REF}/*py    ${SRC}/huggingface/

HF_ENDPOINT=https://hf-mirror.com python scripts/legacy_model_merger.py merge \
  --backend fsdp \
  --local_dir ${SRC} \
  --target_dir ${DST}
  #参数全部换成/storage/openpsi/models/internvl3_1b_cot_thinking_with_text
#1B REF/storage/openpsi/models/internvl3_5_1b_grounding_rl/hf_merged_step10
#8B REF /storage/openpsi/models/internvl3_5_8b_grounding_rl/hf_merged_step180



# 清理目标目录中的旧文件
rm -f ${DST}/*json
rm -f ${DST}/*py
rm -f ${DST}/*jinja

# 复制源目录中的文件到目标目录
cp ${REF}/*json  ${DST}/
cp ${REF}/*py    ${DST}/