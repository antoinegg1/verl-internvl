import json

src = "/storage/openpsi/data/grounding_sft_v1/refcoco-train-refbox-addflip570K_1104_final.jsonl"
dst = "/storage/openpsi/data/grounding_sft_v1/refcoco-train-refbox-addflip570K_1107.jsonl"

with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)

        # 删除所有名字里带 qwen 的列
        keys = list(obj.keys())
        for k in keys:
            if "qwen" in k:
                del obj[k]

        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
