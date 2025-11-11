from datasets import Dataset

arrow_path = "/storage/openpsi/data/qwen_result/trial2_cot_global_step_130_4B/refcoco+_testB_result/data-00000-of-00001.arrow"
out_path = "out.jsonl"

ds = Dataset.from_file(arrow_path)
ds.to_json(out_path, lines=True, force_ascii=False)
