<div align="center">

# InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency

<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/930e6814-8a9f-43e1-a284-118a5732daa4">
  <br>
</div>

[\[🔥 InternVL3.5 Report\]](https://huggingface.co/papers/2508.18265)
[\[🗨️ Chat Demo\]](https://chat.intern-ai.org.cn/)

![image/jpg](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B/resolve/main/images/performance.jpg)

</div>

This repository open-sources the training code of InternVL3.5 during the online RL stage, which is built upon the [PR](https://github.com/volcengine/verl/pull/2327) in [verl](https://github.com/volcengine/verl). Compared to the original PR, we have corrected the dialogue template for InternVL and updated a monkey patch for InternVL to enable sequence-parallel. For training details, please refer to the provided [scripts](shell).

We use [MMPR-Tiny](https://huggingface.co/datasets/OpenGVLab/MMPR-Tiny) as the training dataset and initialize the model with InternVL3.5 trained after MPO. We also provide a [packaged conda environment](https://huggingface.co/Weiyun1025/InternVL3_5-RL-conda-env/blob/main/verl-internvl.tar.gz) for easy reproduction.

For the original README of verl, please refer to [this file](README_verl.md).

## Grounding

*Remember to replace all local paths, IP addresses, and ports in the programs with the corresponding ones on your server.*


### Infer/Evaluate

```bash
nohup python -m sglang.launch_server \
  --model-path #{Your_model_path}   \
  --host 0.0.0.0 --port 30000 \
  --nnodes 1 --node-rank 0 \
  --dp-size 8 \
  --tp-size 1 \
  --dtype auto \
  --mem-fraction-static 0.8 \
  > /var/log/sglang_node0.log 2>&1 &
```
After sglang launched, run
```bash 
bash scripts/sglang_infer.sh
```
### RL Training


Preprocess your dataset with
```bash 
bash /examples/data_preprocess/grounding.sh
```
which simples 10% from each validation dataset.

For single node training (1B model),
```bash 
bash /shell/grounding_internvl3_5_1b.sh
```
For multinode training (8B model), run
```bash 
bash /scripts/launch_ray.sh
```
on the head node and run
```bash 
bash /scripts/launch_ray_worker.sh
```
on the worker nodes

Then run
```bash 
bash /scripts/submit_job.sh
```
on the head node.



## Experimental Results

Based on this codebase, the InternVL3.5 series across all model scales achieve a significant improvement in reasoning performance.

![image/jpg](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B/resolve/main/images/ablation_cascade_rl.jpg)

![image/jpg](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B/resolve/main/images/ablation_cascade_rl_table.jpg)

## Quick Start

### Training

We open-source our training data (i.e., [MMPR-Tiny](https://huggingface.co/datasets/OpenGVLab/MMPR-Tiny)) on HuggingFace. To reproduce our training results, you need to download this dataset and move it into this folder.
Additionally, considering that verl requires a validation dataset to be loaded, please prepare this data using this [script](examples/data_preprocess/geo3k.py).

```
├── MMPR-Tiny
│   ├── images
│   └── mmpr_tiny.parquet
├── verl_data
│   └── geo3k
│       └── test.parquet
├── verl
└── README.md
```

We also provide a [packaged conda environment](https://huggingface.co/Weiyun1025/InternVL3_5-RL-conda-env/blob/main/verl-internvl.tar.gz) for easy reproduction. After preparing the dataset and conda environment, you can launch the training using the commond as follows:

```shell
sh shell/internvl3_5_8b.sh
```



### Evaluation

We mainly use [VLMEvalkit](https://github.com/open-compass/VLMEvalKit) to evaluate our models. Please refer to their documentation and our model configs for more details. As an example, you can set the config as follows:

```python
"InternVL3_5-8B-Thinking": partial(
    InternVLChat,
    model_path="/path/to/your/moel",
    version="V2.0",
    cot_prompt_version="r1",
    max_new_tokens=32768,
    do_sample=True,
    use_lmdeploy=True,
),
```

## Citation
If you find this project useful in your research, please consider citing:
```BibTeX
@article{wang2025internvl3_5,
  title={InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency},
  author={Wang, Weiyun and Gao, Zhangwei and Gu, Lixin and Pu, Hengjun and Cui, Long and Wei, Xingguang and Liu, Zhaoyang and Jing, Linglin and Ye, Shenglong and Shao, Jie and others},
  journal={arXiv preprint arXiv:2508.18265},
  year={2025}
}
@article{wang2024mpo,
  title={Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization},
  author={Wang, Weiyun and Chen, Zhe and Wang, Wenhai and Cao, Yue and Liu, Yangzhou and Gao, Zhangwei and Zhu, Jinguo and Zhu, Xizhou and Lu, Lewei and Qiao, Yu and Dai, Jifeng},
  journal={arXiv preprint arXiv:2411.10442},
  year={2024}
}
```
