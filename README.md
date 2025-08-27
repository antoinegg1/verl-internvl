<div align="center">

# InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency

<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/930e6814-8a9f-43e1-a284-118a5732daa4">
  <br>
</div>

[\[🔥 InternVL3.5 Report\]](https://huggingface.co/papers/2508.18265)
[\[🗨️ Chat Demo\]]()

![image/jpg](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B/resolve/main/images/performance.jpg)

</div>

This repository open-sources the training code of InternVL3.5 during the online RL stage, which is built upon the [PR](https://github.com/volcengine/verl/pull/2327) in [verl](https://github.com/volcengine/verl). Compared to the original PR, we have corrected the dialogue template for InternVL and updated a monkey patch for InternVL to enable sequence-parallel. For training details, please refer to the provided [scripts](shell).

We use [MMPR-Tiny](https://huggingface.co/datasets/OpenGVLab/MMPR-Tiny) as the training dataset and initialize the model with InternVL3.5 trained after MPO. We also provide a [packaged conda environment](https://huggingface.co/Weiyun1025/InternVL3_5-RL-conda-env/blob/main/verl-internvl.tar.gz) for easy reproduction.

For the original README of verl, please refer to [this file](README_verl.md).

## Experimental Results

Based on this codebase, the InternVL3.5 series across all model scales achieve a significant improvement in reasoning performance.

![image/jpg](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B/resolve/main/images/ablation_cascade_rl.jpg)

![image/jpg](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B/resolve/main/images/ablation_cascade_rl_table.jpg)
