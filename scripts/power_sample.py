#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
torchrun --nproc_per_node=8 verl-internvl/scripts/power_sample.py --dataset <path.jsonl> --model_path <local_dir> --save_dir <out_dir>
'''
import os, json, math, random, argparse, re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import tqdm
import torch.distributed as dist
import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset 
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
PROMPT_TEMPLATE = (
    "Please provide the bounding box coordinate of the region this sentence describes: <ref>{sent}</ref> " 
)

# Paths and parsing helpers
BASE_IMAGE_DIR = "/storage/openpsi/data"

# InternVL tiling preprocess defaults
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size: int):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image: Image.Image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file: str, input_size=448, max_num=12) -> torch.Tensor:
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# Precompiled patterns for bbox extraction
NUM_RE = r"([+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)"
PAT_NAMED = re.compile(rf"\"bbox_2d\"\s*:\s*\[\s*{NUM_RE}\s*,\s*{NUM_RE}\s*,\s*{NUM_RE}\s*,\s*{NUM_RE}\s*\]", re.S)
PAT_ANY = re.compile(rf"\[\s*{NUM_RE}\s*,\s*{NUM_RE}\s*,\s*{NUM_RE}\s*,\s*{NUM_RE}\s*\]")

def parse_bbox_from_text(text: str) -> Optional[List[float]]:
    m = PAT_NAMED.search(text)
    if not m:
        m = PAT_ANY.search(text)
    if not m:
        return None
    vals = [float(m.group(i)) for i in range(1, 5)]
    x1, y1, x2, y2 = vals
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]

def compute_rank_metrics(jsonl_path: str, dataset) -> Dict[str, Dict[str, float]]:
    ious: List[float] = []
    with open(jsonl_path, "r", encoding="utf-8") as fin:
        for ex, line in zip(dataset, fin):
            rec = json.loads(line)
            gt = ex.get("bbox") if isinstance(ex, dict) else ex["bbox"]
            pred = parse_bbox_from_text(rec.get("mcmc_completion", "") or "")
            ious.append(compute_iou(pred, gt) if gt is not None else 0.0)
    return {"mcmc": summarize_ious(ious)}

def aggregate_metrics_across_ranks(save_dir: str, model_basename: str, alpha: float, proposal_T: float, mcmc_steps: int) -> Dict[str, Dict[str, float]]:
    import glob
    pattern = os.path.join(save_dir, f"{model_basename}_alpha{alpha}_T{proposal_T}_steps{mcmc_steps}_rank*.jsonl")
    files = sorted(glob.glob(pattern))

    # Collect IoUs directly from outputs using gtix and embedded ground truth
    seen: set = set()
    ious: List[float] = []
    for p in files:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                gtix = rec.get("gtix")
                if gtix is None or gtix in seen:
                    continue
                pred = parse_bbox_from_text(rec.get("mcmc_completion", "") or "")
                gt = rec.get("gt_bbox")
                if pred is None or gt is None:
                    continue
                seen.add(gtix)
                ious.append(compute_iou(pred, gt))
    return {"mcmc": summarize_ious(ious)}

def summarize_ious(ious: List[float]) -> Dict[str, float]:
    valid = [x for x in ious if x is not None]
    if not valid:
        return {"mean_iou": 0.0, "pass_rate_05": 0.0}
    mean_iou = sum(valid) / len(valid)
    pass_rate_05 = sum(1 for x in valid if x > 0.5) / len(valid)
    return {"mean_iou": mean_iou, "pass_rate_05": pass_rate_05}

def scale_bbox(bbox, w, h):
    x1, y1, x2, y2 = bbox
    x1 = round(x1 * w / 1000.0)
    x2 = round(x2 * w / 1000.0)
    y1 = round(y1 * h / 1000.0)
    y2 = round(y2 * h / 1000.0)

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    if x1 == x2 and w > 1: x2 = min(x1 + 1, w - 1)
    if y1 == y2 and h > 1: y2 = min(y1 + 1, h - 1)
    return [float(x1), float(y1), float(x2), float(y2)]

def compute_iou(boxA, boxB):
    """计算两个bbox的IoU"""
    if boxA is None or boxB is None:
        return 0.0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    areaB = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))

    if areaA + areaB - inter_area == 0:
        return 0.0
    return inter_area / (areaA + areaB - inter_area)

@dataclass
class VLMInputs:
    model_inputs: Dict[str, torch.Tensor]
    # 为了计算 token 对齐的 logprobs，我们保留 prompt len
    prompt_len: int

class VLMWrapper:
    def __init__(self, model_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.isdir(model_path):
            raise ValueError(f"Model path '{model_path}' is not a valid directory.")

        # InternVL-style model and tokenizer
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        self.eos_token_id = self.tokenizer.eos_token_id
        # Preprocess defaults
        self.input_size = 448
        self.max_tiles = 12
        # Ensure InternVL image context token is configured for generate()
        self._ensure_img_ctx_token()

    def _tokens_per_tile(self) -> int:
        cfg = getattr(self.model, "config", None)
        tpt = None
        # Prefer model attribute if provided by remote code
        tpt = getattr(self.model, "img_context_token_num", None)
        if tpt is None and cfg is not None:
            tpt = getattr(cfg, "img_context_token_num", None)
        # Fallback to conservative default 256 (observed in traces)
        try:
            tpt = int(tpt) if tpt is not None else 256
        except Exception:
            tpt = 256
        return tpt

    def _estimate_vision_tokens(self, num_tiles: int) -> int:
        return int(num_tiles) * self._tokens_per_tile()

    def _ensure_img_ctx_token(self):
        cfg = getattr(self.model, "config", None)
        token_str = None
        if cfg is not None:
            token_str = getattr(cfg, "img_context_token", None)
        # try to find a plausible token in tokenizer specials
        if not token_str:
            specials = getattr(self.tokenizer, "additional_special_tokens", []) or []
            for t in specials:
                if isinstance(t, str) and ("image" in t.lower() or "img" in t.lower()):
                    token_str = t
                    break
        if not token_str:
            token_str = "<image>"

        tok_id = self.tokenizer.convert_tokens_to_ids(token_str)
        # if tokenizer doesn't know this token, register it and resize embeddings
        if tok_id is None or tok_id == self.tokenizer.unk_token_id or (isinstance(tok_id, int) and tok_id < 0):
            self.tokenizer.add_special_tokens({"additional_special_tokens": [token_str]})
            if hasattr(self.model, "resize_token_embeddings"):
                self.model.resize_token_embeddings(len(self.tokenizer))
            tok_id = self.tokenizer.convert_tokens_to_ids(token_str)

        if hasattr(self.model, "img_context_token_id") and isinstance(tok_id, int) and tok_id >= 0:
            self.model.img_context_token_id = int(tok_id)
        # keep token string for prompt construction
        self.image_token_str = token_str

    def build_inputs(self, image_path: Optional[str], text: str) -> VLMInputs:
        # InternVL-style: text via tokenizer; image -> pixel_values tiles
        pixel_values = None
        if image_path:
            pv = load_image(image_path, input_size=self.input_size, max_num=self.max_tiles)
            pixel_values = pv.to(self.device)
            # Cast to model dtype if needed
            pixel_values = pixel_values.to(self.model.dtype)

        # Ensure image context token in prompt when image is present
        if pixel_values is not None:
            tok_str = getattr(self, "image_token_str", "<image>")
            # Repeat context token to match expected vision tokens count exactly
            needed = self._estimate_vision_tokens(pixel_values.shape[0])
            if needed > 0:
                # Avoid accidental duplicate if user already placed tokens; we use a clean prefix
                text = (tok_str + "\n") * needed + text

        tok = self.tokenizer(text, return_tensors='pt')
        model_inputs: Dict[str, torch.Tensor] = {k: v.to(self.device) for k, v in tok.items()}
        if pixel_values is not None:
            model_inputs['pixel_values'] = pixel_values
            # Some InternVL generate() variants expect num_patches_list to split tiles
            model_inputs['num_patches_list'] = [int(pixel_values.shape[0])]

        prompt_len = model_inputs['input_ids'].shape[-1]
        return VLMInputs(model_inputs=model_inputs, prompt_len=prompt_len)

    @torch.inference_mode()
    def generate_with_scores(
        self,
        vlm_inputs: VLMInputs,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
    ):
        """
        返回生成 token ids（仅新生成段），以及每步 logits -> 逐 token logprob 可用。
        """
        out = self.model.generate(
            **vlm_inputs.model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=max(1e-6, temperature),
            return_dict_in_generate=True,
            output_scores=True,
        )
        # 只取新生成段
        generated_ids = out.sequences[:, vlm_inputs.prompt_len:]
        # out.scores: List[tensor]，每步的未归一化 logits（batch, vocab)
        # 注意：有些实现是 logits_processor 后的分布前的 "scores"，可近似看成 logits
        return generated_ids, out.scores

    def tokens_logprobs(self, scores: List[torch.Tensor], token_ids: torch.LongTensor, temp: float = 1.0) -> List[float]:
        """
        给定每步 scores（相当于 logits），和实际选择的 token 序列，计算每步的 logprob。
        支持一个“归一化温度 temp”： temp=1 => 原始分布； temp<1 => 降温（等价于 α-power）。
        """
        logps = []
        for step, logits in enumerate(scores):
            # 对应本步采样到的 token
            tok = token_ids[0, step].item()
            scaled = logits / max(1e-6, temp)
            logp = torch.log_softmax(scaled, dim=-1)[0, tok].item()
            logps.append(float(logp))
        return logps

def naive_temp_vlm(
    vlm: VLMWrapper,
    image_path: Optional[str],
    prompt_text: str,
    prefix_input_ids: Optional[List[int]],
    seq_len: int,
    proposal_T: float,
) -> Tuple[List[int], List[float], List[float]]:
    """
    用温度 T 的分布一次性生成到目标长度 seq_len，并返回
    - 全部 token 序列（含 prefix）
    - "norm"：在提议分布（温度=proposal_T）下的逐 token logprob
    - "target"：在目标分布（温度=1.0，即原始）下的逐 token logprob
    """
    # 重新构造输入：把 prefix 拼接到文本末尾；为简洁起见，我们把 prefix 直接 decode 到文本后（适合聊天风格）
    # 也可以用 tokenizer 的强拼接方式；这里两种都行
    if prefix_input_ids:
        prefix_text = vlm.tokenizer.decode(prefix_input_ids, skip_special_tokens=True)
        merged_text = prompt_text + prefix_text
    else:
        merged_text = prompt_text

    inputs = vlm.build_inputs(image_path, merged_text)

    max_new_tokens = max(0, seq_len)
    gen_ids, scores_prop = vlm.generate_with_scores(
        inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=proposal_T
    )

    # 在提议分布与目标分布下分别计算 logprob（目标 temp=1.0 视作原模型）
    lp_prop = vlm.tokens_logprobs(scores_prop, gen_ids, temp=proposal_T)
    lp_target = vlm.tokens_logprobs(scores_prop, gen_ids, temp=1.0)

    # 返回“包含 prefix”的完整 token 序列：这里直接用 prefix + 新生成 ids
    full_seq = (prefix_input_ids or []) + gen_ids[0].tolist()
    return full_seq, lp_prop, lp_target

def mcmc_power_samp_vlm(
    vlm: VLMWrapper,
    image_path: Optional[str],
    prompt_text: str,
    alpha: float,
    mcmc_steps: int,
    max_new_tokens: int,
    block_num: int = 16,
    proposal_T: float = 0.5,
) -> Tuple[List[int], float]:
    """
    与 LLM 版一致的流程；差异仅在“多模态打包 + 逐步 logprob 计算”。
    目标分布 ~ p(y|x)^alpha；采样提议 ~ softmax(logits / T)。
    MH 接受率里需要 target 和 proposal 的逐 token 对数和。
    """
    assert max_new_tokens % block_num == 0
    jump = max_new_tokens // block_num

    # 初始：直接从空 prefix 开始
    prefix = []
    log_norm: List[float] = []
    log_target: List[float] = []

    attempts = 0
    accepts = 0

    for _ in tqdm.tqdm(range(block_num), desc="Block grow"):
        # 先扩一段
        full_seq, lp_prop, lp_tgt = naive_temp_vlm(
            vlm, image_path, prompt_text, prefix, seq_len=jump, proposal_T=proposal_T
        )
        # 通过 α-power 改写“目标”对数（相当于温度 1/α）
        lp_tgt = [alpha * v for v in lp_tgt]

        # 追加缓存
        if not prefix:
            # 第一块
            log_norm = lp_prop.copy()
            log_target = lp_tgt.copy()
            prefix = full_seq
        else:
            log_norm.extend(lp_prop)
            log_target.extend(lp_tgt)
            prefix = full_seq

        # MCMC 后缀重采样
        for _ in range(mcmc_steps):
            attempts += 1
            t = len(prefix)
            # 随机切一刀（不能在用户 prompt 内，因我们是文本拼接，切点从已有 prefix 的任意位置）
            idx = random.randint(0, t - 1)

            # 以切点前缀为条件，重采整段至当前长度
            prop_full, prop_lp_prop, prop_lp_tgt = naive_temp_vlm(
                vlm, image_path, prompt_text, prefix[:idx], seq_len=t - idx, proposal_T=proposal_T
            )
            prop_lp_tgt = [alpha * v for v in prop_lp_tgt]

            # 当前后缀段
            cur_lp_prop = log_norm[idx:]
            cur_lp_tgt = log_target[idx:]

            # MH： sum(target_prop) + sum(proposal_cur) - sum(target_cur) - sum(proposal_prop)
            log_r = sum(prop_lp_tgt) + sum(cur_lp_prop) - sum(cur_lp_tgt) - sum(prop_lp_prop)
            if math.log(random.random() + 1e-12) < log_r:
                accepts += 1
                prefix = prop_full
                log_norm[idx:] = prop_lp_prop
                log_target[idx:] = prop_lp_tgt

        # 早停：遇到 eos
        if vlm.eos_token_id is not None and vlm.eos_token_id in prefix:
            eos_idx = prefix.index(vlm.eos_token_id)
            prefix = prefix[: eos_idx + 1]
            log_norm = log_norm[: max(0, eos_idx)]
            log_target = log_target[: max(0, eos_idx)]
            break

    acc_ratio = accepts / max(1, attempts)
    return prefix, acc_ratio

def get_dist_info():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] if os.environ.get("CUDA_VISIBLE_DEVICES") else "0"))
    return rank, world_size, local_rank


def init_distributed():
    rank, world_size, local_rank = get_dist_info()
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" 
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, world_size, local_rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_json", type=str, required=True, help="Path to .json/.jsonl with fields: image (optional), instruction")
    parser.add_argument("--save_dir", type=str, default="results_vlm")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the local model directory")
    parser.add_argument("--alpha", type=float, default=1.5, help="power exponent for target distribution p^alpha")
    parser.add_argument("--proposal_T", type=float, default=0.5, help="temperature of proposal distribution")
    parser.add_argument("--mcmc_steps", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--block_num", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Distributed setup
    rank, world_size, local_rank = init_distributed()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    # TODO load from .json/.jsonl
    data = load_dataset("json", data_files=args.data_json)["train"]

    # 模型：每个 rank 固定到本地 GPU
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    vlm = VLMWrapper(args.model_path, device=device)

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(
        args.save_dir,
        f"{os.path.basename(args.model_path)}_alpha{args.alpha}_T{args.proposal_T}_steps{args.mcmc_steps}_rank{rank}.jsonl",
    )
    fout = open(out_path, "w", encoding="utf-8")

    # 按全局索引做步长切分，确保 gtix 全局一致
    total = len(data)
    local_indices = list(range(rank, total, world_size))
    for k, gidx in enumerate(tqdm.tqdm(local_indices, desc=f"Sampling (rank {rank}/{world_size})")):
        ex = data[gidx]
        image_path = os.path.join(BASE_IMAGE_DIR, ex["image"].replace("data","").lstrip("/")) 
        instruction = PROMPT_TEMPLATE.format(sent=ex["sent"])

        # 仅使用 power sampling (MCMC)
        final_ids, acc = mcmc_power_samp_vlm(
            vlm,
            image_path=image_path,
            prompt_text=instruction,
            alpha=args.alpha,
            mcmc_steps=args.mcmc_steps,
            max_new_tokens=args.max_new_tokens,
            block_num=args.block_num,
            proposal_T=args.proposal_T,
        )
        mcmc_text = vlm.tokenizer.decode(torch.tensor(final_ids, dtype=torch.long), skip_special_tokens=True)

        gtix = gidx
        record = {
            "image": image_path,
            "instruction": instruction,
            "mcmc_completion": mcmc_text,
            "acceptance_ratio": acc,
            "alpha": args.alpha,
            "proposal_T": args.proposal_T,
            "gtix": int(gtix),
            "gt_bbox": ex["bbox"],
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()

    fout.close()
    print(f"[DONE] saved -> {out_path}")

    # 同步并在 rank0 汇总全局指标
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        if dist.get_rank() == 0:
            model_base = os.path.basename(args.model_path)
            agg = aggregate_metrics_across_ranks(
                save_dir=args.save_dir,
                model_basename=model_base,
                alpha=args.alpha,
                proposal_T=args.proposal_T,
                mcmc_steps=args.mcmc_steps,
            )
            agg_path = os.path.join(
                args.save_dir,
                f"{model_base}_alpha{args.alpha}_T{args.proposal_T}_steps{args.mcmc_steps}_ALL.metrics.json",
            )
            with open(agg_path, "w", encoding="utf-8") as fa:
                json.dump(agg, fa, ensure_ascii=False, indent=2)
            print("Aggregated Metrics:", json.dumps(agg, ensure_ascii=False))

if __name__ == "__main__":
    main()
