#!/usr/bin/env python3
"""
RefCOCO Evaluation Script for Qwen3-VL
Evaluates Qwen3-VL models on RefCOCO/RefCOCO+/RefCOCOg datasets
Supports multi-GPU distributed evaluation
"""
import argparse
import itertools
import json
import os
import re
import time
from functools import partial
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from torchvision.ops.boxes import box_area

# Dataset paths - RefCOCO standard 8 splits (absolute paths)
Base_dir="/storage/openpsi/data/grounding_sft_v1/"
DS_COLLECTIONS = {
    # RefCOCO
    'refcoco_testA': Base_dir+'refcoco_testA.jsonl',
    'refcoco_testB': Base_dir+'/refcoco_testB.jsonl',
    'refcoco_val': Base_dir+'refcoco_val.jsonl',
    # RefCOCO+
    'refcoco+_val': Base_dir+'refcoco+_val.jsonl',
    'refcoco+_testA': Base_dir+'refcoco+_testA.jsonl',
    'refcoco+_testB': Base_dir+'refcoco+_testB.jsonl',
    # RefCOCOg
    'refcocog_val': Base_dir+'refcocog_val.jsonl',
    'refcocog_test': Base_dir+'refcocog_test.jsonl',
}

# Base directory for COCO images
COCO_IMAGE_ROOT = '/storage/openpsi/'

# Default prompt template for grounding task (official format)
DEFAULT_PROMPT = "Locate {}, output its bbox coordinates using JSON format."


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate IoU between two sets of boxes"""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou, union


def parse_bbox_from_response(response: str) -> Tuple[float, float, float, float]:
    """
    Parse bounding box coordinates from model response
    Supports multiple formats including Thinking model output
    
    For Thinking models, bbox is usually after </think> tag
    """
    # For Thinking models: extract content after </think> if present
    if '</think>' in response:
        # Get the part after the last </think>
        after_think = response.split('</think>')[-1]
        
        # Try to find bbox in the content after </think>
        # Pattern 1: [x1, y1, x2, y2] format
        pattern_list = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
        matches = re.findall(pattern_list, after_think)
        if matches:
            try:
                # Take the last match (most likely the final answer)
                x1, y1, x2, y2 = map(float, matches[-1])
                return (x1, y1, x2, y2)
            except:
                pass
        
        # Pattern 2: JSON format {"bbox_2d": [x1, y1, x2, y2]}
        pattern_json = r'"bbox_2d"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
        matches = re.findall(pattern_json, after_think)
        if matches:
            try:
                x1, y1, x2, y2 = map(float, matches[-1])
                return (x1, y1, x2, y2)
            except:
                pass
    
    # Fallback: search in the whole response
    # Pattern 1: <box>[[x1,y1,x2,y2]]</box>
    pattern1 = r'<box>\[\[(\d+),(\d+),(\d+),(\d+)\]\]</box>'
    # Pattern 2: [[x1,y1,x2,y2]]
    pattern2 = r'\[\[(\d+),(\d+),(\d+),(\d+)\]\]'
    # Pattern 3: JSON format {"bbox_2d": [x1, y1, x2, y2]}
    pattern3 = r'"bbox_2d"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    # Pattern 4: Simple list [x1, y1, x2, y2]
    pattern4 = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    
    for pattern in [pattern1, pattern2, pattern3, pattern4]:
        matches = re.findall(pattern, response)
        if matches:
            try:
                # Take the last match
                x1, y1, x2, y2 = map(float, matches[-1])
                return (x1, y1, x2, y2)
            except:
                continue
    
    # If no valid bbox found, return zeros
    return (0.0, 0.0, 0.0, 0.0)


class InferenceSampler(torch.utils.data.sampler.Sampler):
    """Sampler for distributed inference"""
    
    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


class RefCOCODataset(torch.utils.data.Dataset):
    """Dataset for RefCOCO evaluation"""
    
    def __init__(self, data_path: str, prompt_template: str, max_samples: int = None):
        """
        Args:
            data_path: Path to JSONL file
            prompt_template: Prompt template with {} for text placeholder
            max_samples: Maximum number of samples to evaluate (None for all)
        """
        with open(data_path, 'r') as f:
            self.data = [json.loads(line.strip()) for line in f]
        
        if max_samples is not None and max_samples > 0:
            self.data = self.data[:max_samples]
            if dist.get_rank() == 0:
                print(f"[INFO] Limited to first {max_samples} samples")
        
        self.prompt_template = prompt_template
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        # Handle image path - convert relative to absolute if needed
        image_path = item['image']
        if not os.path.isabs(image_path):
            # Image path is relative, prepend COCO_IMAGE_ROOT
            image_path = os.path.join(COCO_IMAGE_ROOT, image_path)
        
        return {
            'image_path': image_path,
            'text': item['sent'],
            'bbox': item['bbox'],  # [x1, y1, x2, y2] in pixel coordinates (InternVL format)
            'width': item['width'],
            'height': item['height'],
            'prompt': self.prompt_template.format(item['sent'])
        }


def collate_fn(batches):
    """Collate function for dataloader"""
    return batches


def evaluate_model(args):
    """Main evaluation function with distributed support"""
    
    # Initialize distributed
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Setup output directory
    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
    
    # Load model and processor
    if rank == 0:
        print(f"Loading model: {args.model_path}")
        print(f"World size: {world_size}, Rank: {rank}")
    
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": f"cuda:{rank}"},
        trust_remote_code=True
    )
    
    # Load processor from specified path or fallback to model path
    processor_path = args.processor_path if args.processor_path else args.model_path
    if rank == 0 and args.processor_path:
        print(f"Loading processor from: {processor_path}")
    
    try:
        processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    except Exception as e:
        if rank == 0:
            print(f"[WARNING] Failed to load processor from {processor_path}: {e}")
            print(f"[INFO] Trying to infer base model from checkpoint...")
        
        # Try to infer base model from checkpoint name
        # e.g., "2b_thinking_grounding_sft" -> try Qwen3-VL-2B-Thinking
        checkpoint_name = args.model_path.split('/')[-2] if '/' in args.model_path else args.model_path
        
        # Map common patterns
        base_model_map = {
            '2b_instruct': 'Qwen/Qwen3-VL-2B-Instruct',
            '2b_thinking': 'Qwen/Qwen3-VL-2B-Thinking',
            '4b_instruct': 'Qwen/Qwen3-VL-4B-Instruct',
            '4b_thinking': 'Qwen/Qwen3-VL-4B-Thinking',
            '8b_instruct': 'Qwen/Qwen3-VL-8B-Instruct',
            '8b_thinking': 'Qwen/Qwen3-VL-8B-Thinking',
        }
        
        # Try to find matching base model
        base_model = None
        for key, value in base_model_map.items():
            if key in checkpoint_name.lower():
                base_model = value
                break
        
        if base_model:
            if rank == 0:
                print(f"[INFO] Inferred base model: {base_model}")
                print(f"[INFO] Loading processor from base model...")
            processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        else:
            raise RuntimeError(f"Could not load processor and failed to infer base model from '{checkpoint_name}'. "
                             f"Please specify --processor-path with the base model name (e.g., Qwen/Qwen3-VL-2B-Thinking)")
    
    # Evaluation results summary
    all_results = []
    
    # Evaluate each dataset
    for ds_name in args.datasets:
        if ds_name not in DS_COLLECTIONS:
            if rank == 0:
                print(f"[WARNING] Unknown dataset: {ds_name}, skipping...")
            continue
        
        ds_path = DS_COLLECTIONS[ds_name]
        if not os.path.exists(ds_path):
            if rank == 0:
                print(f"[WARNING] Dataset file not found: {ds_path}, skipping...")
            continue
        
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Evaluating {ds_name}")
            print(f"{'='*60}")
        
        # Load dataset
        dataset = RefCOCODataset(
            data_path=ds_path,
            prompt_template=args.prompt,
            max_samples=args.max_samples
        )
        
        # Create distributed dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
        
        # Run inference
        predictions = []
        
        for batch in tqdm(dataloader, desc=f"[Rank {rank}] {ds_name}", disable=(rank != 0)):
            # Process batch (support batch_size > 1)
            batch_samples = batch if isinstance(batch, list) else [batch]
            
            for sample in batch_samples:
                # Load image
                try:
                    image = Image.open(sample['image_path']).convert('RGB')
                except Exception as e:
                    if rank == 0:
                        print(f"[ERROR] Failed to load image {sample['image_path']}: {e}")
                    continue
                
                # Prepare messages for Qwen3-VL
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": sample['prompt']}
                        ]
                    }
                ]
                
                # Generate prediction
                try:
                    inputs = processor.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt"
                    )
                    inputs = inputs.to(model.device)
                    
                    with torch.no_grad():
                        generated_ids = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=args.temperature > 0,
                            temperature=args.temperature if args.temperature > 0 else 1.0,
                            top_p=args.top_p,
                        )
                    
                    # Decode response
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] 
                        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    
                    # Calculate number of generated tokens
                    num_tokens = len(generated_ids_trimmed[0])
                    
                    response = processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )[0]
                    
                except Exception as e:
                    if rank == 0:
                        print(f"[ERROR] Generation failed: {e}")
                    response = ""
                    num_tokens = 0
                
                # Parse predicted bbox
                pred_bbox = parse_bbox_from_response(response)
                
                # Ground truth bbox is already in [x1, y1, x2, y2] pixel format
                gt_bbox = sample['bbox']
                
                # Convert predicted bbox from 0-1000 scale to pixel coordinates
                # Qwen3-VL uses relative coordinates in 0-1000 range
                pred_bbox_pixels = [
                    pred_bbox[0] / 1000 * sample['width'],
                    pred_bbox[1] / 1000 * sample['height'],
                    pred_bbox[2] / 1000 * sample['width'],
                    pred_bbox[3] / 1000 * sample['height']
                ]
                
                # Calculate IoU
                pred_tensor = torch.tensor(pred_bbox_pixels, dtype=torch.float32).view(-1, 4)
                gt_tensor = torch.tensor(gt_bbox, dtype=torch.float32).view(-1, 4)
                
                iou, _ = box_iou(pred_tensor, gt_tensor)
                iou_value = iou.item()
                
                # Check if correct (IoU >= 0.5)
                is_correct = iou_value >= 0.5
                
                # Save prediction
                predictions.append({
                    'image_path': sample['image_path'],
                    'text': sample['text'],
                    'prompt': sample['prompt'],
                    'response': response,
                    'num_tokens': num_tokens,
                    'pred_bbox': list(pred_bbox),
                    'pred_bbox_pixels': pred_bbox_pixels,
                    'gt_bbox': gt_bbox,
                    'hw': (sample['height'], sample['width']),
                    'iou': iou_value,
                    'correct': is_correct
                })
        
        # Gather results from all ranks
        dist.barrier()
        
        world_size = dist.get_world_size()
        merged_predictions = [None for _ in range(world_size)]
        dist.all_gather_object(merged_predictions, predictions)
        
        merged_predictions = [_ for _ in itertools.chain.from_iterable(merged_predictions)]
        
        # Calculate accuracy and average tokens on rank 0
        if rank == 0:
            correct = sum(p['correct'] for p in merged_predictions)
            total = len(merged_predictions)
            accuracy = correct / total if total > 0 else 0.0
            
            # Calculate average number of tokens
            total_tokens = sum(p['num_tokens'] for p in merged_predictions)
            avg_tokens = total_tokens / total if total > 0 else 0.0
            
            # Save detailed results
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            model_name = args.model_path.split('/')[-1]
            results_file = os.path.join(
                args.out_dir,
                f"{model_name}_{ds_name}_{timestamp}.json"
            )
            
            result_summary = {
                'model': args.model_path,
                'dataset': ds_name,
                'total_samples': total,
                'correct': correct,
                'accuracy': accuracy,
                'avg_tokens': avg_tokens,
                'total_tokens': total_tokens,
                'timestamp': timestamp,
                'world_size': world_size,
                'args': vars(args),
                'predictions': merged_predictions if args.save_predictions else []
            }
            
            with open(results_file, 'w') as f:
                json.dump(result_summary, f, indent=2)
            
            print(f"\n{ds_name} Results:")
            print(f"  Total samples: {total}")
            print(f"  Correct: {correct}")
            print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  Avg tokens: {avg_tokens:.2f}")
            print(f"  Total tokens: {total_tokens}")
            print(f"  Results saved to: {results_file}")
            
            all_results.append({
                'dataset': ds_name,
                'total': total,
                'correct': correct,
                'accuracy': accuracy,
                'avg_tokens': avg_tokens,
                'total_tokens': total_tokens
            })
        
        dist.barrier()
    
    # Save summary on rank 0
    if rank == 0:
        summary_file = os.path.join(args.out_dir, f"{model_name}_summary_{timestamp}.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Evaluation Time: {timestamp}\n")
            f.write(f"World Size: {world_size}\n")
            f.write(f"{'='*60}\n\n")
            
            for result in all_results:
                f.write(f"{result['dataset']:20s}: Acc={result['accuracy']:.4f} ({result['accuracy']*100:.2f}%) "
                       f"[{result['correct']}/{result['total']}] "
                       f"AvgTokens={result['avg_tokens']:.2f} "
                       f"TotalTokens={result['total_tokens']}\n")
            
            if len(all_results) > 1:
                avg_acc = sum(r['accuracy'] for r in all_results) / len(all_results)
                overall_avg_tokens = sum(r['avg_tokens'] for r in all_results) / len(all_results)
                overall_total_tokens = sum(r['total_tokens'] for r in all_results)
                f.write(f"\n{'='*60}\n")
                f.write(f"Average Accuracy: {avg_acc:.4f} ({avg_acc*100:.2f}%)\n")
                f.write(f"Average Tokens per Sample (across all splits): {overall_avg_tokens:.2f}\n")
                f.write(f"Total Tokens Generated: {overall_total_tokens}\n")
        
        print(f"\n{'='*60}")
        print(f"Summary saved to: {summary_file}")
        print(f"{'='*60}\n")
        
        # Print final summary
        print("Final Results:")
        for result in all_results:
            print(f"  {result['dataset']:20s}: Acc={result['accuracy']:.4f} ({result['accuracy']*100:.2f}%) "
                 f"AvgTokens={result['avg_tokens']:.2f}")
        
        if len(all_results) > 1:
            avg_acc = sum(r['accuracy'] for r in all_results) / len(all_results)
            overall_avg_tokens = sum(r['avg_tokens'] for r in all_results) / len(all_results)
            overall_total_tokens = sum(r['total_tokens'] for r in all_results)
            print(f"\nAverage Accuracy: {avg_acc:.4f} ({avg_acc*100:.2f}%)")
            print(f"Average Tokens per Sample: {overall_avg_tokens:.2f}")
            print(f"Total Tokens Generated: {overall_total_tokens}")
    
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Qwen3-VL on RefCOCO datasets')
    
    # Model arguments
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path or name of Qwen3-VL model (e.g., Qwen/Qwen3-VL-2B-Instruct)')
    parser.add_argument('--processor-path', type=str, default=None,
                       help='Path to load processor from (for SFT checkpoints that lack processor config). If not specified, uses --model-path')
    
    # Data arguments
    parser.add_argument('--data-root', type=str, 
                       default='',
                       help='Root directory containing RefCOCO data (not used, absolute paths are used)')
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['refcoco_val', 'refcoco_testA', 'refcoco_testB',
                               'refcoco+_val', 'refcoco+_testA', 'refcoco+_testB',
                               'refcocog_val', 'refcocog_test'],
                       help='Datasets to evaluate')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to evaluate per dataset (None for all)')
    
    # Generation arguments
    parser.add_argument('--prompt', type=str, 
                       default=DEFAULT_PROMPT,
                       help='Prompt template with {} for text')
    parser.add_argument('--max-new-tokens', type=int, default=None,
                       help='Maximum number of tokens to generate (auto: 512 for Instruct, 1024 for Thinking)')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Sampling temperature (0 for greedy)')
    parser.add_argument('--top-p', type=float, default=0.9,
                       help='Top-p sampling parameter')
    
    # Output arguments
    parser.add_argument('--out-dir', type=str, 
                       default='./evaluation/refcoco/results',
                       help='Output directory for results')
    parser.add_argument('--save-predictions', action='store_true',
                       help='Save detailed predictions in output JSON')
    
    # Batch size argument
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for inference (default: 1)')
    
    args = parser.parse_args()
    
    # Auto-adjust parameters for Thinking models
    is_thinking_model = 'thinking' in args.model_path.lower()
    
    if args.max_new_tokens is None:
        # Auto set based on model type
        args.max_new_tokens = 1024 if is_thinking_model else 512
    
    if is_thinking_model:
        print("="*60)
        print("[INFO] Detected Thinking model - Auto-adjusted parameters:")
        print(f"  max_new_tokens: {args.max_new_tokens} (Thinking models generate long outputs)")
        print(f"  Recommendation: Use smaller batch_size (1-4) for better performance")
        print("="*60)
    
    print("="*60)
    print("Qwen3-VL RefCOCO Evaluation")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Max samples per dataset: {args.max_samples if args.max_samples else 'All'}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {args.out_dir}")
    print("="*60)
    
    evaluate_model(args)


if __name__ == '__main__':
    main()
