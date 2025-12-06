#!/usr/bin/env python3
"""
Test CLS-attention token pruning method.

Based on FasterVLM research:
- CLS attention from ViT is more reliable than LLM text-visual attention
- Can identify important visual tokens before LLM processing
- Key advantage: Single forward pass, no gradient computation needed

Usage:
    python experiments/test_cls_attention.py --device cuda:0
"""

import os
import sys
import json
import argparse
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

from llava_methods import cls_attention_llava, rel_attention_llava
from utils import bbox_from_att_image_adaptive, create_masked_image
from get_score import get_acc


def load_model(device):
    """Load LLaVA model."""
    model_id = "llava-hf/llava-1.5-7b-hf"
    print(f"Loading model {model_id}...")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="eager"
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def compare_attention_methods(model, processor, image, question):
    """Compare CLS attention vs rel_attention methods."""
    device = model.device

    # Get CLS attention (single forward pass through ViT)
    cls_map = cls_attention_llava(image, question, None, model, processor)

    # Get rel_attention (two forward passes through full model)
    prompt = f"<image>\nUSER: {question} Answer the question using a single word or phrase.\nASSISTANT:"
    general_prompt = f"<image>\nUSER: Write a general description of the image. Answer the question using a single word or phrase.\nASSISTANT:"
    rel_map = rel_attention_llava(image, prompt, general_prompt, model, processor)

    return cls_map, rel_map


def test_single_sample(model, processor, image_path, question, labels, method="cls_att"):
    """Test a single sample with given method."""
    device = model.device
    image = Image.open(image_path).convert("RGB")

    prompt = f"<image>\nUSER: {question} Answer the question using a single word or phrase.\nASSISTANT:"
    general_prompt = f"<image>\nUSER: Write a general description of the image. Answer the question using a single word or phrase.\nASSISTANT:"

    # Get original answer
    inputs = processor(images=image, text=prompt, return_tensors="pt", padding=True).to(device, torch.bfloat16)
    with torch.no_grad():
        ori_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    ori_answer = processor.batch_decode(ori_ids, skip_special_tokens=True)[0].split('ASSISTANT: ')[1].strip()

    # Get attention map based on method
    if method == "cls_att":
        att_map = cls_attention_llava(image, question, None, model, processor)
    else:  # rel_att
        att_map = rel_attention_llava(image, prompt, general_prompt, model, processor)

    # Create masked image
    masked_image = create_masked_image(image, att_map, threshold_percentile=40,
                                       soft_mask=True, mask_style="black")

    # Get answer with masked image
    masked_prompt = f"<image><image>\nUSER: The second image highlights the relevant region. {question} Answer the question using a single word or phrase.\nASSISTANT:"
    masked_inputs = processor(images=[image, masked_image], text=masked_prompt,
                              return_tensors="pt", padding=True).to(device, torch.bfloat16)
    with torch.no_grad():
        masked_ids = model.generate(**masked_inputs, max_new_tokens=20, do_sample=False)
    masked_answer = processor.batch_decode(masked_ids, skip_special_tokens=True)[0].split('ASSISTANT: ')[1].strip()

    # Check correctness (get_acc returns 0-100, >0 means at least partial match)
    ori_correct = get_acc(ori_answer, labels) > 0
    masked_correct = get_acc(masked_answer, labels) > 0

    return {
        "ori_answer": ori_answer,
        "masked_answer": masked_answer,
        "ori_correct": ori_correct,
        "masked_correct": masked_correct,
        "fixed": not ori_correct and masked_correct,
        "broke": ori_correct and not masked_correct
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=50)
    args = parser.parse_args()

    # Load model
    model, processor = load_model(args.device)

    # Load test data
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data/single_crop_failures.json")

    if os.path.exists(data_path):
        print(f"Loading failure cases from {data_path}")
        with open(data_path) as f:
            test_data = json.load(f)[:args.num_samples]
    else:
        # Load from textvqa if no failure file exists
        textvqa_path = os.path.join(project_root, "data/textvqa/data.json")
        print(f"Loading from {textvqa_path}")
        with open(textvqa_path) as f:
            test_data = json.load(f)[:args.num_samples]

    image_base = os.path.join(project_root, "data/textvqa/images")

    # Test both methods
    results = {"cls_att": defaultdict(int), "rel_att": defaultdict(int)}

    for method in ["cls_att", "rel_att"]:
        print(f"\n{'='*60}")
        print(f"Testing {method.upper()} method")
        print(f"{'='*60}")

        for sample in tqdm(test_data, desc=f"Testing {method}"):
            image_path = os.path.join(image_base, sample.get("image_path", f"{sample.get('image_id', sample.get('id'))}.jpg"))

            if not os.path.exists(image_path):
                continue

            try:
                result = test_single_sample(
                    model, processor, image_path,
                    sample["question"], sample["labels"], method
                )

                results[method]["total"] += 1
                results[method]["ori_correct"] += result["ori_correct"]
                results[method]["masked_correct"] += result["masked_correct"]
                results[method]["fixed"] += result["fixed"]
                results[method]["broke"] += result["broke"]

            except Exception as e:
                print(f"Error: {e}")
                continue

        # Print results for this method
        total = results[method]["total"]
        if total > 0:
            print(f"\n{method} Results ({total} samples):")
            print(f"  Original accuracy: {results[method]['ori_correct']/total*100:.1f}%")
            print(f"  Masked accuracy:   {results[method]['masked_correct']/total*100:.1f}%")
            print(f"  Fixed:             {results[method]['fixed']} ({results[method]['fixed']/total*100:.1f}%)")
            print(f"  Broke:             {results[method]['broke']} ({results[method]['broke']/total*100:.1f}%)")

    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    for method in ["cls_att", "rel_att"]:
        total = results[method]["total"]
        if total > 0:
            net_gain = results[method]["fixed"] - results[method]["broke"]
            print(f"{method}: Fixed {results[method]['fixed']}, Broke {results[method]['broke']}, Net gain: {net_gain}")


if __name__ == "__main__":
    main()
