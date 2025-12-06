#!/usr/bin/env python3
"""
Analyze when multi-crop could beat single-crop.

Hypothesis: Single-crop fails when attention focuses on WRONG region.
Multi-crop might help by covering MORE regions.

Strategy: For single-crop failures, try different multi-crop approaches:
1. Top-K attention peaks (not just top-1)
2. Different window sizes
3. Grid-based sampling
"""

import os
import json
import torch
import gc
import numpy as np
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
from tqdm import tqdm

from utils import bbox_from_att_image_adaptive, get_multi_crop_bboxes
from llava_methods import rel_attention_llava
from info import model_to_fullname


def evaluate_answer(pred, labels):
    """Check if prediction matches any label."""
    if pred is None:
        return False
    pred = pred.lower().strip()
    for label in labels:
        if label.lower() in pred or pred in label.lower():
            return True
    return False


def get_answer_with_crops(image, crops, question, model, processor):
    """Get answer using original image + crop images."""
    all_images = [image] + crops
    multi_prompt = "<image>" * len(all_images) + f"\nUSER: {question} Answer the question using a single word or phrase.\nASSISTANT:"

    inputs = processor(images=all_images, text=multi_prompt, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)

    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)

    response = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    answer = response.split('ASSISTANT:')[1].strip() if 'ASSISTANT:' in response else response.strip()

    del inputs, gen_ids
    torch.cuda.empty_cache()

    return answer


def main(num_samples=50):
    device = "cuda:2"
    print(f"Loading LLaVA model on {device}...")

    model_id = model_to_fullname["llava"]
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="eager"
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()

    # Load single_crop failures
    with open("data/single_crop_failures.json") as f:
        failures = json.load(f)[:num_samples]

    print(f"\n{'='*70}")
    print(f"Analyzing {len(failures)} single_crop failure cases")
    print(f"Goal: Find cases where multi-crop beats single-crop")
    print(f"{'='*70}\n")

    bbox_size = 336
    results = []

    # Strategies to test
    strategies = {
        "top2_crops": 2,  # Top 2 attention peaks
        "top3_crops": 3,  # Top 3 attention peaks
    }

    fixed_by = {s: 0 for s in strategies}
    fixed_by["any_multicrop"] = 0

    for i, item in enumerate(tqdm(failures, desc="Analyzing")):
        image_path = item['image_path']
        image = Image.open(image_path).convert("RGB")
        question = item['question']
        labels = item['labels']

        # Get attention map
        short_prompt = f"<image>\nUSER: {question} Answer the question using a single word or phrase.\nASSISTANT:"
        general_prompt = f"<image>\nUSER: Write a general description of the image.\nASSISTANT:"

        att_map = rel_attention_llava(image, short_prompt, general_prompt, model, processor)

        result = {
            "question": question[:60],
            "labels": labels[:2],
            "single_crop_answer": item['crop_answer'],
            "strategies": {}
        }

        any_fixed = False

        # Try each multi-crop strategy
        for strategy_name, num_crops in strategies.items():
            try:
                # Get top-K bounding boxes from attention
                bboxes_with_scores = get_multi_crop_bboxes(att_map, image.size, num_crops, bbox_size)
                bboxes = [bbox for bbox, score in bboxes_with_scores]
                crops = [image.crop(bbox) for bbox in bboxes]

                answer = get_answer_with_crops(image, crops, question, model, processor)
                correct = evaluate_answer(answer, labels)

                result["strategies"][strategy_name] = {
                    "answer": answer,
                    "correct": correct,
                    "num_crops": len(crops)
                }

                if correct:
                    fixed_by[strategy_name] += 1
                    any_fixed = True

            except Exception as e:
                result["strategies"][strategy_name] = {"error": str(e)}

        if any_fixed:
            fixed_by["any_multicrop"] += 1

        results.append(result)

        gc.collect()
        torch.cuda.empty_cache()

        # Progress update
        if (i + 1) % 10 == 0:
            print(f"\n--- Progress {i+1}/{len(failures)} ---")
            for s, count in fixed_by.items():
                print(f"  {s}: {count} fixed ({100*count/(i+1):.1f}%)")

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Total single_crop failures analyzed: {len(results)}")
    print()

    for strategy, count in fixed_by.items():
        pct = 100 * count / len(results)
        print(f"{strategy:20}: {count:3} fixed ({pct:.1f}%)")

    # Show example fixes
    print(f"\n{'='*70}")
    print("EXAMPLE CASES WHERE MULTI-CROP FIXED THE ANSWER")
    print(f"{'='*70}")

    shown = 0
    for r in results:
        for strat, data in r["strategies"].items():
            if data.get("correct"):
                print(f"\nQ: {r['question']}...")
                print(f"Labels: {r['labels']}")
                print(f"Single-crop (WRONG): '{r['single_crop_answer']}'")
                print(f"{strat} (FIXED): '{data['answer']}'")
                shown += 1
                if shown >= 5:
                    break
        if shown >= 5:
            break

    # Save results
    with open("multicrop_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to multicrop_analysis.json")


if __name__ == "__main__":
    main(num_samples=50)
