#!/usr/bin/env python3
"""
Unified experiment runner for testing different cropping strategies.

Usage:
    python experiments/run_experiment.py --mode compare --samples 100
    python experiments/run_experiment.py --mode masked --samples 50
    python experiments/run_experiment.py --mode visualize --samples 20
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch
import gc
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
from tqdm import tqdm

from run import vicrop_qa
from info import model_to_fullname
from utils import create_masked_image, get_multi_crop_bboxes, bbox_from_att_image_adaptive
from llava_methods import rel_attention_llava


def evaluate_answer(pred, labels):
    """Check if prediction matches any label."""
    if pred is None:
        return False
    pred = pred.lower().strip()
    for label in labels:
        if label.lower() in pred or pred in label.lower():
            return True
    return False


def load_model(device="cuda:2"):
    """Load LLaVA model."""
    print(f"Loading LLaVA model on {device}...")
    model_id = model_to_fullname["llava"]
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="eager"
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor


def run_comparison(model, processor, data, modes, output_file="comparison_results.json"):
    """Compare different crop modes on the same data."""
    results = {mode: {"correct": 0, "total": 0, "answers": []} for mode in modes}

    for i, item in enumerate(tqdm(data, desc="Comparing modes")):
        image_path = os.path.join("./data/textvqa/images", item['image_path'])
        if not os.path.exists(image_path):
            image_path = item.get('image_path', image_path)

        question = item['question']
        labels = item['labels']
        short_question = item.get('short_question', question)

        for mode in modes:
            try:
                result = vicrop_qa(
                    "llava", "rel_att", image_path, question,
                    model, processor, short_question, crop_mode=mode
                )
                answer = result.get("crop_answer", "")
                correct = evaluate_answer(answer, labels)

                results[mode]["total"] += 1
                if correct:
                    results[mode]["correct"] += 1
                results[mode]["answers"].append({
                    "idx": i,
                    "question": question[:50],
                    "answer": answer,
                    "correct": correct
                })
            except Exception as e:
                print(f"\nError with {mode} on sample {i}: {e}")
                results[mode]["total"] += 1

            gc.collect()
            torch.cuda.empty_cache()

        # Progress update
        if (i + 1) % 20 == 0:
            print(f"\n--- Progress at {i+1}/{len(data)} ---")
            for mode in modes:
                acc = results[mode]["correct"] / max(results[mode]["total"], 1) * 100
                print(f"  {mode}: {results[mode]['correct']}/{results[mode]['total']} ({acc:.1f}%)")

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    return results


def print_summary(results, modes):
    """Print summary of results."""
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")

    for mode in modes:
        total = results[mode]["total"]
        correct = results[mode]["correct"]
        acc = correct / max(total, 1) * 100
        print(f"{mode:20}: {correct:3}/{total:3} ({acc:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Run cropping strategy experiments")
    parser.add_argument("--mode", choices=["compare", "masked", "multicrop"],
                        default="compare", help="Experiment mode")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples")
    parser.add_argument("--device", default="cuda:2", help="GPU device")
    parser.add_argument("--data", default="./data/textvqa/data.json", help="Data file")
    parser.add_argument("--output", default="experiment_results.json", help="Output file")
    args = parser.parse_args()

    # Load data
    with open(args.data) as f:
        data = json.load(f)[:args.samples]

    # Load model
    model, processor = load_model(args.device)

    # Run experiment based on mode
    if args.mode == "compare":
        modes = ["no_crop", "single_crop", "masked_black"]
        results = run_comparison(model, processor, data, modes, args.output)
        print_summary(results, modes)

    elif args.mode == "masked":
        modes = ["single_crop", "masked_black", "masked_blur", "masked_dim"]
        results = run_comparison(model, processor, data, modes, args.output)
        print_summary(results, modes)

    elif args.mode == "multicrop":
        modes = ["single_crop", "smart_multi_crop"]
        results = run_comparison(model, processor, data, modes, args.output)
        print_summary(results, modes)


if __name__ == "__main__":
    main()
