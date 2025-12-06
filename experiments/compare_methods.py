#!/usr/bin/env python3
"""
Compare no_crop vs single_crop vs zoom_refine on TextVQA.
"""

import os
import json
import torch
import gc
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
from tqdm import tqdm

from run import vicrop_qa
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


def main(num_samples=100):
    device = "cuda:2"
    print(f"Loading LLaVA model on {device}...")

    model_id = model_to_fullname["llava"]
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="eager"
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()

    # Load data
    with open("./data/textvqa/data.json") as f:
        data = json.load(f)[:num_samples]

    print(f"\n{'='*70}")
    print(f"Comparing: NO_CROP vs SINGLE_CROP vs ZOOM_REFINE")
    print(f"Samples: {num_samples}")
    print(f"{'='*70}\n")

    results = {
        "no_crop": {"correct": 0, "total": 0, "answers": []},
        "single_crop": {"correct": 0, "total": 0, "answers": []},
        "zoom_refine": {"correct": 0, "total": 0, "answers": []}
    }

    for i, item in enumerate(tqdm(data, desc="Processing")):
        image_path = os.path.join("./data/textvqa/images", item['image_path'])
        question = item['question']
        labels = item['labels']
        short_question = item.get('short_question', question)

        for mode in ["no_crop", "single_crop", "zoom_refine"]:
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
                    "question": question[:50],
                    "answer": answer,
                    "labels": labels[:2],
                    "correct": correct
                })
            except Exception as e:
                print(f"\nError with {mode} on sample {i}: {e}")
                results[mode]["total"] += 1
                results[mode]["answers"].append({
                    "question": question[:50],
                    "answer": "ERROR",
                    "labels": labels[:2],
                    "correct": False
                })

            gc.collect()
            torch.cuda.empty_cache()

        # Print progress every 20 samples
        if (i + 1) % 20 == 0:
            print(f"\n--- Progress at {i+1}/{num_samples} ---")
            for mode in results:
                acc = results[mode]["correct"] / max(results[mode]["total"], 1) * 100
                print(f"  {mode}: {results[mode]['correct']}/{results[mode]['total']} ({acc:.1f}%)")

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")

    for mode in ["no_crop", "single_crop", "zoom_refine"]:
        acc = results[mode]["correct"] / max(results[mode]["total"], 1) * 100
        print(f"{mode:15}: {results[mode]['correct']:3}/{results[mode]['total']:3} ({acc:.1f}%)")

    # Head-to-head comparison
    print(f"\n{'='*70}")
    print("HEAD-TO-HEAD COMPARISON")
    print(f"{'='*70}")

    single_wins = zoom_wins = ties = 0
    for i in range(len(results["single_crop"]["answers"])):
        sc = results["single_crop"]["answers"][i]["correct"]
        zr = results["zoom_refine"]["answers"][i]["correct"]
        if sc and not zr:
            single_wins += 1
        elif zr and not sc:
            zoom_wins += 1
        else:
            ties += 1

    print(f"SINGLE_CROP wins: {single_wins}")
    print(f"ZOOM_REFINE wins: {zoom_wins}")
    print(f"Ties: {ties}")
    print(f"Net advantage: {'ZOOM_REFINE' if zoom_wins > single_wins else 'SINGLE_CROP'} (+{abs(zoom_wins - single_wins)})")

    # Save results
    with open("comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to comparison_results.json")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    main(num_samples=n)
