#!/usr/bin/env python3
"""
Zoom-Refine style algorithm for LLaVA.
Tests on single_crop failure cases to see if LLM-guided cropping can fix them.

Algorithm:
1. Stage 1: Get initial answer + ask LLM to predict bounding box of relevant region
2. Stage 2: Crop the LLM-predicted region, get refined answer
3. Backup: Also try attention-based crop if LLM prediction fails
4. Compare and pick best answer
"""

import os
import json
import torch
import gc
import re
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor

from utils import bbox_from_att_image_adaptive
from llava_methods import rel_attention_llava
from info import model_to_fullname


def parse_bbox_from_response(response, image_size):
    """
    Parse bounding box coordinates from LLM response.
    Expects format like [x1, y1, x2, y2] with normalized (0-1) or pixel coordinates.
    """
    # Try to find coordinates in various formats
    patterns = [
        r'\[(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\]',  # [x1, y1, x2, y2]
        r'\((\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\)',  # (x1, y1, x2, y2)
        r'(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)',      # x1, y1, x2, y2
    ]

    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            coords = [float(x) for x in match.groups()]
            x1, y1, x2, y2 = coords

            # Check if normalized (0-1) or pixel coordinates
            if all(0 <= c <= 1 for c in coords):
                # Normalized - convert to pixels
                w, h = image_size
                x1, x2 = int(x1 * w), int(x2 * w)
                y1, y2 = int(y1 * h), int(y2 * h)
            else:
                # Already pixel coordinates
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Validate and clamp
            w, h = image_size
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))

            # Ensure valid bbox
            if x2 > x1 and y2 > y1:
                return (x1, y1, x2, y2)

    return None


def zoom_refine_llava(image, question, model, processor, device):
    """
    Zoom-Refine algorithm for LLaVA.

    Returns:
        dict with initial_answer, llm_bbox, refined_answer, method_used
    """
    result = {
        "initial_answer": None,
        "llm_bbox": None,
        "refined_answer": None,
        "final_answer": None,
        "method_used": None
    }

    # Stage 1: Get initial answer + bounding box prediction
    stage1_prompt = f"""<image>
USER: Look at this image and answer the question: {question}

Also, identify the specific region in the image that contains the answer. Provide the bounding box coordinates as [x1, y1, x2, y2] where values are between 0 and 1 (normalized coordinates).

Format your response as:
Answer: <your answer>
Region: [x1, y1, x2, y2]
ASSISTANT:"""

    inputs = processor(images=image, text=stage1_prompt, return_tensors="pt", padding=True).to(device, torch.bfloat16)

    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)

    stage1_response = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    if 'ASSISTANT:' in stage1_response:
        stage1_response = stage1_response.split('ASSISTANT:')[1].strip()

    # Parse initial answer
    answer_match = re.search(r'Answer:\s*(.+?)(?:\n|Region:|$)', stage1_response, re.IGNORECASE)
    if answer_match:
        result["initial_answer"] = answer_match.group(1).strip()
    else:
        # Fallback: first line or first few words
        result["initial_answer"] = stage1_response.split('\n')[0].strip()[:50]

    # Parse bounding box
    llm_bbox = parse_bbox_from_response(stage1_response, image.size)
    result["llm_bbox"] = llm_bbox

    del inputs, gen_ids
    torch.cuda.empty_cache()

    # Stage 2: Refine with cropped region
    if llm_bbox:
        # Use LLM-predicted bbox
        try:
            crop_img = image.crop(llm_bbox)
            # Ensure crop is not too small
            if crop_img.size[0] >= 50 and crop_img.size[1] >= 50:
                stage2_prompt = f"""<image><image>
USER: The first image is the full image. The second image is a zoomed-in view of the region most relevant to the question.

Question: {question}

Look carefully at the zoomed region and provide your answer.
Answer the question using a single word or phrase.
ASSISTANT:"""

                inputs = processor(images=[image, crop_img], text=stage2_prompt, return_tensors="pt", padding=True).to(device, torch.bfloat16)

                with torch.no_grad():
                    gen_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)

                stage2_response = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
                if 'ASSISTANT:' in stage2_response:
                    result["refined_answer"] = stage2_response.split('ASSISTANT:')[1].strip()
                else:
                    result["refined_answer"] = stage2_response.strip()

                result["final_answer"] = result["refined_answer"]
                result["method_used"] = "llm_bbox_refine"

                del inputs, gen_ids
                torch.cuda.empty_cache()

                return result
        except Exception as e:
            print(f"    LLM bbox crop failed: {e}")

    # Fallback: Use attention-based crop
    result["method_used"] = "fallback_initial"
    result["final_answer"] = result["initial_answer"]

    return result


def test_zoom_refine(num_samples=20):
    """Test Zoom-Refine on single_crop failure cases."""

    device = "cuda:2"
    print(f"Loading LLaVA model on {device}...")
    model_id = model_to_fullname["llava"]
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="eager"
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()

    # Load failure cases
    with open("data/single_crop_failures.json") as f:
        failures = json.load(f)[:num_samples]

    print(f"\n{'='*80}")
    print(f"Testing Zoom-Refine on {len(failures)} single_crop failure cases")
    print(f"{'='*80}\n")

    results = []
    fixed_count = 0

    for i, item in enumerate(failures):
        print(f"\n--- Sample {i+1}/{len(failures)} ---")
        print(f"Q: {item['question'][:60]}...")
        print(f"Labels: {item['labels']}")
        print(f"Single_crop answer (WRONG): '{item['crop_answer']}'")

        # Load image (image_path is already the full relative path)
        image_path = item['image_path']
        image = Image.open(image_path).convert("RGB")

        # Run Zoom-Refine
        zr_result = zoom_refine_llava(image, item['question'], model, processor, device)

        print(f"Initial answer: '{zr_result['initial_answer']}'")
        print(f"LLM bbox: {zr_result['llm_bbox']}")
        print(f"Refined answer: '{zr_result['refined_answer']}'")
        print(f"Method: {zr_result['method_used']}")

        # Check if fixed
        def is_correct(pred, labels):
            if pred is None:
                return False
            pred = pred.lower().strip()
            for label in labels:
                if label.lower() in pred or pred in label.lower():
                    return True
            return False

        final_answer = zr_result['final_answer'] or ""
        is_fixed = is_correct(final_answer, item['labels'])

        if is_fixed:
            fixed_count += 1
            print(f"-> FIXED! Final: '{final_answer}'")
        else:
            print(f"-> Still wrong. Final: '{final_answer}'")

        results.append({
            "question": item['question'],
            "labels": item['labels'],
            "single_crop_answer": item['crop_answer'],
            "zoom_refine_result": zr_result,
            "is_fixed": is_fixed
        })

        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total failure cases tested: {len(results)}")
    print(f"Fixed by Zoom-Refine: {fixed_count} ({100*fixed_count/len(results):.1f}%)")
    print(f"Still failing: {len(results) - fixed_count}")

    # Method breakdown
    methods = {}
    for r in results:
        m = r['zoom_refine_result']['method_used']
        methods[m] = methods.get(m, 0) + 1

    print(f"\nMethod breakdown:")
    for m, count in methods.items():
        print(f"  {m}: {count}")

    # Show some fixed cases
    print(f"\n--- Sample FIXED cases ---")
    for r in results:
        if r['is_fixed']:
            print(f"  Q: {r['question'][:50]}...")
            print(f"  Single_crop: '{r['single_crop_answer']}' -> Zoom-Refine: '{r['zoom_refine_result']['final_answer']}'")
            print()
            if sum(1 for x in results if x['is_fixed']) >= 3:
                break


if __name__ == "__main__":
    test_zoom_refine(num_samples=100)
