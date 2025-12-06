#!/usr/bin/env python3
"""
Test script comparing no_crop vs single_crop vs smart_multi_crop.
Shows which approach works best for each sample.
"""

import os
import json
import torch
import gc
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor

from utils import get_confidence_from_logits, confidence_adaptive_crop, bbox_from_att_image_adaptive
from llava_methods import rel_attention_llava
from info import model_to_fullname

def test_all_modes(num_samples=10):
    """Compare no_crop, single_crop, and smart_multi_crop."""

    device = "cuda:2"
    print(f"Loading LLaVA model on {device}...")
    model_id = model_to_fullname["llava"]
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="eager"
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()

    with open("./data/textvqa/data.json") as f:
        data = json.load(f)[:num_samples]

    print(f"\n{'='*80}")
    print("Comparing: NO_CROP vs SINGLE_CROP vs SMART_MULTI_CROP")
    print(f"{'='*80}\n")

    results = []
    bbox_size = 336
    confidence_threshold = 0.7

    for i, item in enumerate(data):
        print(f"\n{'='*80}")
        print(f"Sample {i+1}/{num_samples}: {item['question'][:50]}...")
        print(f"Labels: {item['labels'][:3]}")
        print(f"{'='*80}")

        image_path = os.path.join("./data/textvqa/images", item['image_path'])
        image = Image.open(image_path).convert("RGB")
        question = item['question']

        prompt = f"<image>\nUSER: {question} Answer the question using a single word or phrase.\nASSISTANT:"
        general_prompt = f"<image>\nUSER: Write a general description of the image. Answer the question using a single word or phrase.\nASSISTANT:"

        # ============ NO CROP ============
        inputs = processor(images=image, text=prompt, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)

        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)
            logits = outputs.logits[:, -1, :]
            no_crop_conf = get_confidence_from_logits(logits)

        gen_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        no_crop_answer = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        if 'ASSISTANT: ' in no_crop_answer:
            no_crop_answer = no_crop_answer.split('ASSISTANT: ')[1]

        del inputs, outputs, gen_ids
        torch.cuda.empty_cache()

        # ============ GET ATTENTION MAP (for single & multi) ============
        short_prompt = prompt
        att_map = rel_attention_llava(image, short_prompt, general_prompt, model, processor)
        torch.cuda.empty_cache()
        gc.collect()

        # ============ SINGLE CROP ============
        single_bbox = bbox_from_att_image_adaptive(att_map, image.size, bbox_size)
        crop_img = image.crop(single_bbox)

        multi_prompt = "<image><image>\nUSER: " + question + " Answer the question using a single word or phrase.\nASSISTANT:"
        inputs = processor(images=[image, crop_img], text=multi_prompt, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)

        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)
            logits = outputs.logits[:, -1, :]
            single_crop_conf = get_confidence_from_logits(logits)

        gen_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        single_crop_answer = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        if 'ASSISTANT: ' in single_crop_answer:
            single_crop_answer = single_crop_answer.split('ASSISTANT: ')[1]

        del inputs, outputs, gen_ids
        torch.cuda.empty_cache()

        # ============ SMART MULTI CROP ============
        bboxes, analysis = confidence_adaptive_crop(
            att_map, image.size, bbox_size,
            original_confidence=no_crop_conf,
            confidence_threshold=confidence_threshold,
            max_crops=3
        )

        smart_num_crops = len(bboxes)

        if smart_num_crops == 0:
            # Model confident, use no-crop answer
            smart_answer = no_crop_answer
            smart_conf = no_crop_conf
        else:
            # Generate with crops
            crop_images = [image.crop(bbox) for bbox in bboxes]
            all_images = [image] + crop_images
            multi_prompt = "<image>" * len(all_images) + f"\nUSER: {question} Answer the question using a single word or phrase.\nASSISTANT:"

            inputs = processor(images=all_images, text=multi_prompt, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)

            with torch.no_grad():
                outputs = model(**inputs, return_dict=True)
                logits = outputs.logits[:, -1, :]
                crop_conf = get_confidence_from_logits(logits)

            gen_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
            crop_answer = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
            if 'ASSISTANT: ' in crop_answer:
                crop_answer = crop_answer.split('ASSISTANT: ')[1]

            # Trust the crop decision - we cropped because confidence was low
            smart_answer = crop_answer
            smart_conf = crop_conf

            del inputs, outputs, gen_ids
            torch.cuda.empty_cache()

        del att_map
        gc.collect()

        # Check correctness
        def is_correct(answer, labels):
            return any(l.lower() in answer.lower() or answer.lower() in l.lower() for l in labels)

        no_crop_correct = is_correct(no_crop_answer, item['labels'])
        single_crop_correct = is_correct(single_crop_answer, item['labels'])
        smart_correct = is_correct(smart_answer, item['labels'])

        # Display results
        print(f"\n  NO_CROP:     '{no_crop_answer}' (conf={no_crop_conf:.3f}) {'Y' if no_crop_correct else 'N'}")
        print(f"  SINGLE_CROP: '{single_crop_answer}' (conf={single_crop_conf:.3f}) {'Y' if single_crop_correct else 'N'}")
        print(f"  SMART({smart_num_crops}):    '{smart_answer}' (conf={smart_conf:.3f}) {'Y' if smart_correct else 'N'}")

        results.append({
            "question": item['question'][:40],
            "no_crop": {"answer": no_crop_answer, "conf": no_crop_conf, "correct": no_crop_correct},
            "single_crop": {"answer": single_crop_answer, "conf": single_crop_conf, "correct": single_crop_correct},
            "smart": {"answer": smart_answer, "conf": smart_conf, "correct": smart_correct, "num_crops": smart_num_crops}
        })

    # ============ SUMMARY ============
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    no_crop_acc = sum(1 for r in results if r['no_crop']['correct'])
    single_crop_acc = sum(1 for r in results if r['single_crop']['correct'])
    smart_acc = sum(1 for r in results if r['smart']['correct'])

    print(f"\nAccuracy:")
    print(f"  NO_CROP:     {no_crop_acc}/{len(results)} ({100*no_crop_acc/len(results):.1f}%)")
    print(f"  SINGLE_CROP: {single_crop_acc}/{len(results)} ({100*single_crop_acc/len(results):.1f}%)")
    print(f"  SMART:       {smart_acc}/{len(results)} ({100*smart_acc/len(results):.1f}%)")

    # Show smart crop distribution
    crop_dist = {}
    for r in results:
        n = r['smart']['num_crops']
        crop_dist[n] = crop_dist.get(n, 0) + 1

    print(f"\nSmart crop distribution:")
    for n in sorted(crop_dist.keys()):
        print(f"  {n} crops: {crop_dist[n]} samples")

    # Show where methods differ
    print(f"\n{'='*80}")
    print("DETAILED COMPARISON (where methods differ)")
    print(f"{'='*80}")

    for r in results:
        nc = 'Y' if r['no_crop']['correct'] else 'N'
        sc = 'Y' if r['single_crop']['correct'] else 'N'
        sm = 'Y' if r['smart']['correct'] else 'N'
        n_crops = r['smart']['num_crops']

        # Highlight differences
        if nc != sc or sc != sm or nc != sm:
            print(f"\n  Q: {r['question']}...")
            print(f"    NO_CROP={nc} | SINGLE={sc} | SMART({n_crops})={sm}")
            print(f"    Answers: '{r['no_crop']['answer']}' / '{r['single_crop']['answer']}' / '{r['smart']['answer']}'")

if __name__ == "__main__":
    test_all_modes(num_samples=10)
