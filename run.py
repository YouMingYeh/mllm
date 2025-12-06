"""
Main entry point for running visual cropping experiments.

This module implements the ViCrop pipeline with multiple crop modes:

Crop Modes (Effective - Keep):
------------------------------
- no_crop: Baseline - use original image only
- single_crop: Original paper method - attention-based single crop + original
- masked / masked_black: Mask out low-attention regions (black) - reduces noise
- masked_blur: Mask out low-attention regions (blur effect)

Attention Methods:
------------------
- rel_att: Relative attention (question vs general description) - most effective
- grad_att: Gradient-weighted attention
- pure_grad: Pure gradient on input pixels
- cls_att: CLS token attention from ViT (FasterVLM-inspired) - single forward pass

Key Insight: "MLLMs are noise-sensitive, not information-starved"
- Adding more images (multi-crop) HURTS - adds noise/confusion
- Masking irrelevant regions HELPS - reduces noise
- Single focused crop WORKS - provides relevant detail

References:
- "MLLMs Know Where to Look" (ICLR 2025) - ViCrop paper
- "FasterVLM" - CLS attention for token importance
"""

import os
import json
import argparse

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration
)

from llava_methods import *
from blip_methods import *
from qwen2_5_methods import *
from utils import *
from info import *


def get_attention_map(model_name, method_name, image, prompt, general_prompt, model, processor):
    """Get attention map based on model and method."""
    if model_name == "llava":
        if method_name == 'grad_att':
            return gradient_attention_llava(image, prompt, general_prompt, model, processor)
        elif method_name == 'grad_att_high':
            return high_res(gradient_attention_llava, image, prompt, general_prompt, model, processor)
        elif method_name == 'rel_att':
            return rel_attention_llava(image, prompt, general_prompt, model, processor)
        elif method_name == 'rel_att_high':
            return high_res(rel_attention_llava, image, prompt, general_prompt, model, processor)
        elif method_name == 'pure_grad':
            return pure_gradient_llava(image, prompt, general_prompt, model, processor)
        elif method_name == 'pure_grad_high':
            return high_res(pure_gradient_llava, image, prompt, general_prompt, model, processor)
        elif method_name == 'cls_att':
            return cls_attention_llava(image, prompt, general_prompt, model, processor)
    elif model_name == "blip":
        if method_name == 'grad_att':
            return gradient_attention_blip(image, prompt, general_prompt, model, processor)
        elif method_name == 'grad_att_high':
            return high_res(gradient_attention_blip, image, prompt, general_prompt, model, processor)
        elif method_name == 'rel_att':
            return rel_attention_blip(image, prompt, general_prompt, model, processor)
        elif method_name == 'rel_att_high':
            return high_res(rel_attention_blip, image, prompt, general_prompt, model, processor)
        elif method_name == 'pure_grad':
            return pure_gradient_blip(image, prompt, general_prompt, model, processor)
        elif method_name == 'pure_grad_high':
            return high_res(pure_gradient_blip, image, prompt, general_prompt, model, processor)
    elif model_name == "qwen2_5":
        if method_name == 'rel_att':
            return rel_attention_qwen2_5(image, prompt, general_prompt, model, processor)
        elif method_name == 'grad_att':
            return gradient_attention_qwen2_5(image, prompt, general_prompt, model, processor)
    return None

def vicrop_qa(model_name, method_name, image_path, question, model, processor, short_question, crop_mode="single_crop"):
    """
    Performs visual cropping and question answering using different attention methods.

    Args:
        model_name: String indicating which model to use ("llava", "blip", or "qwen2_5")
        method_name: String indicating which attention method to use (rel_att, grad_att, pure_grad, etc.)
        image_path: Path to the input image file
        question: The full question to ask about the image
        model: The loaded model instance
        processor: The processor for the corresponding model
        short_question: A shortened version of the question for attention computation
        crop_mode: Cropping strategy to use:
            - "no_crop": Baseline, no cropping
            - "single_crop": Original paper method, attention-based single crop
            - "masked" / "masked_black": Mask out low-attention regions (black)
            - "masked_blur": Mask out low-attention regions (blur effect)
            - "masked_dim": Mask out low-attention regions (dimmed)

    Returns:
        dict with keys:
            - original_answer: Answer from original image only
            - crop_answer: Answer using crop/mask enhancement
            - bbox/bboxes: Bounding box(es) used for cropping
            - analysis: Decision analysis (for smart/masked modes)
    """

    if model_name == "llava":
        bbox_size = 336
    elif model_name == "blip":
        bbox_size = 224
    elif model_name == "qwen2_5":
        bbox_size = 224

    image = Image.open(image_path).convert("RGB")
    model.eval()

    general_question = 'Write a general description of the image.'

    if model_name == "llava":

        short_prompt = f"<image>\nUSER: {short_question} Answer the question using a single word or phrase.\nASSISTANT:"
        prompt = f"<image>\nUSER: {question} Answer the question using a single word or phrase.\nASSISTANT:"
        general_prompt = f"<image>\nUSER: {general_question} Answer the question using a single word or phrase.\nASSISTANT:"

        # Get original answer with confidence
        inputs = processor(images=image, text=prompt, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)

        with torch.no_grad():
            ori_outputs = model(**inputs, return_dict=True)
            ori_logits = ori_outputs.logits[:, -1, :]
            ori_confidence = get_confidence_from_logits(ori_logits)

        ori_generate_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        ori_generation = [i.split('ASSISTANT: ')[1] for i in processor.batch_decode(ori_generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)][0]

        del inputs, ori_outputs
        torch.cuda.empty_cache()

        if crop_mode == "no_crop":
            return {"original_answer": ori_generation, "crop_answer": ori_generation, "bbox": None, "confidence": ori_confidence}

        # Get attention map for single_crop and masked modes
        att_map = get_attention_map(model_name, method_name, image, short_prompt, general_prompt, model, processor)

        # MASKED MODE: Use original + masked image (instead of multi-crop)
        if crop_mode in ["masked", "masked_black", "masked_blur", "masked_dim"]:
            # Determine mask style
            if crop_mode == "masked_blur":
                mask_style = "blur"
            elif crop_mode == "masked_dim":
                mask_style = "dim"
            else:  # "masked" or "masked_black"
                mask_style = "black"

            # Check if masking would help
            should_mask, mask_analysis = should_use_masking(att_map)

            if not should_mask:
                # Attention too diffuse, masking won't help
                mask_analysis["final_decision"] = "skip_mask_diffuse_attention"
                return {
                    "original_answer": ori_generation,
                    "crop_answer": ori_generation,
                    "masked": False,
                    "analysis": mask_analysis,
                    "ori_confidence": ori_confidence
                }

            # Create masked image
            masked_image = create_masked_image(image, att_map, threshold_percentile=40,
                                               soft_mask=True, mask_style=mask_style)

            # Prompt with original + masked (like original + crop in the paper)
            masked_prompt = f"<image><image>\nUSER: The second image highlights the relevant region. {question} Answer the question using a single word or phrase.\nASSISTANT:"
            masked_inputs = processor(images=[image, masked_image], text=masked_prompt,
                                      return_tensors="pt", padding=True).to(model.device, torch.bfloat16)

            with torch.no_grad():
                masked_outputs = model(**masked_inputs, return_dict=True)
                masked_logits = masked_outputs.logits[:, -1, :]
                masked_confidence = get_confidence_from_logits(masked_logits)

            masked_generate_ids = model.generate(**masked_inputs, max_new_tokens=20, do_sample=False)
            masked_generation = [i.split('ASSISTANT: ')[1] for i in processor.batch_decode(masked_generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)][0]

            del masked_inputs, masked_outputs
            torch.cuda.empty_cache()

            mask_analysis["final_decision"] = "using_masked_image"
            mask_analysis["ori_confidence"] = float(ori_confidence)
            mask_analysis["masked_confidence"] = float(masked_confidence)
            mask_analysis["mask_style"] = mask_style

            return {
                "original_answer": ori_generation,
                "crop_answer": masked_generation,
                "masked": True,
                "analysis": mask_analysis
            }

        # single_crop: Original paper method
        bbox = bbox_from_att_image_adaptive(att_map, image.size, bbox_size)
        crop_image = image.crop(bbox)
        multi_prompt = f"<image><image>\nUSER: {question} Answer the question using a single word or phrase.\nASSISTANT:"
        multi_inputs = processor(images=[image, crop_image], text=multi_prompt, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
        multi_generate_ids = model.generate(**multi_inputs, max_new_tokens=20, do_sample=False)
        multi_generation = [i.split('ASSISTANT: ')[1] for i in processor.batch_decode(multi_generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)][0]
        return {"original_answer": ori_generation, "crop_answer": multi_generation, "bbox": bbox}
    
    elif model_name == "blip":

        short_prompt = f"Question: {short_question} Short answer:"
        prompt = f"Question: {question} Short answer:"
        general_prompt = f"Question: {general_question} Short answer:"

        inputs = processor(images=image, text=prompt, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
        ori_generate_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        ori_generation = processor.batch_decode(ori_generate_ids, skip_special_tokens=True)[0]

        del inputs
        torch.cuda.empty_cache()

        if crop_mode == "no_crop":
            return {"original_answer": ori_generation, "crop_answer": ori_generation, "bbox": None}

        att_map = get_attention_map(model_name, method_name, image, short_prompt, general_prompt, model, processor)

        # single_crop: Original paper method
        bbox = bbox_from_att_image_adaptive(att_map, image.size, bbox_size)
        crop_image = image.crop(bbox)
        multi_inputs = processor(images=[image, crop_image], text=prompt, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
        multi_generate_ids = model.generate(**multi_inputs, max_new_tokens=20, do_sample=False)
        multi_generation = processor.batch_decode(multi_generate_ids, skip_special_tokens=True)[0]
        return {"original_answer": ori_generation, "crop_answer": multi_generation, "bbox": bbox}

    elif model_name == "qwen2_5":

        prompt = f'{question} Answer the question using a single word or phrase.'
        general_prompt = f'{general_question} Answer the question using a single word or phrase.'

        image_str = encode_base64(image)

        # Get original answer with confidence
        ori_messages = [{"role": "user", "content": [{"type": "image", "image": f'data:image;base64,{image_str}'}, {"type": "text", "text": prompt}]}]
        ori_inputs = prepare_qwen2_5_input(ori_messages, processor).to(model.device, torch.bfloat16)

        # Get logits for confidence calculation
        with torch.no_grad():
            ori_outputs = model(**ori_inputs, return_dict=True)
            ori_logits = ori_outputs.logits[:, -1, :]  # Last token logits
            ori_confidence = get_confidence_from_logits(ori_logits)

        ori_generate_ids = model.generate(**ori_inputs, max_new_tokens=20, do_sample=False)
        ori_generate_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(ori_inputs.input_ids, ori_generate_ids)]
        ori_generation = processor.batch_decode(ori_generate_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        del ori_inputs, ori_outputs
        torch.cuda.empty_cache()

        if crop_mode == "no_crop":
            return {"original_answer": ori_generation, "crop_answer": ori_generation, "bbox": None, "num_img_tokens": 0, "confidence": ori_confidence}

        att_map = get_attention_map(model_name, method_name, image, prompt, general_prompt, model, processor)

        # single_crop: Original paper method
        bbox = bbox_from_att_image_adaptive(att_map, image.size, bbox_size)
        crop_image = image.crop(bbox)
        crop_image_str = encode_base64(crop_image)

        multi_messages = [{"role": "user", "content": [{"type": "image", "image": f'data:image;base64,{image_str}'}, {"type": "image", "image": f'data:image;base64,{crop_image_str}'}, {"type": "text", "text": prompt}]}]
        multi_inputs = prepare_qwen2_5_input(multi_messages, processor).to(model.device, torch.bfloat16)
        multi_generate_ids = model.generate(**multi_inputs, max_new_tokens=20, do_sample=False)
        num_img_tokens = sum(multi_inputs.input_ids[0] == 151655)
        multi_generate_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(multi_inputs.input_ids, multi_generate_ids)]
        multi_generation = processor.batch_decode(multi_generate_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return {"original_answer": ori_generation, "crop_answer": multi_generation, "bbox": bbox, "num_img_tokens": int(num_img_tokens)}
        

def main(args):
    """
    Main function to run the visual cropping and question answering pipeline.

    Args:
        args: An argparse.Namespace object containing:
            - model, model_id, device, question_path, image_path
            - task, method, output_path, total_chunks, chunk_id
            - crop_mode: "no_crop", "single_crop", or "masked_*" variants
    """

    if args.model == 'llava':
        model = LlavaForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager").to(args.device)
        processor = AutoProcessor.from_pretrained(args.model_id)
    elif args.model == 'blip':
        model = InstructBlipForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(args.device)
        processor = InstructBlipProcessor.from_pretrained(args.model_id)
    elif args.model == 'qwen2_5':
        max_pixels = 64 * 28 * 28  # Reduced from 256 to 64 for memory with attention extraction
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager").to(args.device)
        processor = AutoProcessor.from_pretrained(args.model_id, max_pixels=max_pixels)
        processor.image_processor.size["longest_edge"] = max_pixels
    
    if os.path.exists(args.question_path):
        with open(args.question_path, "r") as f:
            whole_data = json.load(f)
    else:
        whole_data = list(load_dataset(args.question_path)['test'])
    
    # Limit samples if specified
    if args.num_samples is not None:
        whole_data = whole_data[:args.num_samples]
    
    for data in whole_data:
        data["image_path"] = os.path.join(args.image_path, data["image_path"]) if "image_path" in data else os.path.join(args.image_path, f"{data['image_id']}.jpg")

    splited_data = np.array_split(whole_data, args.total_chunks)
    data = splited_data[args.chunk_id]

    new_datas = []

    for d in tqdm(data, desc=f"Processing ({args.crop_mode})", ncols=100):
        question = d["question"]
        image_path = d["image_path"]
        short_question = d.get("short_question", d["question"])

        result = vicrop_qa(args.model, args.method, image_path, question, model, processor, short_question, args.crop_mode)
        
        d["original_answer"] = result["original_answer"]
        d["crop_answer"] = result["crop_answer"]
        d["crop_mode"] = args.crop_mode
        d["bbox"] = result.get("bbox")

        # Include analysis for masked modes
        if "analysis" in result:
            d["analysis"] = result["analysis"]
        
        if args.model == "qwen2_5":
            d["num_img_tokens"] = result.get("num_img_tokens", 0)

        new_datas.append(d)

    out_put_dir = os.path.dirname(args.output_path)
    if not os.path.exists(out_put_dir):
        os.makedirs(out_put_dir)

    if os.path.exists(args.output_path):
        with open(args.output_path, "r") as f:
            old_datas = json.load(f)
        new_datas = old_datas + new_datas
    
    with open(args.output_path, "w") as f:
        json.dump(new_datas, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava", choices=model_to_fullname.keys())
    parser.add_argument("--task", type=str, default="textvqa", choices=task_to_question_path.keys())
    parser.add_argument("--method", type=str, default="rel_att", choices=["rel_att", "pure_grad", "grad_att", "rel_att_high", "pure_grad_high", "grad_att_high", "cls_att"])
    parser.add_argument("--crop_mode", type=str, default="single_crop", choices=["no_crop", "single_crop", "masked", "masked_black", "masked_blur", "masked_dim"])
    parser.add_argument("--save_path", type=str, default="./playground/data/results")
    parser.add_argument("--total_chunks", type=int, default=1)
    parser.add_argument("--chunk_id", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of samples to process (for testing)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Include crop_mode in output name for easy comparison
    output_name = f'{args.model}-{args.task}-{args.method}-{args.crop_mode}.json'

    args.model_id = model_to_fullname[args.model]
    args.output_path = os.path.join(args.save_path, output_name)
    args.image_path = task_to_image_path[args.task]
    args.question_path = task_to_question_path[args.task]

    main(args)