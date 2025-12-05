import os
from PIL import Image
import torch
import numpy as np
from transformers import AutoProcessor, LlavaForConditionalGeneration, InstructBlipProcessor, InstructBlipForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
import argparse
from tqdm import tqdm
import json
from datasets import load_dataset

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
        method_name: String indicating which attention method to use
        image_path: Path to the input image file
        question: The full question to ask about the image
        model: The loaded model instance
        processor: The processor for the corresponding model
        short_question: A shortened version of the question for attention computation
        crop_mode: "no_crop", "single_crop", or "smart_multi_crop"
        
    Returns:
        dict with keys: original_answer, crop_answer, bbox (or bboxes for multi), analysis (for smart mode)
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

        # Get original answer
        inputs = processor(prompt, image, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
        ori_generate_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        ori_generation = [i.split('ASSISTANT: ')[1] for i in processor.batch_decode(ori_generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)][0]

        del inputs
        torch.cuda.empty_cache()

        if crop_mode == "no_crop":
            return {"original_answer": ori_generation, "crop_answer": ori_generation, "bbox": None}

        # Get attention map
        att_map = get_attention_map(model_name, method_name, image, short_prompt, general_prompt, model, processor)
        
        if crop_mode == "smart_multi_crop":
            bboxes, analysis = smart_multi_crop(att_map, image.size, bbox_size)
            if not bboxes:
                return {"original_answer": ori_generation, "crop_answer": ori_generation, "bboxes": [], "analysis": analysis}
            
            crop_images = [image.crop(bbox) for bbox in bboxes]
            all_images = [image] + crop_images
            multi_prompt = "<image>" * len(all_images) + f"\nUSER: {question} Answer the question using a single word or phrase.\nASSISTANT:"
            multi_inputs = processor(multi_prompt, all_images, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
            multi_generate_ids = model.generate(**multi_inputs, max_new_tokens=20, do_sample=False)
            multi_generation = [i.split('ASSISTANT: ')[1] for i in processor.batch_decode(multi_generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)][0]
            return {"original_answer": ori_generation, "crop_answer": multi_generation, "bboxes": bboxes, "analysis": analysis}
        
        else:  # single_crop
            bbox = bbox_from_att_image_adaptive(att_map, image.size, bbox_size)
            crop_image = image.crop(bbox)
            multi_prompt = f"<image><image>\nUSER: {question} Answer the question using a single word or phrase.\nASSISTANT:"
            multi_inputs = processor(multi_prompt, [image, crop_image], return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
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
        
        if crop_mode == "smart_multi_crop":
            bboxes, analysis = smart_multi_crop(att_map, image.size, bbox_size)
            if not bboxes:
                return {"original_answer": ori_generation, "crop_answer": ori_generation, "bboxes": [], "analysis": analysis}
            
            crop_images = [image.crop(bbox) for bbox in bboxes]
            all_images = [image] + crop_images
            multi_inputs = processor(images=all_images, text=prompt, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
            multi_generate_ids = model.generate(**multi_inputs, max_new_tokens=20, do_sample=False)
            multi_generation = processor.batch_decode(multi_generate_ids, skip_special_tokens=True)[0]
            return {"original_answer": ori_generation, "crop_answer": multi_generation, "bboxes": bboxes, "analysis": analysis}
        
        else:  # single_crop
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
        
        # Get original answer
        ori_messages = [{"role": "user", "content": [{"type": "image", "image": f'data:image;base64,{image_str}'}, {"type": "text", "text": prompt}]}]
        ori_inputs = prepare_qwen2_5_input(ori_messages, processor).to(model.device, torch.bfloat16)
        ori_generate_ids = model.generate(**ori_inputs, max_new_tokens=20, do_sample=False)
        ori_generate_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(ori_inputs.input_ids, ori_generate_ids)]
        ori_generation = processor.batch_decode(ori_generate_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        del ori_inputs
        torch.cuda.empty_cache()

        if crop_mode == "no_crop":
            return {"original_answer": ori_generation, "crop_answer": ori_generation, "bbox": None, "num_img_tokens": 0}

        att_map = get_attention_map(model_name, method_name, image, prompt, general_prompt, model, processor)
        
        if crop_mode == "smart_multi_crop":
            bboxes, analysis = smart_multi_crop(att_map, image.size, bbox_size)
            if not bboxes:
                return {"original_answer": ori_generation, "crop_answer": ori_generation, "bboxes": [], "analysis": analysis, "num_img_tokens": 0}
            
            content = [{"type": "image", "image": f'data:image;base64,{image_str}'}]
            for bbox in bboxes:
                crop_str = encode_base64(image.crop(bbox))
                content.append({"type": "image", "image": f'data:image;base64,{crop_str}'})
            content.append({"type": "text", "text": prompt})
            
            multi_messages = [{"role": "user", "content": content}]
            multi_inputs = prepare_qwen2_5_input(multi_messages, processor).to(model.device, torch.bfloat16)
            multi_generate_ids = model.generate(**multi_inputs, max_new_tokens=20, do_sample=False)
            num_img_tokens = sum(multi_inputs.input_ids[0] == 151655)
            multi_generate_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(multi_inputs.input_ids, multi_generate_ids)]
            multi_generation = processor.batch_decode(multi_generate_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return {"original_answer": ori_generation, "crop_answer": multi_generation, "bboxes": bboxes, "analysis": analysis, "num_img_tokens": int(num_img_tokens)}
        
        else:  # single_crop
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
            - crop_mode: "no_crop", "single_crop", or "smart_multi_crop"
    """

    if args.model == 'llava':
        model = LlavaForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager").to(args.device)
        processor = AutoProcessor.from_pretrained(args.model_id)
    elif args.model == 'blip':
        model = InstructBlipForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(args.device)
        processor = InstructBlipProcessor.from_pretrained(args.model_id)
    elif args.model == 'qwen2_5':
        max_pixels = 256 * 28 * 28
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
        
        if args.crop_mode == "smart_multi_crop":
            d["bboxes"] = result.get("bboxes", [])
            d["analysis"] = result.get("analysis", {})
        else:
            d["bbox"] = result.get("bbox")
        
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
    parser.add_argument("--method", type=str, default="grad_att", choices=["rel_att", "pure_grad", "grad_att", "rel_att_high", "pure_grad_high", "grad_att_high"])
    parser.add_argument("--crop_mode", type=str, default="single_crop", choices=["no_crop", "single_crop", "smart_multi_crop"])
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