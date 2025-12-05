import numpy as np
from utils import bbox_from_att_image_adaptive

def smart_multi_crop(att_map, image_size, bbox_size=224, max_crops=4, min_rel_threshold=0.4):
    """
    Iteratively generates crops using the original ViCrop method (bbox_from_att_image_adaptive).
    
    This is a strict extension of the original ViCrop method.
    1. It calculates the first crop exactly as ViCrop does.
    2. It then masks that region and checks if there is another significant region.
    3. It repeats this until the next region's attention score is too low relative to the first one.
    
    Args:
        att_map: 2D attention map
        image_size: (width, height) of original image
        bbox_size: Base size for crops
        max_crops: Maximum number of crops to generate
        min_rel_threshold: Minimum score ratio relative to the first crop to keep a subsequent crop.
        
    Returns:
        tuple: (list of bboxes, analysis dict)
    """
    bboxes = []
    scores = []
    
    # Working copy of attention map
    att_remaining = att_map.copy()
    
    # Helper to calculate score (mean attention in bbox)
    def get_bbox_score(bbox, att_m):
        # Convert bbox to attention map coordinates
        scale_x = att_m.shape[1] / image_size[0]
        scale_y = att_m.shape[0] / image_size[1]
        
        x1, y1, x2, y2 = bbox
        x1_idx = int(x1 * scale_x)
        y1_idx = int(y1 * scale_y)
        x2_idx = min(int(x2 * scale_x), att_m.shape[1])
        y2_idx = min(int(y2 * scale_y), att_m.shape[0])
        
        if x2_idx <= x1_idx or y2_idx <= y1_idx:
            return 0.0
            
        region = att_m[y1_idx:y2_idx, x1_idx:x2_idx]
        return float(region.mean())

    # Helper to mask out a bbox
    def mask_bbox(bbox, att_m):
        scale_x = att_m.shape[1] / image_size[0]
        scale_y = att_m.shape[0] / image_size[1]
        
        x1, y1, x2, y2 = bbox
        x1_idx = int(x1 * scale_x)
        y1_idx = int(y1 * scale_y)
        x2_idx = min(int(x2 * scale_x), att_m.shape[1])
        y2_idx = min(int(y2 * scale_y), att_m.shape[0])
        
        att_m[y1_idx:y2_idx, x1_idx:x2_idx] = 0
        return att_m

    # 1. First Crop (The Original ViCrop)
    # This ensures that if we stop here, we are identical to the original method.
    first_bbox = bbox_from_att_image_adaptive(att_remaining, image_size, bbox_size)
    first_score = get_bbox_score(first_bbox, att_remaining)
    
    # If the first crop has effectively zero attention, we return nothing (no crop needed/possible)
    if first_score <= 1e-6:
        return [], {"decision_reason": "zero_attention", "crop_scores": []}

    bboxes.append(first_bbox)
    scores.append(first_score)
    att_remaining = mask_bbox(first_bbox, att_remaining)

    # 2. Subsequent Crops
    for _ in range(max_crops - 1):
        if att_remaining.max() <= 1e-6:
            break
            
        next_bbox = bbox_from_att_image_adaptive(att_remaining, image_size, bbox_size)
        next_score = get_bbox_score(next_bbox, att_remaining)
        
        # Stopping condition: Is this crop significant compared to the first one?
        if next_score < first_score * min_rel_threshold:
            break
            
        bboxes.append(next_bbox)
        scores.append(next_score)
        att_remaining = mask_bbox(next_bbox, att_remaining)

    analysis = {
        "crop_scores": scores,
        "decision_reason": "iterative_threshold"
    }
    
    return bboxes, analysis
