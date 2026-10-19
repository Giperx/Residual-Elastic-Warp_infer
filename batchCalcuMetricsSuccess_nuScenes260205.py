import os
import sys
import torch
import numpy as np
import re
import math
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F_trans
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from tqdm import tqdm

# CUDA_VISIBLE_DEVICES=0 python batchCalcuMetricsSuccess_nuScenes260205.py
# ================= Configuration Area =================
# [Batch Size]
BATCH_SIZE = 64

# [Paths]
# Root directory containing stitched results (script searches recursively for scene folders)
STITCHED_RESULT_ROOT = "/home/test/LIVA/XZP/ImageStitch/Residual-Elastic-Warp/nuscenes_stitched_results_with_crop_stats0130"

# Original dataset directory (Used for GT and to determine TOTAL EXPECTED FRAMES)
DATASET_ROOT = "/home/test/LIVA/XZP/FeedForward/fine_tune3/AnySplat_1218infer/datasets/nuscenes/processed_10Hz/trainval2"

# Validation list file
VAL_LIST_PATH = "/home/test/LIVA/XZP/ImageStitch/Residual-Elastic-Warp/nuScenes_Val.txt"

# [Output File]
RESULT_TXT_PATH = os.path.join(STITCHED_RESULT_ROOT, "final_metrics_report_complete.txt")

# [Image Processing Params]
# 1. Crop Wide: Crop size from center of Wide image
CROP_W, CROP_H = 1600, 900 
# 2. Resize Target: Resize to this target size after crop (for metric calculation vs GT)
TARGET_W, TARGET_H = 896, 252

# [Quality Control Params]
WHITE_PIXEL_THRESHOLD = 0.9  # Pixel value > 0.9 considered white
WHITE_AREA_RATIO = 0.50       # Threshold for white area ratio in border regions

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ================= Utils =================

class MetricTracker:
    def __init__(self):
        self.reset()
    def reset(self):
        self.psnr = []
        self.ssim = []
        self.lpips = []
        self.count = 0
    def update(self, psnr, ssim, lpips):
        self.psnr.append(psnr)
        self.ssim.append(ssim)
        self.lpips.append(lpips)
        self.count += 1
    def get_avg(self):
        if self.count == 0:
            return {'psnr': 0.0, 'ssim': 0.0, 'lpips': 0.0, 'count': 0}
        return {
            'psnr': np.mean(self.psnr),
            'ssim': np.mean(self.ssim),
            'lpips': np.mean(self.lpips),
            'count': self.count
        }

def format_metrics(m):
    avg = m.get_avg()
    return f"N={avg['count']} | P={avg['psnr']:.2f} | S={avg['ssim']:.4f} | L={avg['lpips']:.4f}"

def load_scene_list(txt_path):
    if not os.path.exists(txt_path): return []
    with open(txt_path, 'r') as f:
        scenes = [line.strip() for line in f.readlines() if line.strip()]
    return scenes

def get_scene_frames(scene_path):
    img_dir = os.path.join(scene_path, "images")
    if not os.path.exists(img_dir): return []
    frame_ids = []
    pattern = re.compile(r"^(.*)_0\.(jpg|png|jpeg)$", re.IGNORECASE)
    for f in sorted(os.listdir(img_dir)):
        match = pattern.match(f)
        if match: frame_ids.append(match.group(1))
    return sorted(frame_ids)

def check_stitching_quality(img_tensor):
    """
    Check if stitching failed (White Border Detection).
    img_tensor: [C, H, W] in [0, 1]
    """
    C, H, W = img_tensor.shape
    quarter_w = W // 4
    if quarter_w == 0: return True 
    
    # Check left and right 1/4 regions
    left_region = img_tensor[:, :, :quarter_w]
    right_region = img_tensor[:, :, -quarter_w:]
    
    # Calculate white pixel ratio (RGB mean > Threshold)
    left_white_mask = (left_region.mean(dim=0) > WHITE_PIXEL_THRESHOLD).float()
    right_white_mask = (right_region.mean(dim=0) > WHITE_PIXEL_THRESHOLD).float()
    
    left_ratio = left_white_mask.mean().item()
    right_ratio = right_white_mask.mean().item()
    
    if left_ratio > WHITE_AREA_RATIO or right_ratio > WHITE_AREA_RATIO:
        return False # Quality Check Failed
    return True # Quality Check Passed

def process_wide_image(img_path):
    """
    Process Stitched Result Image:
    1. Load
    2. Check Quality (Fail -> Return None)
    3. Center Crop (1600x900)
    4. Resize (896x252)
    """
    # Note: Existence check is done before calling this function in the main loop logic now
    if not os.path.exists(img_path): return None, "File Not Found"
    
    try:
        img = Image.open(img_path).convert('RGB')
        tensor = transforms.ToTensor()(img) # [C, H, W] in [0, 1]
        
        # 1. Quality Check
        if not check_stitching_quality(tensor):
            return None, "Quality Check Failed (White borders)"
            
        # 2. Center Crop 1600x900
        _, h, w = tensor.shape
        if h < CROP_H or w < CROP_W:
            center_y, center_x = h // 2, w // 2
            start_y = max(0, center_y - CROP_H // 2)
            start_x = max(0, center_x - CROP_W // 2)
            tensor_crop = tensor[:, start_y : start_y+CROP_H, start_x : start_x+CROP_W]
        else:
            center_y, center_x = h // 2, w // 2
            start_y = center_y - CROP_H // 2
            start_x = center_x - CROP_W // 2
            tensor_crop = tensor[:, start_y : start_y+CROP_H, start_x : start_x+CROP_W]
            
        # 3. Resize to Target
        tensor_crop = tensor_crop.unsqueeze(0) 
        tensor_final = torch.nn.functional.interpolate(
            tensor_crop, size=(TARGET_H, TARGET_W), mode='bilinear', align_corners=True
        ).squeeze(0)
        
        return tensor_final, "Success"
        
    except Exception as e:
        return None, str(e)

def process_gt_image(img_path):
    """
    Process GT Image (Cam0 or Cam5):
    Resize directly to Target Size
    """
    if not os.path.exists(img_path): return None
    img = Image.open(img_path).convert('RGB')
    tensor = transforms.ToTensor()(img)
    tensor = tensor.unsqueeze(0)
    tensor_final = torch.nn.functional.interpolate(
        tensor, size=(TARGET_H, TARGET_W), mode='bilinear', align_corners=True
    ).squeeze(0)
    return tensor_final

# ================= Main Process =================

def main():
    print(f"Metrics Calculation Started on {DEVICE}")
    print(f"Config: Batch={BATCH_SIZE} | Crop={CROP_W}x{CROP_H} | Resize={TARGET_W}x{TARGET_H}")
    
    # Init LPIPS
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type='alex', reduction='none').to(DEVICE)
    lpips_fn.eval()
    
    scenes = load_scene_list(VAL_LIST_PATH)
    
    # Global Statistics
    global_stats = {
        'total_expected': 0,    # Total frames in GT
        'missing_files': 0,     # Not found in result folder
        'quality_failed': 0,    # Found but white borders
        'valid_success': 0,     # Found and Good
        'metrics': MetricTracker()
    }

    # Per Scene Statistics
    scene_stats = {} 
    
    all_tasks = []
    
    # 1. Scanning Tasks based on GT
    print("Scanning GT dataset for expected frames...")
    
    for scene_id in tqdm(scenes, desc="Scanning Scenes"):
        # Paths
        gt_scene_dir = os.path.join(DATASET_ROOT, scene_id)
        result_scene_dir = os.path.join(STITCHED_RESULT_ROOT, scene_id)
        
        # Even if result dir doesn't exist, we must count these as missing!
        # if not os.path.exists(result_scene_dir): ... 
        
        frames = get_scene_frames(gt_scene_dir)
        
        # Init Scene Stats
        scene_stats[scene_id] = {
            'total_expected': 0,
            'missing_files': 0,
            'quality_failed': 0,
            'valid_success': 0,
            'tracker': MetricTracker()
        }
        
        for frame_id in frames:
            # We expect two outputs per timestamp: Front and Back
            for view_type, cam_idx in [('front', '0'), ('back', '5')]:
                
                # --- Update Counts ---
                global_stats['total_expected'] += 1
                scene_stats[scene_id]['total_expected'] += 1
                
                # Define Expected Paths
                wide_name = f"{frame_id}_{view_type}_wide.jpg"
                wide_path = os.path.join(result_scene_dir, wide_name)
                gt_path = os.path.join(gt_scene_dir, "images", f"{frame_id}_{cam_idx}.jpg")
                
                # Check Existence immediately
                if not os.path.exists(wide_path):
                    global_stats['missing_files'] += 1
                    scene_stats[scene_id]['missing_files'] += 1
                    continue # Skip adding to processing queue
                
                # If exists, add to processing queue for Quality Check & Metrics
                all_tasks.append({
                    'scene_id': scene_id,
                    'wide_path': wide_path,
                    'gt_path': gt_path,
                    'view_type': view_type
                })

    # 2. Process Existing Files (Batch)
    total_existing = len(all_tasks)
    num_batches = math.ceil(total_existing / BATCH_SIZE)
    print(f"Total Expected Frames: {global_stats['total_expected']}")
    print(f"Missing Files: {global_stats['missing_files']}")
    print(f"Files to Process (Found): {total_existing}")
    
    for i in tqdm(range(num_batches), desc="Evaluating Batches"):
        batch_meta = all_tasks[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
        
        wide_list = []
        gt_list = []
        valid_indices = [] # Indices in current batch that passed IO and Quality Check
        
        # A. Load & Quality Check
        for idx, task in enumerate(batch_meta):
            scene_id = task['scene_id']
            
            # Process Wide (Quality Check happens here)
            wide_tensor, msg = process_wide_image(task['wide_path'])
            
            if wide_tensor is None:
                # File existed (passed loop 1) but Quality Failed (or read error)
                if "Quality" in msg:
                    global_stats['quality_failed'] += 1
                    scene_stats[scene_id]['quality_failed'] += 1
                else:
                    # Rare read error, treat as missing or failed
                    global_stats['missing_files'] += 1 
                    scene_stats[scene_id]['missing_files'] += 1
                continue
            
            # Process GT
            gt_tensor = process_gt_image(task['gt_path'])
            if gt_tensor is None: continue # Should not happen if GT exists
            
            # If we reach here, it is a VALID SUCCESS
            global_stats['valid_success'] += 1
            scene_stats[scene_id]['valid_success'] += 1
            
            wide_list.append(wide_tensor)
            gt_list.append(gt_tensor)
            valid_indices.append(idx)

        if not wide_list: continue
        
        # B. Metrics Calculation (GPU)
        wide_batch = torch.stack(wide_list).to(DEVICE)
        gt_batch = torch.stack(gt_list).to(DEVICE)
        
        current_bs = wide_batch.shape[0]
        
        with torch.no_grad():
            lpips_vals = lpips_fn(wide_batch, gt_batch).view(-1)
            
            for b in range(current_bs):
                p = peak_signal_noise_ratio(wide_batch[b:b+1], gt_batch[b:b+1], data_range=1.0).item()
                s = structural_similarity_index_measure(wide_batch[b:b+1], gt_batch[b:b+1], data_range=1.0).item()
                l = lpips_vals[b].item()
                
                # Update Metrics
                global_stats['metrics'].update(p, s, l)
                
                task_idx = valid_indices[b]
                s_id = batch_meta[task_idx]['scene_id']
                scene_stats[s_id]['tracker'].update(p, s, l)

    # 3. Generate Report
    print("Generating Report...")
    
    # Overall Rates
    total = global_stats['total_expected']
    valid = global_stats['valid_success']
    missing = global_stats['missing_files']
    quality_bad = global_stats['quality_failed']
    
    # "Total Success Rate" = Valid / Total Expected
    final_success_rate = (valid / total * 100) if total > 0 else 0
    # "Generation Rate" = (Total - Missing) / Total
    generation_rate = ((total - missing) / total * 100) if total > 0 else 0
    
    with open(RESULT_TXT_PATH, 'w') as f:
        f.write("Evaluation Report: Complete Stitching Pipeline\n")
        f.write("==================================================\n")
        f.write(f"Source Dir: {STITCHED_RESULT_ROOT}\n")
        f.write(f"Ref Crop: {CROP_W}x{CROP_H} -> Resize: {TARGET_W}x{TARGET_H}\n\n")
        
        f.write(">>> Global Summary <<<\n")
        f.write(f"Total Expected Frames: {total}\n")
        f.write(f"  - Missing Output:    {missing} ({missing/total*100:.2f}%)\n")
        f.write(f"  - Quality Failed:    {quality_bad} ({quality_bad/total*100:.2f}%) [White Borders]\n")
        f.write(f"  - Valid Success:     {valid} ({valid/total*100:.2f}%)\n\n")
        
        f.write(f"Pipeline Success Rate (Valid/Total): {final_success_rate:.2f}%\n")
        f.write(f"Metrics (Computed on Valid Images Only):\n")
        f.write(f"  {format_metrics(global_stats['metrics'])}\n\n")
        
        f.write("==================================================\n")
        f.write(">>> Per Scene Breakdown <<<\n")
        f.write(f"{'Scene ID':<20} | {'Valid/Total':<12} | {'Rate %':<7} | {'PSNR':<6} | {'SSIM':<6} | {'LPIPS':<6} | {'Miss':<5} | {'Fail':<5}\n")
        f.write("-" * 90 + "\n")
        
        for scene_id in sorted(scene_stats.keys()):
            s = scene_stats[scene_id]
            rate = (s['valid_success'] / s['total_expected'] * 100) if s['total_expected'] > 0 else 0
            avg = s['tracker'].get_avg()
            
            f.write(f"{scene_id:<20} | {s['valid_success']}/{s['total_expected']:<4} | {rate:6.2f}% | "
                    f"{avg['psnr']:6.2f} | {avg['ssim']:6.4f} | {avg['lpips']:6.4f} | "
                    f"{s['missing_files']:<5} | {s['quality_failed']:<5}\n")

    print(f"Evaluation Finished. Report saved to: {RESULT_TXT_PATH}")

if __name__ == "__main__":
    main()