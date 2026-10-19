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
# CUDA_VISIBLE_DEVICES=0 python batchCalcuMetrics_nuScenes.py
# ================= 配置区域 =================
# [Batch Size]
BATCH_SIZE = 64

# [路径配置]
# 拼接结果所在的根目录 (脚本会递归搜索该目录下的 scene 文件夹)
STITCHED_RESULT_ROOT = "/home/test/LIVA/XZP/ImageStitch/Residual-Elastic-Warp/nuscenes_stitched_results_with_crop_stats0130"

# 原始数据集目录 (用于读取 GT)
DATASET_ROOT = "/home/test/LIVA/XZP/FeedForward/fine_tune3/AnySplat_1218infer/datasets/nuscenes/processed_10Hz/trainval2"
# 验证集列表文件
VAL_LIST_PATH = "/home/test/LIVA/XZP/ImageStitch/Residual-Elastic-Warp/nuScenes_Val.txt"

# [输出文件]
RESULT_TXT_PATH = os.path.join(STITCHED_RESULT_ROOT, "final_metrics_report.txt")

# [图像处理参数]
# 1. Crop Wide: 从 Wide 图中心裁剪的大小
CROP_W, CROP_H = 1600, 900 
# 2. Resize Target: 裁剪后缩放到的目标大小 (用于和 GT 对比)
TARGET_W, TARGET_H = 896, 252

# [质量控制参数]
WHITE_PIXEL_THRESHOLD = 0.9  # 像素值 > 0.9 视为白色
WHITE_AREA_RATIO = 0.50       # 边缘区域白色占比阈值

# 设备
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ================= 工具函数 =================

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
    检查拼接图片是否失败 (白边检测)。
    img_tensor: [C, H, W] in [0, 1]
    """
    C, H, W = img_tensor.shape
    quarter_w = W // 4
    if quarter_w == 0: return True # 太小了无法判断，默认通过
    
    # 取左右 1/4 区域
    left_region = img_tensor[:, :, :quarter_w]
    right_region = img_tensor[:, :, -quarter_w:]
    
    # 计算白色比例 (RGB均值 > 阈值)
    # mean(0) -> [H, W_sub]
    left_white_mask = (left_region.mean(dim=0) > WHITE_PIXEL_THRESHOLD).float()
    right_white_mask = (right_region.mean(dim=0) > WHITE_PIXEL_THRESHOLD).float()
    
    left_ratio = left_white_mask.mean().item()
    right_ratio = right_white_mask.mean().item()
    
    if left_ratio > WHITE_AREA_RATIO or right_ratio > WHITE_AREA_RATIO:
        return False # 失败
    return True # 成功

def process_wide_image(img_path):
    """
    读取并处理拼接结果图:
    1. Load
    2. Check Quality (Fail -> Return None)
    3. Center Crop (1600x900)
    4. Resize (896x252)
    """
    if not os.path.exists(img_path): return None, "File Not Found"
    
    try:
        img = Image.open(img_path).convert('RGB')
        tensor = transforms.ToTensor()(img) # [C, H, W] in [0, 1]
        
        # 1. 质量检测 (在原图上做检测最准确)
        if not check_stitching_quality(tensor):
            return None, "Quality Check Failed (White borders)"
            
        # 2. Center Crop 1600x900
        _, h, w = tensor.shape
        if h < CROP_H or w < CROP_W:
            # 如果原图比裁剪框还小，用白色填充还是跳过？通常拼接图很大。
            # 这里做个中心裁剪的保护
            center_y, center_x = h // 2, w // 2
            start_y = max(0, center_y - CROP_H // 2)
            start_x = max(0, center_x - CROP_W // 2)
            # 这里的切片可能会小于 Crop 大小，如果原图不够大
            tensor_crop = tensor[:, start_y : start_y+CROP_H, start_x : start_x+CROP_W]
        else:
            center_y, center_x = h // 2, w // 2
            start_y = center_y - CROP_H // 2
            start_x = center_x - CROP_W // 2
            tensor_crop = tensor[:, start_y : start_y+CROP_H, start_x : start_x+CROP_W]
            
        # 3. Resize to 896x252
        # 使用 interpolate 需要 [B, C, H, W]
        tensor_crop = tensor_crop.unsqueeze(0) 
        tensor_final = torch.nn.functional.interpolate(
            tensor_crop, size=(TARGET_H, TARGET_W), mode='bilinear', align_corners=True
        ).squeeze(0)
        
        return tensor_final, "Success"
        
    except Exception as e:
        return None, str(e)

def process_gt_image(img_path):
    """
    读取并处理 GT 图 (Cam0 or Cam5):
    直接 Resize 到 896x252 (假设 GT 内容本身就是中心视角的)
    注意：题目要求 GT 也是 "crop再resize"? 
    通常 GT (原始NuScenes) 是 1600x900。
    如果 GT 是 1600x900，那么 "Crop 1600x900" 等于没动。
    所以这里直接 Resize。
    """
    if not os.path.exists(img_path): return None
    img = Image.open(img_path).convert('RGB')
    
    # 假设 GT 原始就是 1600x900 (NuScenes 默认)
    # 按照需求：Crop 1600x900 (即全图) -> Resize 896x252
    tensor = transforms.ToTensor()(img)
    
    tensor = tensor.unsqueeze(0)
    tensor_final = torch.nn.functional.interpolate(
        tensor, size=(TARGET_H, TARGET_W), mode='bilinear', align_corners=True
    ).squeeze(0)
    
    return tensor_final

# ================= 主流程 =================

def main():
    print(f"Metrics Calculation Started on {DEVICE}")
    print(f"Config: Batch={BATCH_SIZE} | Crop={CROP_W}x{CROP_H} | Resize={TARGET_W}x{TARGET_H}")
    
    # 初始化 LPIPS
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type='alex', reduction='none').to(DEVICE)
    lpips_fn.eval()
    
    scenes = load_scene_list(VAL_LIST_PATH)
    
    # 统计器
    stats = {
        'total_attempts': 0,
        'success_count': 0, # 拼接成功（且文件存在）的数量
        'metrics': MetricTracker(), # 仅统计成功的图片
        'scene_details': {} # {scene_id: {'success': 0, 'total': 0, 'tracker': MetricTracker()}}
    }
    
    all_tasks = []
    
    # 1. 扫描任务
    print("Scanning dataset...")
    for scene_id in tqdm(scenes, desc="Scanning Scenes"):
        # 结果目录
        result_scene_dir = os.path.join(STITCHED_RESULT_ROOT, scene_id)
        # GT 目录
        gt_scene_dir = os.path.join(DATASET_ROOT, scene_id)
        
        if not os.path.exists(result_scene_dir): continue
        
        frames = get_scene_frames(gt_scene_dir)
        
        # 初始化场景统计
        stats['scene_details'][scene_id] = {
            'success': 0, 'total': 0, 'tracker': MetricTracker()
        }
        
        for frame_id in frames:
            # 两个任务: front (cam0) 和 back (cam5)
            # Front Wide
            all_tasks.append({
                'scene_id': scene_id,
                'wide_path': os.path.join(result_scene_dir, f"{frame_id}_front_wide.jpg"),
                'gt_path': os.path.join(gt_scene_dir, "images", f"{frame_id}_0.jpg"), # Cam 0
                'type': 'front'
            })
            # Back Wide
            all_tasks.append({
                'scene_id': scene_id,
                'wide_path': os.path.join(result_scene_dir, f"{frame_id}_back_wide.jpg"),
                'gt_path': os.path.join(gt_scene_dir, "images", f"{frame_id}_5.jpg"), # Cam 5
                'type': 'back'
            })

    total_tasks = len(all_tasks)
    num_batches = math.ceil(total_tasks / BATCH_SIZE)
    print(f"Total tasks found: {total_tasks} | Batches: {num_batches}")

    # 2. Batch 处理
    for i in tqdm(range(num_batches), desc="Evaluating"):
        batch_meta = all_tasks[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
        
        wide_list = []
        gt_list = []
        valid_indices = [] # 记录在这个 batch 中有效的索引
        
        # A. 加载与预处理 (CPU密集型)
        for idx, task in enumerate(batch_meta):
            stats['total_attempts'] += 1
            stats['scene_details'][task['scene_id']]['total'] += 1
            
            # 处理 Wide 图 (含质量检测)
            wide_tensor, msg = process_wide_image(task['wide_path'])
            
            if wide_tensor is None:
                # 拼接失败或文件丢失，跳过指标计算
                # print(f"Skip {task['wide_path']}: {msg}")
                continue
            
            # 处理 GT 图
            gt_tensor = process_gt_image(task['gt_path'])
            if gt_tensor is None: continue
            
            # 加入队列
            wide_list.append(wide_tensor)
            gt_list.append(gt_tensor)
            valid_indices.append(idx)
            
            # 更新计数 (仅代表文件有效且质量检测通过，尚未计算指标)
            stats['success_count'] += 1
            stats['scene_details'][task['scene_id']]['success'] += 1

        if not wide_list: continue
        
        # B. 组装 Batch Tensor -> GPU
        wide_batch = torch.stack(wide_list).to(DEVICE) # [B, 3, H, W]
        gt_batch = torch.stack(gt_list).to(DEVICE)     # [B, 3, H, W]
        
        current_bs = wide_batch.shape[0]
        
        # C. 计算指标 (GPU)
        with torch.no_grad():
            # 1. LPIPS (Batch)
            # reduction='none' 返回 [B]
            lpips_vals = lpips_fn(wide_batch, gt_batch).view(-1)
            
            # 2. PSNR & SSIM (Loop in batch is usually fast enough for simple metrics)
            for b in range(current_bs):
                p = peak_signal_noise_ratio(wide_batch[b:b+1], gt_batch[b:b+1], data_range=1.0).item()
                s = structural_similarity_index_measure(wide_batch[b:b+1], gt_batch[b:b+1], data_range=1.0).item()
                l = lpips_vals[b].item()
                
                # 更新全局统计
                stats['metrics'].update(p, s, l)
                
                # 更新场景统计
                task_idx = valid_indices[b]
                scene_id = batch_meta[task_idx]['scene_id']
                stats['scene_details'][scene_id]['tracker'].update(p, s, l)

    # 3. 生成报告
    print("Generating Report...")
    success_rate = (stats['success_count'] / stats['total_attempts'] * 100) if stats['total_attempts'] > 0 else 0
    
    with open(RESULT_TXT_PATH, 'w') as f:
        f.write("Evaluation Report: Stitched Image Quality\n")
        f.write("==================================================\n")
        f.write(f"Source Dir: {STITCHED_RESULT_ROOT}\n")
        f.write(f"Ref Crop: {CROP_W}x{CROP_H} -> Resize: {TARGET_W}x{TARGET_H}\n")
        f.write(f"Quality Filter: White Area > {WHITE_AREA_RATIO*100}%\n\n")
        
        f.write(">>> Global Summary <<<\n")
        f.write(f"Stitching Success Rate: {success_rate:.2f}% ({stats['success_count']}/{stats['total_attempts']})\n")
        f.write(f"Metrics (Computed on Success Images Only):\n")
        f.write(f"  {format_metrics(stats['metrics'])}\n\n")
        
        f.write("==================================================\n")
        f.write(">>> Per Scene Breakdown <<<\n")
        
        for scene_id in sorted(stats['scene_details'].keys()):
            s_data = stats['scene_details'][scene_id]
            s_rate = (s_data['success'] / s_data['total'] * 100) if s_data['total'] > 0 else 0
            
            f.write(f"Scene {scene_id}:\n")
            f.write(f"  Success Rate: {s_rate:.2f}% ({s_data['success']}/{s_data['total']})\n")
            if s_data['success'] > 0:
                f.write(f"  Metrics: {format_metrics(s_data['tracker'])}\n")
            else:
                f.write(f"  Metrics: N/A (All Failed)\n")
            f.write("\n")

    print(f"Evaluation Finished. Report saved to: {RESULT_TXT_PATH}")

if __name__ == "__main__":
    main()