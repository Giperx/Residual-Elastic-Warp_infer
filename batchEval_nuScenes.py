import os
import sys
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import re
import traceback
import math
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.transforms import GaussianBlur

import os
import sys
import torch
# CUDA_VISIBLE_DEVICES=0 python batchEval_nuScenes.py
# ==========================================
# [全局修复] 劫持 torch.inverse 到 CPU
# ==========================================
# _original_inverse = torch.inverse
# def safe_inverse_cpu(A):
#     """
#     强制 CPU 求逆，彻底规避 MAGMA。
#     对于单应性矩阵（3x3/4x4）这种小矩阵，CPU 比 GPU 更稳定。
#     """
#     device = A.device
#     dtype = A.dtype
    
#     # 转 CPU 计算
#     A_cpu = A.detach().cpu()
    
#     # 检查数值稳定性
#     if not torch.isfinite(A_cpu).all():
#         print("[WARNING] Non-finite values in matrix, using pseudo-inverse")
#         result = torch.linalg.pinv(A_cpu)
#     else:
#         try:
#             result = torch.linalg.inv(A_cpu)
#         except Exception as e:
#             print(f"[WARNING] Inversion failed: {e}, using pseudo-inverse")
#             result = torch.linalg.pinv(A_cpu)
    
#     # 转回原设备
#     return result.to(dtype).to(device)
# # 全局替换
# torch.inverse = safe_inverse_cpu
# print("✅ torch.inverse has been redirected to CPU-based implementation")

if not hasattr(torch, 'gesv'):
    def _gesv_compat(B, A):
        # Solve AX = B => X = A^{-1} B
        # torch.linalg.solve expects (A, B) and solves AX = B
        X = torch.linalg.solve(A, B)
        # gesv 返回 (solution, LU) -> we only mimic the first output
        return (X, None)
    torch.gesv = _gesv_compat
# ==========================================
import utils
import models

# ================= 配置区域 =================
# [Batch Size]
# 即使是 1，之前的 MAGMA 也会崩。现在切到 CPU 了，Batch Size 依然建议保持较小(1-4)，
# 因为 grid_sample 依然占用大量显存。
BATCH_SIZE = 1 

CHECKPOINT_PATH = "/home/test/LIVA/XZP/ImageStitch/Residual-Elastic-Warp/pretrained/rewarp.pth" 
DATASET_ROOT = "/home/test/LIVA/XZP/FeedForward/fine_tune3/AnySplat_1218infer/datasets/nuscenes/processed_10Hz/trainval2"
VAL_LIST_PATH = "/home/test/LIVA/XZP/ImageStitch/Residual-Elastic-Warp/nuScenes_ValTmp.txt"
SAVE_ROOT = "./nuscenes_stitched_results_with_crop_stats0130tmp"
REPORT_PATH = os.path.join(SAVE_ROOT, "stitching_report.txt")

DEVICE = 'cuda'

MAX_CANVAS_SIZE = 10000 
WHITE_PIXEL_THRESHOLD = 0.95 
WHITE_AREA_RATIO = 0.70 

# ================= 统计类 =================
class StitchStats:
    def __init__(self):
        self.global_stats = {'total': 0, 'success': 0, 'fail': 0}
        self.scene_stats = {} 

    def init_scene(self, scene_id):
        if scene_id not in self.scene_stats:
            self.scene_stats[scene_id] = {'total': 0, 'success': 0, 'fail': 0, 'errors': []}

    def update(self, scene_id, success, error_msg=None):
        self.init_scene(scene_id)
        
        self.scene_stats[scene_id]['total'] += 1
        self.global_stats['total'] += 1
        
        if success:
            self.scene_stats[scene_id]['success'] += 1
            self.global_stats['success'] += 1
        else:
            self.scene_stats[scene_id]['fail'] += 1
            self.global_stats['fail'] += 1
            if error_msg:
                # 截断错误信息防止日志过大
                self.scene_stats[scene_id]['errors'].append(error_msg[:200])

    def save_report(self, path):
        try:
            with open(path, 'w') as f:
                f.write("Stitching Evaluation Report\n")
                f.write("==================================================\n")
                g = self.global_stats
                rate = (g['success'] / g['total'] * 100) if g['total'] > 0 else 0
                f.write(f"Global Summary:\n")
                f.write(f"  Total Pairs: {g['total']}\n")
                f.write(f"  Success: {g['success']}\n")
                f.write(f"  Fail: {g['fail']}\n")
                f.write(f"  Success Rate: {rate:.2f}%\n")
                f.write("==================================================\n")
                f.write("Per Scene Breakdown:\n")
                for sid, stats in self.scene_stats.items():
                    s_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
                    f.write(f"\nScene {sid}: {s_rate:.2f}% ({stats['success']}/{stats['total']})\n")
                    if stats['fail'] > 0:
                        for err in stats['errors'][:3]:
                            f.write(f"    - {err}\n")
        except Exception:
            print("Error saving report.")

# ================= 基础工具 =================

def process_image_nuscenes(img_path):
    img = Image.open(img_path).convert('RGB')
    tensor = transforms.ToTensor()(img) 
    tensor = (tensor - 0.5) * 2.0       
    return tensor.unsqueeze(0).to(DEVICE)

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

def get_image_paths_for_group(scene_dir, frame_id, group_type):
    img_dir = os.path.join(scene_dir, "images")
    if group_type == 'front': indices = [0, 1, 2] 
    else: indices = [5, 4, 3] 
    paths = []
    for idx in indices:
        found = False
        for ext in ['.jpg', '.png', '.jpeg']:
            p = os.path.join(img_dir, f"{frame_id}_{idx}{ext}")
            if os.path.exists(p):
                paths.append(p)
                found = True
                break
        if not found: return None
    return paths

# ================= 拼接与裁剪核心 =================

class StitchingError(Exception):
    pass

def linear_blender(ref, tgt, ref_m, tgt_m):
    try:
        blur = GaussianBlur(kernel_size=(21,21), sigma=20)
        if ref_m.sum() == 0 or tgt_m.sum() == 0:
            return ref * ref_m + tgt * tgt_m 

        r1, c1 = torch.nonzero(ref_m[0, 0], as_tuple=True)
        r2, c2 = torch.nonzero(tgt_m[0, 0], as_tuple=True)
        
        if len(r1) == 0 or len(r2) == 0:
            return ref * ref_m + tgt * tgt_m

        center1 = (r1.float().mean(), c1.float().mean())
        center2 = (r2.float().mean(), c2.float().mean())
        vec = (center2[0] - center1[0], center2[1] - center1[1])

        ovl = (ref_m * tgt_m).round()[:, 0].unsqueeze(1)
        ref_m_ = ref_m[:, 0].unsqueeze(1) - ovl
        r, c = torch.nonzero(ovl[0, 0], as_tuple=True)

        if len(r) == 0: return ref * ref_m + tgt * tgt_m

        ovl_mask = torch.zeros_like(ref_m_).to(DEVICE)
        proj_val = (r - center1[0]) * vec[0] + (c - center1[1]) * vec[1]
        denom = proj_val.max() - proj_val.min()
        if denom < 1e-3: denom = 1e-3
        
        ovl_mask[ovl.bool()] = (proj_val - proj_val.min()) / denom
        mask1 = (blur(ref_m_ + (1-ovl_mask)*ref_m[:,0].unsqueeze(1)) * ref_m + ref_m_).clamp(0,1)
        mask2 = (1-mask1) * tgt_m
        return ref * mask1 + tgt * mask2
    except Exception:
        return ref * ref_m + tgt * tgt_m

def stitch_pair(ref_img, tgt_img, model_H, model_TPS):
    b, c, h, w = ref_img.shape
    
    if h != 512 or w != 512:
        ref_in = F.interpolate(ref_img, size=(512,512), mode='bilinear')
        tgt_in = F.interpolate(tgt_img, size=(512,512), mode='bilinear')
    else:
        ref_in, tgt_in = ref_img, tgt_img
    
    scale_h = h / ref_in.shape[-2]

    try:
        _, disps, _ = model_H(ref_in, tgt_in, iters_lev0=6)
    except Exception as e:
        raise StitchingError(f"Model H inference failed: {str(e)}")

    H_mat, img_h, img_w, offset = utils.get_warped_coords(disps[-1], scale=(h/512, w/512), size=(h,w))
    
    if img_h > MAX_CANVAS_SIZE or img_w > MAX_CANVAS_SIZE:
        raise StitchingError(f"Canvas size too large: {img_w}x{img_h}")
    
    H_inv, *_ = utils.get_H(disps[-1].reshape(b,2,-1).permute(0,2,1), [512, 512])
    H_inv = utils.compens_H(H_inv, [512, 512])
    
    try:
        grid = utils.make_coordinate_grid([512, 512], type=H_inv.type())
        grid = grid.reshape(1, -1, 2).repeat(b, 1, 1)
        mesh_homography = utils.warp_coord(grid, H_inv.cuda()).reshape(b, 512, 512, -1)
        tgt_w_512 = F.grid_sample(tgt_in, mesh_homography, align_corners=True)

        flows = model_TPS(tgt_w_512, ref_in, iters=3, scale=scale_h)
        
        translation = utils.get_translation(*offset)
        T_ref = translation.clone()
        T_tgt = torch.inverse(H_mat).double() @ translation.cuda() 

        sizes = (img_h, img_w)
        
        coord1 = utils.to_pixel_samples(None, sizes=sizes).cuda()
        mesh_r, _ = utils.gridy2gridx_homography(coord1.contiguous(), *sizes, h, w, T_ref.cuda(), cpu=False)
        mesh_r = mesh_r.reshape(b, img_h, img_w, 2).cuda().flip(-1)

        coord2 = utils.to_pixel_samples(None, sizes=sizes).cuda()
        mesh_t, _ = utils.gridy2gridx_homography(coord2.contiguous(), *sizes, h, w, T_tgt.cuda(), cpu=False)
        mesh_t = mesh_t.reshape(b, img_h, img_w, 2).cuda().flip(-1)

        flow = flows[-1] / 511
        flow_full = F.interpolate(flow.permute(0,3,1,2), size=(img_h, img_w), mode='bilinear').permute(0,2,3,1) * 2
        mesh_t = mesh_t + flow_full

        ones = torch.ones(b, 1, h, w).cuda()
        ref_w = F.grid_sample(ref_img, mesh_r, mode='bilinear', align_corners=True)
        mask_r = F.grid_sample(ones, mesh_r, mode='nearest', align_corners=True)
        tgt_w = F.grid_sample(tgt_img, mesh_t, mode='bilinear', align_corners=True)
        mask_t = F.grid_sample(ones, mesh_t, mode='nearest', align_corners=True)

        ref_w_01 = (ref_w + 1) / 2 * mask_r
        tgt_w_01 = (tgt_w + 1) / 2 * mask_t

        stitched_01 = linear_blender(ref_w_01, tgt_w_01, mask_r, mask_t)
        stitched_out = stitched_01 * 2 - 1
        stitched_mask = ((mask_r + mask_t) > 0).float()
        
        stitched_out = stitched_out * stitched_mask + (1.0) * (1 - stitched_mask)
        
    except RuntimeError as e:
        if 'out of memory' in str(e):
            torch.cuda.empty_cache()
            raise StitchingError("CUDA Out of Memory")
        else:
            raise StitchingError(f"Runtime Error during warp: {str(e)}")

    return stitched_out, stitched_mask, offset

def save_expanded_reference_crop(final_img, ref_top_left, ref_h, ref_w, save_path):
    try:
        if final_img.dim() == 4:
            final_img = final_img.squeeze(0)
            
        C, H_src, W_src = final_img.shape
        src_ref_y, src_ref_x = int(ref_top_left[0]), int(ref_top_left[1])
        
        H_dst = ref_h
        W_dst = ref_w * 2
        
        final_img_01 = (final_img.float() + 1) / 2
        output_canvas = torch.ones(C, H_dst, W_dst).to(DEVICE)
        
        dst_ref_y = (H_dst - ref_h) // 2
        dst_ref_x = (W_dst - ref_w) // 2
        
        shift_y = dst_ref_y - src_ref_y
        shift_x = dst_ref_x - src_ref_x
        
        dst_y_start = max(0, shift_y)
        dst_y_end = min(H_dst, H_src + shift_y)
        dst_x_start = max(0, shift_x)
        dst_x_end = min(W_dst, W_src + shift_x)
        
        if dst_x_end <= dst_x_start or dst_y_end <= dst_y_start:
            save_image(output_canvas, save_path)
            return False, "Ref image completely out of canvas bounds"

        src_y_start = dst_y_start - shift_y
        src_y_end = dst_y_end - shift_y
        src_x_start = dst_x_start - shift_x
        src_x_end = dst_x_end - shift_x
        
        output_canvas[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            final_img_01[:, src_y_start:src_y_end, src_x_start:src_x_end]
        
        quarter_w = W_dst // 4
        if quarter_w > 0:
            left_region = output_canvas[:, :, :quarter_w]
            right_region = output_canvas[:, :, -quarter_w:]
            
            left_white_mask = (left_region.mean(dim=0) > WHITE_PIXEL_THRESHOLD).float()
            right_white_mask = (right_region.mean(dim=0) > WHITE_PIXEL_THRESHOLD).float()
            
            if left_white_mask.mean().item() > WHITE_AREA_RATIO:
                save_image(output_canvas, save_path)
                return False, f"Left expansion failed"
            if right_white_mask.mean().item() > WHITE_AREA_RATIO:
                save_image(output_canvas, save_path)
                return False, f"Right expansion failed"

        save_image(output_canvas, save_path)
        return True, "Success"
    except Exception as e:
        return False, f"Save crop failed: {str(e)}"

# ================= 主程序 =================

def main():
    stats = StitchStats()
    
    try:
        print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model_H = models.make(checkpoint['H_model'], load_sd=True).to(DEVICE)
        model_T = models.make(checkpoint['T_model'], load_sd=True).to(DEVICE)
        model_H.eval()
        model_T.eval()
    except Exception as e:
        print(f"Critical Error loading model: {e}")
        return
    
    scenes = load_scene_list(VAL_LIST_PATH)
    os.makedirs(SAVE_ROOT, exist_ok=True)
    
    print(f"Processing {len(scenes)} scenes...")
    
    try:
        for scene_id in tqdm(scenes, desc="Scenes"):
            scene_dir = os.path.join(DATASET_ROOT, scene_id)
            frames = get_scene_frames(scene_dir)
            scene_save_dir = os.path.join(SAVE_ROOT, scene_id)
            os.makedirs(scene_save_dir, exist_ok=True)
            
            stats.init_scene(scene_id)
            
            # 定期清理显存
            if len(frames) > 0: torch.cuda.empty_cache()

            for frame_id in frames:
                groups = [('front', 'front'), ('back', 'back')]
                
                for group_name, _ in groups:
                    task_id = f"{frame_id}_{group_name}"
                    
                    try:
                        paths = get_image_paths_for_group(scene_dir, frame_id, group_name)
                        if paths is None: 
                            stats.update(scene_id, False, f"{task_id}: Missing input images")
                            continue
                        
                        img0 = process_image_nuscenes(paths[0]) 
                        img1 = process_image_nuscenes(paths[1])
                        img2 = process_image_nuscenes(paths[2])
                        ref_h, ref_w = img0.shape[-2], img0.shape[-1]
                        
                        with torch.no_grad():
                            stitched_1, mask_1, offset_1 = stitch_pair(img0, img1, model_H, model_T)
                            stitched_final, mask_final, offset_2 = stitch_pair(stitched_1, img2, model_H, model_T)
                            
                            final_ref_y = offset_1[0] + offset_2[0]
                            final_ref_x = offset_1[1] + offset_2[1]
                            
                            save_tensor = (stitched_final + 1) / 2
                            file_name = f"{frame_id}_{group_name}_stitched.jpg"
                            save_path = os.path.join(scene_save_dir, file_name)
                            save_image(save_tensor, save_path)
                            
                            crop_name = f"{frame_id}_{group_name}_wide.jpg"
                            crop_path = os.path.join(scene_save_dir, crop_name)
                            
                            is_valid, msg = save_expanded_reference_crop(
                                stitched_final, 
                                (final_ref_y, final_ref_x), 
                                ref_h, ref_w, 
                                crop_path
                            )
                            
                            if is_valid:
                                stats.update(scene_id, True)
                            else:
                                stats.update(scene_id, False, f"{task_id}: {msg}")
                    
                    except StitchingError as se:
                        stats.update(scene_id, False, f"{task_id}: {str(se)}")
                        torch.cuda.empty_cache()
                    except Exception as e:
                        err_msg = f"{task_id}: Unhandled error {str(e)}"
                        stats.update(scene_id, False, err_msg)
                        torch.cuda.empty_cache()

    except Exception as global_e:
        print(f"Global Crash intercepted: {str(global_e)}")
        traceback.print_exc()
    finally:
        stats.save_report(REPORT_PATH)
        print(f"Done. Report saved to {REPORT_PATH}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main()