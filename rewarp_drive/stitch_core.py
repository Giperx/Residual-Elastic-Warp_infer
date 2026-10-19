"""Camera 5/4/3 stitch math for the driving pipeline.

The behaviour is ported from the remote ``eval_nuScenes.py`` and the remote
``stitch_drive.py`` drivers: 512-pixel model inference, the elastic warp scaled
back to the source resolution, white background on invalid pixels, a masked
linear blend with seam fusion, and a chunked large-canvas path used when the
blended canvas does not fit in GPU memory.

Only the model load/import plumbing lives in ``drive.py``; everything that
touches pixels lives here so the scripts stay thin.
"""
import sys
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import GaussianBlur

from . import compat

# ``models`` relies on torch.gesv, which recent PyTorch removed. Install the
# shim before the models package is imported.
compat.install()

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import models  # noqa: E402
import utils  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_CANVAS_SIZE = 10000  # mirrors the remote driver default
WHITE_LEVEL = 0.98  # min-channel threshold that marks background white
CHUNK_ROWS = 64

CAMERAS = (5, 4, 3)
CHECKPOINT = _REPO_ROOT / "pretrained" / "rewarp.pth"


class StitchingError(Exception):
    """Raised when a pair cannot be stitched (bad canvas, OOM, bad shape)."""


def _meshgrid(y_seq, x_seq):
    try:
        return torch.meshgrid(y_seq, x_seq, indexing="ij")
    except TypeError:
        return torch.meshgrid(y_seq, x_seq)


def _as_int(value):
    return int(value.item()) if torch.is_tensor(value) else int(value)


def process_image(img_path, device=None):
    """Open an RGB image and return a ``[1, 3, H, W]`` tensor in ``[-1, 1]``."""
    image = Image.open(img_path).convert("RGB")
    array = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
    tensor = (tensor - 0.5) * 2.0
    return tensor.unsqueeze(0).to(device if device is not None else DEVICE)


def load_models(checkpoint=None, device=None):
    checkpoint = Path(checkpoint) if checkpoint is not None else CHECKPOINT
    device = device if device is not None else DEVICE
    state = torch.load(str(checkpoint), map_location=device)
    model_h = models.make(state["H_model"], load_sd=True).to(device).eval()
    model_t = models.make(state["T_model"], load_sd=True).to(device).eval()
    return model_h, model_t


def linear_blender(ref, tgt, ref_m, tgt_m):
    """Masked linear blend with a directional seam ramp (remote behaviour)."""
    blur = GaussianBlur(kernel_size=(21, 21), sigma=20)
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
    if len(r) == 0:
        return ref * ref_m + tgt * tgt_m

    ovl_mask = torch.zeros_like(ref_m_)
    proj_val = (r - center1[0]) * vec[0] + (c - center1[1]) * vec[1]
    denom = proj_val.max() - proj_val.min()
    if denom < 1e-3:
        denom = 1e-3
    ovl_mask[ovl.bool()] = (proj_val - proj_val.min()) / denom
    mask1 = (blur(ref_m_ + (1 - ovl_mask) * ref_m[:, 0].unsqueeze(1)) * ref_m + ref_m_).clamp(0, 1)
    mask2 = (1 - mask1) * tgt_m
    return ref * mask1 + tgt * mask2


def stitch_pair(ref_img, tgt_img, model_h, model_tps):
    """Stitch two ``[-1, 1]`` images, returning ``(image, mask, offset)``."""
    b, c, h, w = ref_img.shape

    if h != 512 or w != 512:
        ref_in = F.interpolate(ref_img, size=(512, 512), mode="bilinear")
        tgt_in = F.interpolate(tgt_img, size=(512, 512), mode="bilinear")
    else:
        ref_in, tgt_in = ref_img, tgt_img

    scale_h = h / ref_in.shape[-2]

    try:
        _, disps, _ = model_h(ref_in, tgt_in, iters_lev0=6)
    except Exception as error:
        raise StitchingError(f"Model H inference failed: {error}")

    h_mat, img_h, img_w, offset = utils.get_warped_coords(disps[-1], scale=(h / 512, w / 512), size=(h, w))

    if img_h > MAX_CANVAS_SIZE or img_w > MAX_CANVAS_SIZE:
        raise StitchingError(f"Canvas size too large: {img_w}x{img_h} > {MAX_CANVAS_SIZE}")

    h_inv, *_ = utils.get_H(disps[-1].reshape(b, 2, -1).permute(0, 2, 1), [512, 512])
    h_inv = utils.compens_H(h_inv, [512, 512])

    try:
        grid = utils.make_coordinate_grid([512, 512], type=h_inv.type())
        grid = grid.reshape(1, -1, 2).repeat(b, 1, 1)
        mesh_homography = utils.warp_coord(grid, h_inv.to(DEVICE)).reshape(b, 512, 512, -1)
        tgt_w_512 = F.grid_sample(tgt_in, mesh_homography, align_corners=True)

        flows = model_tps(tgt_w_512, ref_in, iters=3, scale=scale_h)

        translation = utils.get_translation(*offset)
        t_ref = translation.clone()
        t_tgt = torch.inverse(h_mat).double() @ translation.to(DEVICE)

        sizes = (img_h, img_w)

        coord1 = utils.to_pixel_samples(None, sizes=sizes).to(DEVICE)
        mesh_r, _ = utils.gridy2gridx_homography(coord1.contiguous(), *sizes, h, w, t_ref.to(DEVICE), cpu=False)
        mesh_r = mesh_r.reshape(b, img_h, img_w, 2).to(DEVICE).flip(-1)

        coord2 = utils.to_pixel_samples(None, sizes=sizes).to(DEVICE)
        mesh_t, _ = utils.gridy2gridx_homography(coord2.contiguous(), *sizes, h, w, t_tgt.to(DEVICE), cpu=False)
        mesh_t = mesh_t.reshape(b, img_h, img_w, 2).to(DEVICE).flip(-1)

        flow = flows[-1] / 511
        flow_full = F.interpolate(flow.permute(0, 3, 1, 2), size=(img_h, img_w), mode="bilinear").permute(0, 2, 3, 1) * 2
        mesh_t = mesh_t + flow_full

        ones = torch.ones(b, 1, h, w).to(ref_img.device)
        ref_w = F.grid_sample(ref_img, mesh_r, mode="bilinear", align_corners=True)
        mask_r = F.grid_sample(ones, mesh_r, mode="nearest", align_corners=True)
        tgt_w = F.grid_sample(tgt_img, mesh_t, mode="bilinear", align_corners=True)
        mask_t = F.grid_sample(ones, mesh_t, mode="nearest", align_corners=True)

        ref_w_01 = (ref_w + 1) / 2 * mask_r
        tgt_w_01 = (tgt_w + 1) / 2 * mask_t

        stitched_01 = linear_blender(ref_w_01, tgt_w_01, mask_r, mask_t)
        stitched_out = stitched_01 * 2 - 1
        stitched_mask = ((mask_r + mask_t) > 0).float()

        # Invalid background is white (1.0 in [-1, 1]).
        stitched_out = stitched_out * stitched_mask + 1.0 * (1 - stitched_mask)

    except RuntimeError as error:
        if "out of memory" in str(error):
            torch.cuda.empty_cache()
            raise StitchingError("CUDA Out of Memory")
        raise StitchingError(f"Runtime Error during warp: {error}")

    return stitched_out, stitched_mask, offset


def pair_geometry(ref_img, tgt_img, model_h, model_t):
    """Estimate the blended canvas without allocating it."""
    if ref_img.device.type != "cuda":
        ref_img = ref_img.cuda()
    if tgt_img.device.type != "cuda":
        tgt_img = tgt_img.cuda()
    batch, _, height, width = ref_img.shape
    if height != 512 or width != 512:
        ref_in = F.interpolate(ref_img, size=(512, 512), mode="bilinear", align_corners=False)
        tgt_in = F.interpolate(tgt_img, size=(512, 512), mode="bilinear", align_corners=False)
    else:
        ref_in, tgt_in = ref_img, tgt_img
    try:
        _, disps, _ = model_h(ref_in, tgt_in, iters_lev0=6)
    except Exception as error:
        raise StitchingError(f"Model H inference failed: {error}")
    homography, canvas_h, canvas_w, offset = utils.get_warped_coords(
        disps[-1], scale=(height / 512, width / 512), size=(height, width)
    )
    if canvas_h < 2 or canvas_w < 2:
        raise StitchingError(f"invalid canvas {canvas_w}x{canvas_h}")
    homography_inv, *_ = utils.get_H(disps[-1].reshape(batch, 2, -1).permute(0, 2, 1), [512, 512])
    homography_inv = utils.compens_H(homography_inv, [512, 512])
    grid = utils.make_coordinate_grid([512, 512], type=homography_inv.type())
    grid = grid.reshape(1, -1, 2).repeat(batch, 1, 1)
    mesh = utils.warp_coord(grid, homography_inv.cuda()).reshape(batch, 512, 512, -1)
    target_512 = F.grid_sample(tgt_in, mesh, align_corners=True)
    flows = model_t(target_512, ref_in, iters=3, scale=height / ref_in.shape[-2])
    translation = utils.get_translation(*offset)
    return {
        "ref": ref_img,
        "tgt": tgt_img,
        "height": height,
        "width": width,
        "canvas_h": canvas_h,
        "canvas_w": canvas_w,
        "offset": offset,
        "reference_transform": translation.clone(),
        "target_transform": torch.inverse(homography).double() @ translation.cuda(),
        "flow": flows[-1] / 511,
    }


def _sample_rows(geom, y0, y1):
    canvas_h, canvas_w = geom["canvas_h"], geom["canvas_w"]
    height, width = geom["height"], geom["width"]
    y_seq = -1 + (2.0 / canvas_h) * torch.arange(y0, y1, device="cuda")
    x_seq = -1 + (2.0 / canvas_w) * torch.arange(canvas_w, device="cuda")
    yy, xx = _meshgrid(y_seq, x_seq)
    samples = torch.stack([yy, xx], dim=-1).reshape(-1, 2).contiguous()

    def source_mesh(transform):
        mapped, _ = utils.gridy2gridx_homography(
            samples.clone(), canvas_h, canvas_w, height, width, transform.cuda(), cpu=False
        )
        return mapped.reshape(1, y1 - y0, canvas_w, 2).flip(-1)

    reference_mesh = source_mesh(geom["reference_transform"])
    target_mesh = source_mesh(geom["target_transform"])
    sample_y = (torch.arange(y0, y1, device="cuda") + 0.5) / canvas_h * 2 - 1
    sample_x = (torch.arange(canvas_w, device="cuda") + 0.5) / canvas_w * 2 - 1
    sample_yy, sample_xx = _meshgrid(sample_y, sample_x)
    flow_grid = torch.stack([sample_xx, sample_yy], dim=-1).unsqueeze(0)
    flow_rows = F.grid_sample(
        geom["flow"].permute(0, 3, 1, 2), flow_grid, mode="bilinear", align_corners=False
    ).permute(0, 2, 3, 1) * 2
    target_mesh = target_mesh + flow_rows
    ones = torch.ones(1, 1, height, width, device="cuda")
    reference = F.grid_sample(geom["ref"], reference_mesh, mode="bilinear", align_corners=True)
    reference_mask = F.grid_sample(ones, reference_mesh, mode="nearest", align_corners=True)
    target = F.grid_sample(geom["tgt"], target_mesh, mode="bilinear", align_corners=True)
    target_mask = F.grid_sample(ones, target_mesh, mode="nearest", align_corners=True)
    return reference, reference_mask, target, target_mask


def _blend_rows(reference, reference_mask, target, target_mask, centers, proj_range, y_offset):
    ref_01 = (reference + 1) / 2 * reference_mask
    tgt_01 = (target + 1) / 2 * target_mask
    if centers is None or reference_mask.sum() == 0 or target_mask.sum() == 0:
        return ref_01 * reference_mask + tgt_01 * target_mask
    center_ref, center_tgt = centers
    vector = (center_tgt[0] - center_ref[0], center_tgt[1] - center_ref[1])
    overlap = (reference_mask * target_mask).round()[:, 0]
    rows, cols = torch.nonzero(overlap[0], as_tuple=True)
    overlap_mask = torch.zeros_like(reference_mask)
    if len(rows):
        projected = (rows + y_offset - center_ref[0]) * vector[0] + (cols - center_ref[1]) * vector[1]
        low, high = proj_range
        overlap_mask[0, 0, rows, cols] = (projected - low) / max(high - low, 1e-3)
    exclusive = reference_mask - overlap.unsqueeze(1)
    blurred = GaussianBlur(kernel_size=(21, 21), sigma=20)(
        exclusive + (1 - overlap_mask) * reference_mask
    )
    mask_ref = (blurred * reference_mask + exclusive).clamp(0, 1)
    mask_tgt = (1 - mask_ref) * target_mask
    return ref_01 * mask_ref + tgt_01 * mask_tgt


def _mask_stats(mask, y0):
    rows, cols = torch.nonzero(mask[0, 0], as_tuple=True)
    if len(rows) == 0:
        return 0.0, 0.0, 0
    return float((rows + y0).sum()), float(cols.sum()), int(len(rows))


def stitch_large(geom):
    """Blend a canvas that does not fit in GPU memory, keeping the seam fusion."""
    canvas_h, canvas_w = geom["canvas_h"], geom["canvas_w"]
    if canvas_h * canvas_w > 1_200_000_000:
        raise StitchingError(f"Canvas {canvas_w}x{canvas_h} is too large to store")
    chunk = CHUNK_ROWS
    ref_y = ref_x = tgt_y = tgt_x = 0.0
    ref_n = tgt_n = 0
    for y0 in range(0, canvas_h, chunk):
        y1 = min(canvas_h, y0 + chunk)
        _, reference_mask, _, target_mask = _sample_rows(geom, y0, y1)
        sy, sx, count = _mask_stats(reference_mask, y0)
        ref_y += sy
        ref_x += sx
        ref_n += count
        sy, sx, count = _mask_stats(target_mask, y0)
        tgt_y += sy
        tgt_x += sx
        tgt_n += count
        del reference_mask, target_mask
    if ref_n == 0 or tgt_n == 0:
        raise StitchingError("one side has no valid pixels")
    centers = ((ref_y / ref_n, ref_x / ref_n), (tgt_y / tgt_n, tgt_x / tgt_n))
    vector = (centers[1][0] - centers[0][0], centers[1][1] - centers[0][1])
    low, high = None, None
    for y0 in range(0, canvas_h, chunk):
        y1 = min(canvas_h, y0 + chunk)
        _, reference_mask, _, target_mask = _sample_rows(geom, y0, y1)
        overlap = (reference_mask * target_mask).round()[0, 0]
        rows, cols = torch.nonzero(overlap, as_tuple=True)
        if len(rows):
            projected = (rows + y0 - centers[0][0]) * vector[0] + (cols - centers[0][1]) * vector[1]
            lo, hi = float(projected.min()), float(projected.max())
            low = lo if low is None else min(low, lo)
            high = hi if high is None else max(high, hi)
        del reference_mask, target_mask, overlap
    image = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    halo = 16
    for y0 in range(0, canvas_h, chunk):
        y1 = min(canvas_h, y0 + chunk)
        pad0 = max(0, y0 - halo)
        pad1 = min(canvas_h, y1 + halo)
        reference, reference_mask, target, target_mask = _sample_rows(geom, pad0, pad1)
        blended = _blend_rows(
            reference, reference_mask, target, target_mask, centers, (low or 0.0, high or 1.0), pad0
        )
        # Invalid samples are 0. Paint them white, matching stitch_pair.
        valid = ((reference_mask + target_mask) > 0).float()
        blended = blended * valid + (1.0 - valid)
        inner0 = y0 - pad0
        inner1 = inner0 + (y1 - y0)
        part = blended[0, :, inner0:inner1].clamp(0, 1)
        image[y0:y1] = (part.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        del reference, reference_mask, target, target_mask, blended
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255
    return tensor.unsqueeze(0) * 2 - 1, geom["offset"]


def stitch_blended(ref_img, tgt_img, model_h, model_t):
    """Stitch a pair, switching to the chunked path for very large canvases."""
    geom = pair_geometry(ref_img, tgt_img, model_h, model_t)
    pixels = geom["canvas_h"] * geom["canvas_w"]
    print(f"canvas {geom['canvas_w']}x{geom['canvas_h']}", flush=True)
    if pixels <= 8_000_000:
        torch.cuda.empty_cache()
        return stitch_pair(geom["ref"], geom["tgt"], model_h, model_t)
    rendered, offset = stitch_large(geom)
    return rendered, None, offset


def side_counts(image, origin, ref_w):
    """Count non-white pixels on each side of the unwarped camera 5 rectangle."""
    picture = image[0] if image.dim() == 4 else image
    picture = (picture.detach().float().cpu() + 1) / 2
    content = picture.min(dim=0).values < WHITE_LEVEL
    origin_x = origin[1]
    left = int(content[:, :origin_x].sum())
    right = int(content[:, origin_x + ref_w :].sum())
    return left, right


def composite_by_reference(image_a, origin_a, image_b, origin_b):
    """Place two camera-5 stitches so their camera-5 rectangles coincide."""
    first = image_a[0] if image_a.dim() == 4 else image_a
    second = image_b[0] if image_b.dim() == 4 else image_b
    first = (first.detach().float().cpu() + 1) / 2
    second = (second.detach().float().cpu() + 1) / 2

    def extent(picture, origin):
        _, height, width = picture.shape
        return -origin[0], -origin[1], height - origin[0], width - origin[1]

    bounds = [extent(first, origin_a), extent(second, origin_b)]
    y0 = min(item[0] for item in bounds)
    x0 = min(item[1] for item in bounds)
    y1 = max(item[2] for item in bounds)
    x1 = max(item[3] for item in bounds)
    canvas = torch.ones(3, y1 - y0, x1 - x0)

    def paste(picture, origin):
        _, height, width = picture.shape
        top, left = -origin[0] - y0, -origin[1] - x0
        incoming = picture.min(dim=0).values < WHITE_LEVEL
        current = canvas[:, top : top + height, left : left + width]
        occupied = current.min(dim=0).values < WHITE_LEVEL
        both = occupied & incoming
        only = incoming & ~both
        current[:, only] = picture[:, only]
        current[:, both] = 0.5 * (current[:, both] + picture[:, both])
        canvas[:, top : top + height, left : left + width] = current

    paste(first, origin_a)
    paste(second, origin_b)
    return (canvas * 2 - 1).unsqueeze(0), (-y0, -x0)


def crop_triple(final, origin, ref_h, ref_w):
    """Cut one reference-width on each side of camera 5. Missing pixels stay white."""
    image = final[0] if final.dim() == 4 else final
    image = (image.detach().float().cpu() + 1) / 2
    _, src_h, src_w = image.shape
    origin_y, origin_x = origin
    output = torch.ones(3, ref_h, ref_w * 3)
    src_y0, src_x0 = origin_y, origin_x - ref_w
    dst_y0, dst_x0 = max(0, -src_y0), max(0, -src_x0)
    dst_y1, dst_x1 = min(ref_h, src_h - src_y0), min(ref_w * 3, src_w - src_x0)
    if dst_y1 > dst_y0 and dst_x1 > dst_x0:
        output[:, dst_y0:dst_y1, dst_x0:dst_x1] = image[
            :, src_y0 + dst_y0 : src_y0 + dst_y1, src_x0 + dst_x0 : src_x0 + dst_x1
        ]
    return output


def reference_masks(final, origin, ref_h, ref_w, threshold=WHITE_LEVEL):
    """Build the camera-5 rectangle mask and the non-white content mask.

    Returns ``(cam5, valid)`` as boolean CPU tensors on the final canvas. The
    camera-5 mask is the reference rectangle at ``origin`` clipped to the
    canvas; the valid mask is ``min(channel) < threshold`` after mapping the
    ``[-1, 1]`` canvas to ``[0, 1]``.
    """
    image = final[0] if final.dim() == 4 else final
    _, height, width = image.shape
    cam5 = torch.zeros(height, width, dtype=torch.bool)
    y0 = max(0, int(origin[0]))
    x0 = max(0, int(origin[1]))
    y1 = min(height, int(origin[0]) + ref_h)
    x1 = min(width, int(origin[1]) + ref_w)
    if y1 > y0 and x1 > x0:
        cam5[y0:y1, x0:x1] = True
    picture = (image.detach().float().cpu() + 1) / 2
    valid = picture.min(dim=0).values < threshold
    return cam5, valid


def to_rgb_uint8(tensor):
    """Convert a ``[0, 1]`` tensor (or batch) to an HxWx3 uint8 RGB array."""
    image = tensor[0] if tensor.dim() == 4 else tensor
    array = (image.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return array


def crop_cam5_lr_max(image, valid, cam5):
    """Vertical: camera-5 bounding box. Horizontal: maximum stitched image."""
    ys, xs = np.where(cam5)
    if len(xs) == 0:
        raise StitchingError("camera 5 mask is empty")
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    content_ys, content_xs = np.where(valid)
    if len(content_xs) == 0:
        raise StitchingError("stitched image is empty")
    x0 = int(content_xs.min())
    x1 = int(content_xs.max()) + 1
    y0 = max(0, min(y0, image.shape[0] - 1))
    y1 = max(y0 + 1, min(image.shape[0], y1))
    x0 = max(0, min(x0, image.shape[1] - 1))
    x1 = max(x0 + 1, min(image.shape[1], x1))
    return image[y0:y1, x0:x1]


def resize_to(image, width, height):
    """Resize with INTER_AREA when shrinking, else INTER_LINEAR."""
    if image.shape[1] == width and image.shape[0] == height:
        return image
    shrinking = image.shape[1] > width or image.shape[0] > height
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def save_metric_image(final_m11, origin, ref_h, ref_w, gt_w, gt_h, x3_path, metric_png_path):
    """Crop one camera-5 width on each side, then resize that image to the GT.

    Camera 5 is only translated, so its rectangle is still ``ref_h`` by ``ref_w``.
    The crop is exactly that height and three times that width, with camera 5 in
    the center third. Pixels the stitch does not cover stay white. The resized
    image is what the metric reads; the unresized crop is kept for viewing.
    """
    triple = crop_triple(final_m11, origin, ref_h, ref_w)
    rgb = to_rgb_uint8(triple)
    if x3_path is not None:
        x3_path = Path(x3_path)
        x3_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.ascontiguousarray(rgb)).save(x3_path, quality=95)
    if metric_png_path is None:
        return rgb.shape
    resized = resize_to(rgb, gt_w, gt_h)
    metric_png_path = Path(metric_png_path)
    metric_png_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(resized)).save(metric_png_path)
    return resized.shape


def save_picture(tensor, path, quality=95):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_rgb_uint8(tensor)).save(path, quality=quality)
