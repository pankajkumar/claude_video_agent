#!/usr/bin/env python3
"""
Dump every intermediate image of the bg_replace + enhance_grade pipeline for
ONE frame, as individually numbered, LOSSLESS PNG files, so a human can find
exactly which step (if any) is degrading quality.

Why PNG, not the step_*.jpg name pattern someone might expect: saving an
intermediate debug image as JPEG would add JPEG's own compression artifacts
on top of whatever the pipeline step actually did, confusing the very
analysis this script exists for. Every file here is .png (lossless); only
the actual final delivered video is ever JPEG/H.264-compressed.

Covers, for the SAME single frame:
  BACKGROUND side: raw source -> AI-upscale -> top/bottom crop -> cover_resize
    -> bg_blur -> final composited-against background.
  PERSON side: raw decoded video frame -> content-crop -> model input ->
    raw alpha/fgr straight off the matting network -> fgr original-pixel
    substitution -> alpha feather -> edge despill -> relight -> fg_sharpen ->
    raw composite -> each enhance_grade.py filter stage individually.

Usage:
    python3 debug_pipeline_frame.py --input video.mp4 --frame 90 \
      --background bg.png --content-crop 608:1080:656:0 --debug-dir out/
"""
import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from replace_background import pick_device, load_model, cover_resize, local_interior_reference  # noqa: E402


def save(debug_dir, step, name, img_bgr):
    path = debug_dir / f"step_{step:02d}_{name}.png"
    cv2.imwrite(str(path), img_bgr)
    print(f"[debug] wrote {path}")
    return path


def to_bgr_u8(rgb_float_hwc):
    return cv2.cvtColor(np.clip(rgb_float_hwc * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--frame", type=int, required=True)
    p.add_argument("--background", required=True, help="ORIGINAL (non-upscaled) background source image")
    p.add_argument("--content-crop", required=True, help="w:h:x:y")
    p.add_argument("--debug-dir", required=True)
    p.add_argument("--variant", default="resnet50")
    p.add_argument("--bg-crop-top", type=float, default=0.0)
    p.add_argument("--bg-crop-bottom", type=float, default=0.0)
    p.add_argument("--bg-anchor-x", type=float, default=0.5)
    p.add_argument("--bg-anchor-y", type=float, default=0.5)
    p.add_argument("--bg-blur", type=float, default=1.8)
    p.add_argument("--fg-sharpen", type=float, default=0.3)
    p.add_argument("--edge-feather", type=int, default=3)
    p.add_argument("--edge-despill", type=float, default=0.7)
    p.add_argument("--relight-strength", type=float, default=0.078)
    p.add_argument("--relight-luma-weight", type=float, default=0.082)
    # grade params, applied incrementally at the end
    p.add_argument("--denoise", type=float, default=0.06)
    p.add_argument("--sharpen", type=float, default=0.38)
    p.add_argument("--contrast", type=float, default=1.06)
    p.add_argument("--saturation", type=float, default=1.10)
    p.add_argument("--brightness", type=float, default=0.0393)
    p.add_argument("--warmth", type=float, default=0.03)
    p.add_argument("--vignette", type=float, default=1.0)
    p.add_argument("--grain", type=float, default=0.0)
    args = p.parse_args()

    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    step = 0

    crop_w, crop_h, crop_x, crop_y = (int(v) for v in args.content_crop.split(":"))

    # ---------------- BACKGROUND SIDE ----------------
    bg_orig = cv2.imread(args.background)
    step += 1
    save(debug_dir, step, "bg_a_original_source", bg_orig)

    sys.path.insert(0, str(SCRIPT_DIR))
    from prepare_background import prepare_background
    upscaled_path = prepare_background(args.background, crop_w, crop_h, margin=1.0,
                                        bg_crop_top=args.bg_crop_top, bg_crop_bottom=args.bg_crop_bottom)
    bg_upscaled = cv2.imread(upscaled_path)
    step += 1
    save(debug_dir, step, f"bg_b_upscaled_{Path(upscaled_path).stem.split('_')[-1]}", bg_upscaled)

    bg_after_crop = bg_upscaled
    if args.bg_crop_top > 0 or args.bg_crop_bottom > 0:
        bh = bg_upscaled.shape[0]
        y0 = int(bh * args.bg_crop_top)
        y1 = bh - int(bh * args.bg_crop_bottom)
        bg_after_crop = bg_upscaled[y0:y1]
    step += 1
    save(debug_dir, step, "bg_c_after_crop_top_bottom", bg_after_crop)

    bg_cover = cover_resize(bg_after_crop, crop_w, crop_h, args.bg_anchor_x, args.bg_anchor_y)
    step += 1
    save(debug_dir, step, "bg_d_cover_resized_to_canvas", bg_cover)

    bg_blurred = bg_cover
    if args.bg_blur > 0:
        k = max(3, int(args.bg_blur * 6) | 1)
        bg_blurred = cv2.GaussianBlur(bg_cover, (k, k), args.bg_blur)
    step += 1
    save(debug_dir, step, "bg_e_after_bg_blur", bg_blurred)

    # ---------------- PERSON SIDE ----------------
    cap = cv2.VideoCapture(args.input)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, raw_frame = cap.read()
    if not ok:
        sys.exit("could not read frame")
    step += 1
    save(debug_dir, step, "person_a_raw_decoded_frame_native_res", raw_frame)

    cropped_frame = raw_frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
    step += 1
    save(debug_dir, step, "person_b_content_cropped", cropped_frame)

    rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    device = pick_device("auto")
    model = load_model(args.variant, device)
    torch.set_grad_enabled(False)
    src = torch.from_numpy(rgb).float().div(255).permute(2, 0, 1).unsqueeze(0).to(device)
    step += 1
    save(debug_dir, step, "person_c_model_input_sanity_check", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    downsample_ratio = min(1.0, 512 / max(crop_w, crop_h))
    rec = [None] * 4
    fgr, pha, *rec = model(src, *rec, downsample_ratio=downsample_ratio)

    pha_np = pha[0, 0].cpu().numpy()
    step += 1
    save(debug_dir, step, "person_d_raw_alpha_matte", (np.clip(pha_np, 0, 1) * 255).astype(np.uint8))

    fgr_np = fgr[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy().astype(np.float32)
    step += 1
    save(debug_dir, step, "person_e_raw_fgr_from_model", to_bgr_u8(fgr_np))

    # fgr original-pixel substitution (the lesson #15 fix)
    high_conf = np.clip((pha_np - 0.90) / (0.999 - 0.90), 0, 1)[..., None]
    src_np = rgb.astype(np.float32) / 255.0
    fgr_np = src_np * high_conf + fgr_np * (1 - high_conf)
    step += 1
    save(debug_dir, step, "person_f_fgr_after_original_pixel_substitution", to_bgr_u8(fgr_np))

    # edge feather (on alpha)
    if args.edge_feather > 0:
        k = args.edge_feather * 2 + 1
        pha_np = cv2.GaussianBlur(pha_np, (k, k), 0)
    step += 1
    save(debug_dir, step, "person_g_alpha_after_edge_feather", (np.clip(pha_np, 0, 1) * 255).astype(np.uint8))

    # edge despill (inpaint)
    if args.edge_despill > 0:
        edge_strength = np.clip(1 - np.abs(pha_np * 2 - 1), 0, 1) * args.edge_despill
        edge_strength = edge_strength[..., None]
        unreliable_mask = ((pha_np <= 0.7).astype(np.uint8)) * 255
        fgr_8u = np.clip(fgr_np * 255, 0, 255).astype(np.uint8)
        inpainted_8u = cv2.inpaint(fgr_8u, unreliable_mask, 7, cv2.INPAINT_TELEA)
        inpainted = inpainted_8u.astype(np.float32) / 255.0
        fgr_np = fgr_np * (1 - edge_strength) + inpainted * edge_strength
    step += 1
    save(debug_dir, step, "person_h_fgr_after_edge_despill", to_bgr_u8(fgr_np))

    # interior weight (for relight/sharpen masking)
    solid = (pha_np > 0.7).astype(np.float32)
    kernel = np.ones((7, 7), np.uint8)
    eroded = cv2.erode(solid, kernel)
    interior_weight = cv2.GaussianBlur(eroded, (31, 31), 0)[..., None]

    # relight
    target_lab = cv2.cvtColor(bg_blurred.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(axis=0)
    fgr_lab = cv2.cvtColor(fgr_np, cv2.COLOR_RGB2LAB)
    valid = pha_np > 0.4
    frame_mean = fgr_lab[valid].reshape(-1, 3).mean(axis=0) if valid.sum() > 100 else target_lab
    delta = (target_lab - frame_mean) * args.relight_strength
    delta[0] *= args.relight_luma_weight
    fgr_lab = fgr_lab + delta.reshape(1, 1, 3).astype(np.float32) * interior_weight
    fgr_lab[:, :, 0] = np.clip(fgr_lab[:, :, 0], 0, 100)
    fgr_lab[:, :, 1:] = np.clip(fgr_lab[:, :, 1:], -127, 127)
    fgr_np = cv2.cvtColor(fgr_lab, cv2.COLOR_LAB2RGB)
    step += 1
    save(debug_dir, step, "person_i_fgr_after_relight", to_bgr_u8(fgr_np))

    # fg_sharpen
    if args.fg_sharpen > 0:
        blurred = cv2.GaussianBlur(fgr_np, (0, 0), 1.5)
        sharpened = fgr_np + (fgr_np - blurred) * args.fg_sharpen * interior_weight
        fgr_np = np.clip(sharpened, 0, 1)
    step += 1
    save(debug_dir, step, "person_j_fgr_after_fg_sharpen", to_bgr_u8(fgr_np))

    # raw composite
    pha_3 = pha_np[..., None]
    bg_rgb = cv2.cvtColor(bg_blurred, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    comp_rgb = fgr_np * pha_3 + bg_rgb * (1 - pha_3)
    step += 1
    composite_path = save(debug_dir, step, "person_k_raw_composite_pre_grade", to_bgr_u8(comp_rgb))

    # ---------------- enhance_grade.py, incremental ----------------
    filters_incremental = [
        ("denoise_hqdn3d", f"hqdn3d={args.denoise}:{args.denoise}:3:3"),
        ("sharpen_unsharp", f"unsharp=5:5:{args.sharpen}:5:5:0.0"),
        ("grade_curves", "curves=master='0/0.02 0.25/0.27 0.5/0.54 0.75/0.80 1/1'"),
        ("grade_eq", f"eq=contrast={args.contrast}:saturation={args.saturation}:brightness={args.brightness}"),
    ]
    if args.warmth != 0:
        w = args.warmth
        filters_incremental.append(("grade_colorbalance_warmth",
            f"colorbalance=rs={w:.3f}:bs={-w*1.3:.3f}:rm={w*0.6:.3f}:bm={-w*0.6:.3f}:rh={-w*0.3:.3f}:bh={w*0.9:.3f}"))
    if args.vignette > 0:
        filters_incremental.append(("grade_vignette", f"vignette=PI/{max(2.0, 7/args.vignette):.2f}:mode=forward"))
    if args.grain > 0:
        filters_incremental.append(("grade_grain_noise", f"noise=alls={args.grain}:allf=t+u"))

    cumulative = []
    for name, vf in filters_incremental:
        cumulative.append(vf)
        vf_chain = ",".join(cumulative)
        step += 1
        out_path = debug_dir / f"step_{step:02d}_person_l_after_{name}.png"
        cmd = ["ffmpeg", "-y", "-i", str(composite_path), "-vf", vf_chain, "-frames:v", "1", str(out_path)]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"[debug] wrote {out_path}")

    print(f"\n[done] {step} debug images written to {debug_dir}")


if __name__ == "__main__":
    main()
