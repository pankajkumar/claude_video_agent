#!/usr/bin/env python3
"""
Look at the actual (clip, background) pair and propose data-driven starting
values for every replace_background.py / enhance_grade.py param whose
"right" value genuinely depends on the footage itself, instead of reusing
one fixed preset tuned on a different video.

Covered (with what it's derived from):
  bg_blur             background sharpness vs. subject sharpness (Laplacian variance)
  fg_sharpen          same, residual gap not closed by bg_blur
  relight_strength    LAB color distance between subject and background
  relight_luma_weight LAB luma distance between subject and background (kept low regardless)
  relight_smoothing   clip fps -> EMA factor for a fixed real-world time constant
  edge_feather        clip resolution (px-based smoothing should scale with it)
  edge_despill        how much chroma the subject's own matte edge band carries
                       vs. its interior (a tinted source wall/background bleeds
                       through semi-transparent hair-edge pixels more on some
                       footage than others)
  bg_anchor_x/y,
  bg_crop_top         searched directly: which crop/anchor combination keeps
                       the background's head-landing region (this clip's own
                       subject bounding box, not a guessed fixed band) the
                       LEAST visually busy -- this automates the "look for a
                       plain area where the head will land" rule from the
                       README by hand instead of leaving it to eyeballing.
  denoise, sharpen,
  grain (enhance_grade) estimated source sensor noise (Immerkaer's fast noise
                       variance estimator) -- a clean modern-phone source
                       needs little denoise/grain; a noisy source needs more
                       denoise and less added synthetic grain.
  brightness (grade)  subject's measured luma vs. a target luma

NOT covered here, deliberately -- these are subjective creative choices, not
measurable footage properties, so they stay as plain static defaults you set
directly (see pipeline_agent.py's DEFAULT_PARAMS / FALLBACK_SCENE_PARAMS):
  variant, device, downsample_ratio (device/speed tradeoffs; downsample_ratio
    already auto-scales from resolution inside replace_background.py itself
    if left unset), contrast, saturation, warmth, vignette (color-grade taste).
A strong, deliberate --bg-blur for a shallow-DOF/"focus on the person" look
is also a creative choice you dial in directly, not estimated -- see README
lesson #9 for why a 2D-distance-based blur ramp (sharp near the subject,
blurred far away) was tried and reverted in favor of a uniform blur.

Every threshold/gain used below is a keyword argument with a documented
default -- none are buried as inline literals -- so different footage (e.g.
much grainier source, extreme background color cast) can override just the
relevant knob via the CLI flags below without editing code.

Usage:
    python3 estimate_params.py --input clip.mp4 --background bg.jpg \
        --content-crop 608:1080:656:0
"""
import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from replace_background import pick_device, load_model, cover_resize  # noqa: E402


def estimate_noise_sigma(gray):
    """Immerkaer's fast noise variance estimator: convolve with a discrete
    Laplacian-like kernel and normalize. Cheap, no scipy dependency."""
    h, w = gray.shape
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    conv = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    sigma = np.sum(np.abs(conv)) * math.sqrt(0.5 * math.pi) / (6 * max(w - 2, 1) * max(h - 2, 1))
    return float(sigma)


def sample_subject_stats(input_path, content_crop=None, variant="mobilenetv3", device="auto",
                          num_samples=5, mask_threshold=0.7, min_subject_px=500):
    """Run the matting model on a handful of evenly-spaced frames; return the
    subject's mean sharpness, mean LAB color, mean normalized bounding box
    (for locating where the head lands), mean noise sigma, and the matte's
    edge-band-vs-interior saturation ratio (for edge_despill)."""
    device = pick_device(device)
    model = load_model(variant, device)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise SystemExit(f"could not open {input_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total <= 0:
        raise SystemExit(f"no frames found in {input_path}")
    indices = sorted(set(int(i * total / num_samples) for i in range(num_samples)))

    crop_w = crop_h = crop_x = crop_y = None
    if content_crop:
        crop_w, crop_h, crop_x, crop_y = (int(v) for v in content_crop.split(":"))

    lap_vars, labs, bboxes, noise_sigmas = [], [], [], []
    edge_sat_ratios = []
    rec = [None] * 4
    with torch.no_grad():
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            if crop_w is not None:
                frame = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
            fh, fw = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            src = torch.from_numpy(rgb).float().div(255).permute(2, 0, 1).unsqueeze(0).to(device)
            fgr, pha, *rec = model(src, *rec, downsample_ratio=1.0)
            mask_full = pha[0, 0].cpu().numpy()
            mask = mask_full > mask_threshold
            if mask.sum() < min_subject_px:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            noise_sigmas.append(estimate_noise_sigma(gray))

            ys, xs = np.where(mask)
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            bboxes.append((y0 / fh, y1 / fh, x0 / fw, x1 / fw))

            patch_mask = mask[y0:y1, x0:x1]
            patch_gray = gray[y0:y1, x0:x1]
            lap = cv2.Laplacian(patch_gray, cv2.CV_64F)
            lap_vars.append(float(lap[patch_mask].var()) if patch_mask.any() else float(lap.var()))

            rgb_f = rgb.astype(np.float32) / 255.0
            lab = cv2.cvtColor(rgb_f, cv2.COLOR_RGB2LAB)
            labs.append(lab[mask].reshape(-1, 3).mean(axis=0))

            edge_band = (mask_full > 0.05) & (mask_full < 0.95)
            interior = mask_full > 0.85
            if edge_band.sum() > 50 and interior.sum() > 50:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
                edge_sat = hsv[..., 1][edge_band].mean()
                interior_sat = hsv[..., 1][interior].mean()
                edge_sat_ratios.append(float(edge_sat / max(interior_sat, 1e-3)))
    cap.release()

    if not lap_vars:
        raise SystemExit(
            "could not detect a subject in any sampled frame -- check --content-crop, "
            "or raise --num-samples if the subject is off-frame part of the time"
        )
    return {
        "sharpness_lap_var": float(np.mean(lap_vars)),
        "mean_lab": np.mean(labs, axis=0).tolist(),
        "mean_bbox_norm": np.mean(bboxes, axis=0).tolist(),  # (y0,y1,x0,x1) as fractions of frame
        "noise_sigma": float(np.mean(noise_sigmas)) if noise_sigmas else 0.0,
        "edge_interior_sat_ratio": float(np.mean(edge_sat_ratios)) if edge_sat_ratios else 1.0,
        "fps": fps,
    }


def _prep_background_canvas(background_path, anchor_x, anchor_y, crop_top, crop_bottom,
                              canvas_w, canvas_h):
    bg = cv2.imread(background_path)
    if bg is None:
        raise SystemExit(f"could not read background: {background_path}")
    if crop_top > 0 or crop_bottom > 0:
        bh = bg.shape[0]
        y0 = int(bh * crop_top)
        y1 = bh - int(bh * crop_bottom)
        bg = bg[y0:y1]
    return cover_resize(bg, canvas_w, canvas_h, anchor_x, anchor_y)


def sample_background_stats(background_path, anchor_x=0.5, anchor_y=0.3,
                              crop_top=0.0, crop_bottom=0.0,
                              canvas_w=608, canvas_h=1080,
                              sample_region=(0.20, 0.80, 0.30, 0.95)):
    """Background sharpness + mean LAB color in the region where a standing
    subject would land, after the same crop-top/bottom + cover-resize the
    real compositor applies -- so the comparison is apples-to-apples."""
    bg = _prep_background_canvas(background_path, anchor_x, anchor_y, crop_top, crop_bottom, canvas_w, canvas_h)
    x0f, x1f, y0f, y1f = sample_region
    h, w = bg.shape[:2]
    region = bg[int(h * y0f):int(h * y1f), int(w * x0f):int(w * x1f)]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    rgb_f = cv2.cvtColor(region, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    lab = cv2.cvtColor(rgb_f, cv2.COLOR_RGB2LAB).reshape(-1, 3).mean(axis=0)
    return {
        "sharpness_lap_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "mean_lab": lab.tolist(),
    }


def find_blur_sigma_to_match(background_path, target_lap_var, *,
                               anchor_x=0.5, anchor_y=0.3, crop_top=0.0, crop_bottom=0.0,
                               canvas_w=608, canvas_h=1080,
                               sample_region=(0.20, 0.80, 0.30, 0.95),
                               sigma_max=8.0, sigma_step=0.2, match_margin=1.0):
    """Search increasing Gaussian blur sigma on the background until its
    sharpness (in the subject-landing region) drops to roughly
    target_lap_var * match_margin."""
    bg = _prep_background_canvas(background_path, anchor_x, anchor_y, crop_top, crop_bottom, canvas_w, canvas_h)
    x0f, x1f, y0f, y1f = sample_region
    h, w = bg.shape[:2]

    sigma = 0.0
    while sigma <= sigma_max:
        blurred = bg if sigma == 0 else cv2.GaussianBlur(bg, (0, 0), sigma)
        region = blurred[int(h * y0f):int(h * y1f), int(w * x0f):int(w * x1f)]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var <= target_lap_var * match_margin:
            return round(sigma, 2)
        sigma += sigma_step
    return sigma_max


def find_least_cluttered_framing(background_path, head_box_norm, *,
                                   canvas_w=608, canvas_h=1080,
                                   anchor_x_candidates=(0.3, 0.5, 0.7),
                                   anchor_y_candidates=(0.0, 0.15, 0.3, 0.45, 0.6),
                                   crop_top_candidates=(0.0, 0.1, 0.2, 0.3),
                                   head_pad_frac=0.4):
    """Search (bg_anchor_x, bg_anchor_y, bg_crop_top) and return the
    combination that leaves the LEAST busy/detailed background right where
    THIS clip's subject's head actually lands (head_box_norm, from
    sample_subject_stats' mean_bbox_norm -- not a guessed fixed band).
    "Busy" = Laplacian variance in that box on the candidate canvas; this is
    the same quantity the README's "plain ceiling/wall, no clutter" guidance
    describes qualitatively, made into a search instead of eyeballing it.
    head_pad_frac widens the box a bit since framing varies slightly within
    a clip."""
    y0n, y1n, x0n, x1n = head_box_norm
    head_h = y1n - y0n
    head_w = x1n - x0n
    y0n = max(0.0, y0n - head_h * head_pad_frac)
    y1n = y0n + head_h * (1 + head_pad_frac)
    x0n = max(0.0, x0n - head_w * head_pad_frac)
    x1n = min(1.0, x1n + head_w * head_pad_frac)

    best = None
    for crop_top, anchor_y, anchor_x in itertools.product(crop_top_candidates, anchor_y_candidates, anchor_x_candidates):
        canvas = _prep_background_canvas(background_path, anchor_x, anchor_y, crop_top, 0.0, canvas_w, canvas_h)
        h, w = canvas.shape[:2]
        box = canvas[int(h * y0n):int(h * y1n), int(w * x0n):int(w * x1n)]
        if box.size == 0:
            continue
        gray = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
        clutter = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if best is None or clutter < best[0]:
            best = (clutter, anchor_x, anchor_y, crop_top)

    _, anchor_x, anchor_y, crop_top = best
    return {"bg_anchor_x": anchor_x, "bg_anchor_y": anchor_y, "bg_crop_top": crop_top,
            "head_region_clutter_lap_var": best[0]}


def estimate_params(input_path, background_path, *, content_crop=None,
                     variant="mobilenetv3", device="auto", num_samples=5,
                     canvas_w=608, canvas_h=1080,
                     # bg_blur / fg_sharpen
                     match_margin=1.0, sigma_max=8.0, sigma_step=0.2,
                     fg_sharpen_floor=0.3, fg_sharpen_ceiling=1.0,
                     # relight
                     relight_strength_base=0.18, relight_color_distance_ref=20.0,
                     relight_strength_min=0.05, relight_strength_max=0.4,
                     relight_luma_weight_base=0.25, relight_luma_distance_ref=15.0,
                     relight_luma_weight_min=0.05, relight_luma_weight_max=0.4,
                     relight_smoothing_time_constant_s=0.4,
                     # edge fixups
                     edge_feather_resolution_divisor=600.0, edge_feather_max=4,
                     edge_despill_base=0.5, edge_despill_min=0.2, edge_despill_max=1.0,
                     # background framing search
                     anchor_x_candidates=(0.3, 0.5, 0.7),
                     anchor_y_candidates=(0.0, 0.15, 0.3, 0.45, 0.6),
                     crop_top_candidates=(0.0, 0.1, 0.2, 0.3),
                     head_pad_frac=0.4,
                     # enhance_grade noise-informed params
                     noise_ref_sigma=2.0,
                     denoise_base=1.0, denoise_min=0.0, denoise_max=3.0,
                     sharpen_base=0.4, sharpen_min=0.1, sharpen_max=0.8,
                     grain_base=1.5, grain_min=0.0, grain_max=4.0,
                     # grade brightness
                     target_subject_luma_0_255=130.0, grade_brightness_gain=0.0004,
                     grade_brightness_max_abs=0.05):
    """Propose starting values for every footage-dependent param. See module
    docstring for which params are covered and why; everything here is a
    keyword arg with a documented default, not a buried magic number."""
    if content_crop:
        cw, ch, _, _ = (int(v) for v in content_crop.split(":"))
        canvas_w, canvas_h = cw, ch

    subject = sample_subject_stats(input_path, content_crop, variant, device, num_samples)
    bg = sample_background_stats(background_path, 0.5, 0.3, 0.0, 0.0, canvas_w, canvas_h)

    bg_blur = find_blur_sigma_to_match(
        background_path, subject["sharpness_lap_var"],
        canvas_w=canvas_w, canvas_h=canvas_h, sigma_max=sigma_max, sigma_step=sigma_step, match_margin=match_margin,
    )
    fg_sharpen = fg_sharpen_floor
    if bg["sharpness_lap_var"] < subject["sharpness_lap_var"] * match_margin:
        deficit = subject["sharpness_lap_var"] / max(bg["sharpness_lap_var"], 1.0)
        fg_sharpen = float(np.clip(fg_sharpen_floor * deficit, fg_sharpen_floor, fg_sharpen_ceiling))

    subject_lab = np.array(subject["mean_lab"])
    bg_lab = np.array(bg["mean_lab"])
    color_distance = float(np.linalg.norm(subject_lab - bg_lab))
    luma_distance = float(abs(subject_lab[0] - bg_lab[0]))

    relight_strength = float(np.clip(
        relight_strength_base * (relight_color_distance_ref / max(color_distance, 1.0)),
        relight_strength_min, relight_strength_max,
    ))
    # Bigger luma gap -> LOWER luma_weight (don't push brightness hard toward
    # a very differently-lit background -- see README lesson #1).
    relight_luma_weight = float(np.clip(
        relight_luma_weight_base * (relight_luma_distance_ref / max(luma_distance, 1.0)),
        relight_luma_weight_min, relight_luma_weight_max,
    ))
    relight_smoothing = float(np.clip(
        math.exp(-1.0 / max(subject["fps"] * relight_smoothing_time_constant_s, 1e-3)), 0.0, 0.99
    ))

    edge_feather = int(np.clip(round(min(canvas_w, canvas_h) / edge_feather_resolution_divisor), 1, edge_feather_max))
    edge_despill = float(np.clip(
        edge_despill_base * subject["edge_interior_sat_ratio"], edge_despill_min, edge_despill_max
    ))

    framing = find_least_cluttered_framing(
        background_path, subject["mean_bbox_norm"], canvas_w=canvas_w, canvas_h=canvas_h,
        anchor_x_candidates=anchor_x_candidates, anchor_y_candidates=anchor_y_candidates,
        crop_top_candidates=crop_top_candidates, head_pad_frac=head_pad_frac,
    )

    noise_sigma = subject["noise_sigma"]
    noise_ratio = noise_sigma / max(noise_ref_sigma, 1e-6)
    denoise = float(np.clip(denoise_base * noise_ratio, denoise_min, denoise_max))
    # Noisier source -> sharpen a bit less (sharpening amplifies noise).
    sharpen = float(np.clip(sharpen_base / (1 + noise_ratio), sharpen_min, sharpen_max))
    # Noisier source already carries natural grain -> add less synthetic grain.
    grain = float(np.clip(grain_base / (1 + noise_ratio), grain_min, grain_max))

    # Caveat: this is the mean luma over the WHOLE matted subject (clothing
    # included), not just skin/face -- dark clothing biases it low. Treated
    # as a gentle nudge only (low gain, tight cap) for exactly that reason;
    # the brightness check + adjust_params() in pipeline_agent.py can still
    # correct further during the auto-tune loop if this initial guess is off.
    subject_luma_0_255 = float(np.clip(subject_lab[0] / 100.0 * 255.0, 0, 255))
    brightness = float(np.clip(
        (target_subject_luma_0_255 - subject_luma_0_255) * grade_brightness_gain,
        -grade_brightness_max_abs, grade_brightness_max_abs,
    ))

    return {
        "bg_blur": bg_blur,
        "fg_sharpen": round(fg_sharpen, 2),
        "relight_strength": round(relight_strength, 3),
        "relight_luma_weight": round(relight_luma_weight, 3),
        "relight_smoothing": round(relight_smoothing, 3),
        "edge_feather": edge_feather,
        "edge_despill": round(edge_despill, 3),
        "bg_anchor_x": framing["bg_anchor_x"],
        "bg_anchor_y": framing["bg_anchor_y"],
        "bg_crop_top": framing["bg_crop_top"],
        "denoise": round(denoise, 2),
        "sharpen": round(sharpen, 2),
        "grain": round(grain, 2),
        "brightness": round(brightness, 4),
        "_diagnostics": {
            "subject_sharpness_lap_var": subject["sharpness_lap_var"],
            "background_sharpness_lap_var": bg["sharpness_lap_var"],
            "subject_mean_lab": subject["mean_lab"],
            "background_mean_lab": bg["mean_lab"],
            "color_distance_lab": color_distance,
            "luma_distance_lab": luma_distance,
            "fps": subject["fps"],
            "noise_sigma": noise_sigma,
            "edge_interior_sat_ratio": subject["edge_interior_sat_ratio"],
            "head_region_clutter_lap_var": framing["head_region_clutter_lap_var"],
            "subject_mean_bbox_norm": subject["mean_bbox_norm"],
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--background", required=True)
    p.add_argument("--content-crop", default=None)
    p.add_argument("--variant", default="mobilenetv3", choices=["resnet50", "mobilenetv3"],
                    help="mobilenetv3 is enough for this sampling pass and much faster")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--num-samples", type=int, default=5, help="frames sampled across the clip")
    args = p.parse_args()

    result = estimate_params(
        args.input, args.background, content_crop=args.content_crop,
        variant=args.variant, device=args.device, num_samples=args.num_samples,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
