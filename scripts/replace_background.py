#!/usr/bin/env python3
"""
Replace the background of a video with a still image, using RobustVideoMatting
(RVM) for high-accuracy, temporally-consistent alpha matting.

Usage:
    python3 replace_background.py \
        --input /path/to/video.mp4 \
        --background /path/to/background.jpg \
        --output /path/to/output.mp4 \
        --variant resnet50 \
        --device auto
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_DIR = Path(__file__).parent.parent / "models" / "rvm"
sys.path.insert(0, str(REPO_DIR))

from model import MattingNetwork  # noqa: E402


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(variant: str, device: str) -> MattingNetwork:
    ckpt = REPO_DIR / "checkpoints" / f"rvm_{variant}.pth"
    model = MattingNetwork(variant).eval().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model


def cover_resize(image: np.ndarray, target_w: int, target_h: int,
                  anchor_x: float = 0.5, anchor_y: float = 0.5) -> np.ndarray:
    """Resize+crop image to fill target dimensions (like CSS background-size: cover).

    anchor_x/anchor_y in [0,1] control which part of the source is kept after
    cropping: 0.5 = centered, 0 = top/left, 1 = bottom/right.
    """
    h, w = image.shape[:2]
    scale = max(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    x0 = int(round((new_w - target_w) * anchor_x))
    y0 = int(round((new_h - target_h) * anchor_y))
    x0 = max(0, min(x0, new_w - target_w))
    y0 = max(0, min(y0, new_h - target_h))
    return resized[y0:y0 + target_h, x0:x0 + target_w]


def local_interior_reference(value_channel: np.ndarray, solid_mask: np.ndarray, ksize: int = 31) -> np.ndarray:
    """Box-filtered mean of value_channel over solid_mask, evaluated at every
    pixel -- a spatially-local "what should this boundary pixel's brightness
    be" reference that follows e.g. skin vs. sleeve vs. hair, unlike a single
    global interior mean. Shared with verify_quality.py's identical helper;
    duplicated rather than imported so each script stays independently
    runnable -- keep the two in sync if you change the logic."""
    solid_f = solid_mask.astype(np.float32)
    k = (ksize, ksize)
    sum_v = cv2.boxFilter(value_channel * solid_f, -1, k, normalize=False)
    sum_w = cv2.boxFilter(solid_f, -1, k, normalize=False)
    return sum_v / np.maximum(sum_w, 1e-3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Source video path")
    parser.add_argument("--background", required=True, help="Background image path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--variant", default="resnet50", choices=["resnet50", "mobilenetv3"],
                         help="resnet50 = highest accuracy (slower), mobilenetv3 = faster")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--downsample-ratio", type=float, default=None,
                         help="Matting downsample ratio; auto-picked from resolution if unset")
    parser.add_argument("--edge-feather", type=int, default=0,
                         help="Optional extra Gaussian blur (px) applied to alpha edges to soften matte")
    parser.add_argument("--edge-despill", type=float, default=0.5,
                         help="0=off, neutralizes color tint in semi-transparent boundary pixels (fixes warm/yellow hair-edge fringe from original background bleed-through)")
    parser.add_argument("--relight", action="store_true", default=True,
                         help="Color-harmonize the subject to match the new background's lighting (default on)")
    parser.add_argument("--no-relight", dest="relight", action="store_false")
    parser.add_argument("--relight-strength", type=float, default=0.18,
                         help="0=no change, 1=fully matched to background tone (default 0.55)")
    parser.add_argument("--relight-smoothing", type=float, default=0.9,
                         help="EMA smoothing factor for temporal stability of the relight shift (0-1, higher=smoother)")
    parser.add_argument("--relight-luma-weight", type=float, default=0.25,
                         help="Fraction of the relight strength applied to brightness (L); keep low so the subject doesn't go dark/dull")
    parser.add_argument("--bg-anchor-x", type=float, default=0.5, help="0-1, which part of background to keep horizontally after cover-crop")
    parser.add_argument("--bg-anchor-y", type=float, default=0.5, help="0-1, which part of background to keep vertically after cover-crop (lower value = keep more of the top)")
    parser.add_argument("--bg-crop-top", type=float, default=0.0, help="Fraction of the source background image to discard from the top before fitting (use to remove cluttered ceiling/lights)")
    parser.add_argument("--bg-crop-bottom", type=float, default=0.0, help="Fraction of the source background image to discard from the bottom before fitting")
    parser.add_argument("--content-crop", type=str, default=None,
                         help="w:h:x:y crop applied to the INPUT video before processing (e.g. to strip pillarbox bars), as from ffmpeg cropdetect")
    parser.add_argument("--bg-blur", type=float, default=0.0,
                         help="Gaussian blur sigma applied to the background -- to just match source camera "
                              "softness (small, ~1-3) or for a deliberate shallow-DOF/bokeh look (larger, ~4-6; "
                              "pair with --verify-param sharpness-ratio-max=150 so the verifier doesn't try to "
                              "'fix' your own creative choice). This is the FAR/full-strength blur level when "
                              "--bg-blur-ramp-px > 0, or the uniform level everywhere when it's 0 (default).")
    parser.add_argument("--bg-blur-near", type=float, default=0.0,
                         help="Blur sigma right at the subject's silhouette, when --bg-blur-ramp-px > 0. An "
                              "earlier version of this defaulted near=0 (knife-sharp) with a narrow ramp, which "
                              "created a visible halo/ring: straight background lines crossing the ramp zone were "
                              "sharp on one side, blurry on the other. Keep this CLOSE to --bg-blur (e.g. within "
                              "2-3 sigma of it, not near-zero) and pair with a wide --bg-blur-ramp-px -- a gentle, "
                              "wide gradient is far less visible as a discontinuity than a steep, narrow one. This "
                              "is still a 2D screen-distance proxy for depth, not real depth, so check the FULL "
                              "frame (not just a zoomed edge crop) for line discontinuities before trusting it on "
                              "a background with prominent straight lines.")
    parser.add_argument("--bg-blur-ramp-px", type=float, default=0.0,
                         help="0=off (uniform --bg-blur everywhere). >0: width in pixels over which background "
                              "blur ramps from --bg-blur-near (at the silhouette) to --bg-blur (far away), based "
                              "on each pixel's distance to the nearest subject pixel. Use a WIDE value (150-250+) "
                              "so the gradient is gradual across most of the frame rather than a tight band.")
    parser.add_argument("--fg-sharpen", type=float, default=0.0,
                         help="Unsharp-mask strength applied to the foreground (0=off, ~0.5-1.5 typical)")
    parser.add_argument("--light-wrap-strength", type=float, default=0.0,
                         help="0=off. Blends a touch of the (heavily blurred) new background's color into the "
                              "subject right at the alpha edge, fading to 0 deeper inside -- mimics ambient light "
                              "from the new environment bouncing onto the subject, which is what real composites "
                              "look like. Without this, a soft matte edge against a high-contrast background (e.g. "
                              "skin against a near-black backdrop) shows a visible warm/dark blended 'ring' even "
                              "though the matte and despill are both working correctly -- that band is mathematically "
                              "correct alpha blending, but reads as an artifact at high contrast. Try ~0.3-0.6.")
    parser.add_argument("--light-wrap-blur", type=float, default=15.0,
                         help="Gaussian blur sigma for the background glow used by --light-wrap-strength -- kept "
                              "large/diffuse on purpose, this should read as ambient bounce light, not a sharp "
                              "reflection of the background.")
    parser.add_argument("--diagnostics-out", type=str, default=None,
                         help="Path to write a JSON file with quality diagnostics (edge fringe, sharpness ratio, brightness) sampled during the run, for agentic verification")
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"[info] device={device} variant={args.variant}")

    model = load_model(args.variant, device)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sys.exit(f"Could not open input video: {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[info] input: {src_width}x{src_height} @ {fps:.3f}fps, {total_frames} frames")

    crop_w = crop_h = crop_x = crop_y = None
    if args.content_crop:
        crop_w, crop_h, crop_x, crop_y = (int(v) for v in args.content_crop.split(":"))
        width, height = crop_w, crop_h
        print(f"[info] cropping input to {crop_w}x{crop_h} at ({crop_x},{crop_y})")
    else:
        width, height = src_width, src_height

    bg_img = cv2.imread(args.background)
    if bg_img is None:
        sys.exit(f"Could not read background image: {args.background}")
    bg_native_h, bg_native_w = bg_img.shape[:2]
    if args.bg_crop_top > 0 or args.bg_crop_bottom > 0:
        bh = bg_img.shape[0]
        y0 = int(bh * args.bg_crop_top)
        y1 = bh - int(bh * args.bg_crop_bottom)
        bg_img = bg_img[y0:y1]
    # Computed AFTER bg_crop_top/bottom, against the dimensions cover_resize
    # actually scales from below -- computing this from the pre-crop native
    # size understates the real requirement whenever a crop is in effect
    # (discarding rows shrinks the effective source for the dimension being
    # cropped, so it needs MORE enlargement to cover the canvas, not less).
    bg_cropped_h, bg_cropped_w = bg_img.shape[:2]
    needed_scale = max(width / bg_cropped_w, height / bg_cropped_h)
    if needed_scale > 1.0:
        print(f"[warn] background {args.background} is {bg_native_w}x{bg_native_h} natively "
              f"({bg_cropped_w}x{bg_cropped_h} after --bg-crop-top/--bg-crop-bottom), needs {needed_scale:.2f}x "
              f"upscale to cover a {width}x{height} canvas -- cover_resize will upscale it with plain Lanczos, "
              f"which magnifies the source JPEG/WebP's own compression artifacts (reads as 'pixelated background "
              f"even though the original photo looks fine'). Run scripts/prepare_background.py first and pass its "
              f"output path instead, to AI-upscale (Real-ESRGAN) before compositing -- pass the SAME --bg-crop-top/"
              f"--bg-crop-bottom to prepare_background.py so its own margin check matches what's actually used here.")
    # cover_resize crops exactly ONE axis by construction (it scales by
    # whichever of target_w/img_w, target_h/img_h is larger, which by
    # definition makes the OTHER axis the only one with overflow to crop).
    # But --bg-crop-top/--bg-crop-bottom is a SEPARATE, manual crop on the
    # vertical axis applied before cover_resize ever runs -- if cover_resize
    # then crops the horizontal axis (the common case: a landscape photo
    # behind a portrait canvas), the NET result is cropped on three sides
    # (top/bottom AND left/right), not one, even though each individual step
    # is "correct" in isolation. Caught by user review, not by any
    # automated check, on a background whose top-crop turned out to be
    # unnecessary for that specific photo -- see README lesson #16.
    manual_crop_frac = args.bg_crop_top + args.bg_crop_bottom
    if manual_crop_frac > 0.05:
        w_ratio, h_ratio = width / bg_cropped_w, height / bg_cropped_h
        # cover_resize uses scale=max(w_ratio, h_ratio); whichever ratio is
        # the SMALLER one is the axis that ends up with overflow to crop.
        # The manual top/bottom crop always acts on height -- so the "3
        # sides cropped" case is specifically when cover_resize's own crop
        # lands on WIDTH instead (w_ratio < h_ratio), meaning two different
        # axes are both being cropped, not one.
        if w_ratio < h_ratio:
            cover_crop_frac = 1 - width / (bg_cropped_w * h_ratio)
            print(f"[warn] --bg-crop-top/--bg-crop-bottom discards {args.bg_crop_top:.0%} top / "
                  f"{args.bg_crop_bottom:.0%} bottom (a manual VERTICAL crop) while cover_resize separately crops "
                  f"~{cover_crop_frac:.0%} off the HORIZONTAL axis to fit this canvas -- the background ends up "
                  f"cropped on 3 sides total, not 1. If this background photo doesn't actually need the top/bottom "
                  f"crop (e.g. no cluttered ceiling/floor to remove), set --bg-crop-top/bottom to 0 and let "
                  f"cover_resize do its normal single-axis crop instead -- render a background-only canvas preview "
                  f"first to check (see README lesson #16).")
    bg_img = cover_resize(bg_img, width, height, args.bg_anchor_x, args.bg_anchor_y)

    light_wrap_glow_np = None
    if args.light_wrap_strength > 0:
        k = max(3, int(args.light_wrap_blur * 6) | 1)
        glow_img = cv2.GaussianBlur(bg_img, (k, k), args.light_wrap_blur)
        light_wrap_glow_np = cv2.cvtColor(glow_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    def to_bg_tensor(img):
        t = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).float().div(255)
        return t.permute(2, 0, 1).unsqueeze(0).to(device)  # 1,C,H,W

    def blur_image(img, sigma):
        if sigma <= 0:
            return img
        k = max(3, int(sigma * 6) | 1)
        return cv2.GaussianBlur(img, (k, k), sigma)

    use_blur_ramp = args.bg_blur_ramp_px > 0
    if use_blur_ramp:
        bg_near_tensor = to_bg_tensor(blur_image(bg_img, args.bg_blur_near))
        bg_far_tensor = to_bg_tensor(blur_image(bg_img, args.bg_blur))
        bg_tensor = bg_far_tensor  # used below for relight target sampling
    else:
        bg_img = blur_image(bg_img, args.bg_blur)
        bg_tensor = to_bg_tensor(bg_img)

    downsample_ratio = args.downsample_ratio
    if downsample_ratio is None:
        downsample_ratio = min(1.0, 512 / max(width, height))
    print(f"[info] downsample_ratio={downsample_ratio:.4f}")

    # Sample the background's ambient lighting/color from the region where the
    # subject will stand (center band, lower 2/3), so the foreground can be
    # color-harmonized to match instead of carrying its original lighting.
    target_lab = None
    if args.relight:
        bg_rgb = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        y0, y1 = int(height * 0.30), int(height * 0.95)
        x0, x1 = int(width * 0.20), int(width * 0.80)
        sample = bg_rgb[y0:y1, x0:x1]
        sample_lab = cv2.cvtColor(sample, cv2.COLOR_RGB2LAB)
        target_lab = sample_lab.reshape(-1, 3).mean(axis=0)
        print(f"[info] relight target LAB={target_lab}")

    # Pipe composited frames straight into a single ffmpeg encode (rawvideo in,
    # libx264 out, audio muxed from the original input) instead of writing
    # through cv2.VideoWriter's lossy default "mp4v" codec first and then
    # re-encoding that file. Two lossy encoding passes compound macroblock
    # artifacts right at high-contrast silhouette edges (skin against dark
    # hair/background) — that's what produces a visible "stair-step" border
    # around the subject even when the underlying alpha matte is smooth.
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps:.6f}",
        "-i", "-",
        "-i", args.input,
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "slow",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        args.output,
    ]
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    rec = [None] * 4
    frame_idx = 0
    amp = device != "cpu"
    autocast_dtype = torch.float16 if device == "cuda" else torch.float32
    smoothed_mean = None

    # Background sharpness baseline (Laplacian variance), computed once, used to
    # gauge whether the composited foreground looks softer/sharper than the bg.
    bg_gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
    bg_lap_var = float(cv2.Laplacian(bg_gray, cv2.CV_64F).var())
    diag = {
        "fg_lap_var_samples": [], "fg_luma_samples": [],
        "edge_chroma_samples": [], "interior_chroma_samples": [],
        "edge_brightness_halo_samples": [],
        "bg_lap_var": bg_lap_var,
    }
    diag_sample_every = 10
    # Static per-pixel background brightness, for the edge_brightness_halo
    # diagnostic below -- the unblurred-ramp bg_img is a fine approximation
    # even with --bg-blur-ramp-px active (this is a best-effort sampled
    # diagnostic, not the actual composite).
    bg_v_full = cv2.cvtColor(bg_img, cv2.COLOR_BGR2HSV)[..., 2].astype(np.float32)

    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if crop_w is not None:
                frame = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            src = torch.from_numpy(rgb).float().div(255).permute(2, 0, 1).unsqueeze(0).to(device)

            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    fgr, pha, *rec = model(src, *rec, downsample_ratio=downsample_ratio)
            else:
                fgr, pha, *rec = model(src, *rec, downsample_ratio=downsample_ratio)

            # RVM's "fgr" is a NETWORK RECONSTRUCTION of the foreground, not
            # a copy of the input -- necessary at the alpha boundary (where
            # the true foreground color is genuinely unknown, mixed with
            # background), but unnecessary and lossy in fully-opaque interior
            # pixels, where the correct foreground color is just the original
            # pixel itself. Substituting the original there (ramped in
            # smoothly between alpha 0.90-0.999, never touching genuinely
            # semi-transparent edge pixels) recovers detail the network's
            # reconstruction softens -- measured ~30% lower Laplacian-variance
            # sharpness than the source frame in solid regions before this.
            pha_np_pre = pha[0, 0].cpu().numpy()
            high_conf = np.clip((pha_np_pre - 0.90) / (0.999 - 0.90), 0, 1)[..., None]
            if high_conf.max() > 0:
                fgr_np_pre = fgr[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy().astype(np.float32)
                src_np_pre = rgb.astype(np.float32) / 255.0
                fgr_np_pre = src_np_pre * high_conf + fgr_np_pre * (1 - high_conf)
                fgr = torch.from_numpy(np.clip(fgr_np_pre, 0, 1)).permute(2, 0, 1).unsqueeze(0).to(device)

            if args.edge_feather > 0:
                k = args.edge_feather * 2 + 1
                pha_np = pha[0, 0].float().cpu().numpy()
                pha_np = cv2.GaussianBlur(pha_np, (k, k), 0)
                pha = torch.from_numpy(pha_np).to(device).unsqueeze(0).unsqueeze(0)

            if args.edge_despill > 0:
                # Fine hair-edge pixels are semi-transparent and RVM can't fully
                # decontaminate them: their "foreground" color is extrapolated
                # toward the ORIGINAL background (e.g. brighter/grayer than the
                # subject's real skin tone), not just tinted -- desaturating
                # in place (old approach: pull toward local luma) fixes hue but
                # not the brightness contamination, which still shows as a
                # bright rim once composited against a different-luminance new
                # background. Instead, inpaint the unreliable boundary band
                # from the nearest *reliable* interior color (alpha > 0.7), so
                # the edge color matches the actual skin tone, then blend that
                # in proportional to --edge-despill.
                pha_np = pha[0, 0].cpu().numpy()
                edge_strength = np.clip(1 - np.abs(pha_np * 2 - 1), 0, 1) * args.edge_despill
                edge_strength = edge_strength[..., None]
                fgr_np = fgr[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy().astype(np.float32)
                unreliable_mask = ((pha_np <= 0.7).astype(np.uint8)) * 255
                fgr_8u = np.clip(fgr_np * 255, 0, 255).astype(np.uint8)
                inpainted_8u = cv2.inpaint(fgr_8u, unreliable_mask, 7, cv2.INPAINT_TELEA)
                inpainted = inpainted_8u.astype(np.float32) / 255.0
                fgr_np = fgr_np * (1 - edge_strength) + inpainted * edge_strength
                fgr = torch.from_numpy(np.clip(fgr_np, 0, 1)).permute(2, 0, 1).unsqueeze(0).to(device)

            # Interior weight: 1.0 deep inside the subject, fading to 0 near the
            # alpha boundary. Relight/sharpen must not touch boundary pixels —
            # their "foreground" color is extrapolated/unreliable there, and
            # pushing color or contrast on them produces a colored fringe/halo
            # against the new background once alpha-blended.
            interior_weight = None
            if (args.relight and target_lab is not None) or args.fg_sharpen > 0 or args.light_wrap_strength > 0:
                mask_np = pha[0, 0].cpu().numpy()
                solid = (mask_np > 0.7).astype(np.float32)
                kernel = np.ones((7, 7), np.uint8)
                eroded = cv2.erode(solid, kernel)
                interior_weight = cv2.GaussianBlur(eroded, (31, 31), 0)[..., None]

            if args.relight and target_lab is not None:
                fgr_np = fgr[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy().astype(np.float32)
                mask_np = pha[0, 0].cpu().numpy()
                fgr_lab = cv2.cvtColor(fgr_np, cv2.COLOR_RGB2LAB)

                valid = mask_np > 0.4
                if valid.sum() > 100:
                    frame_mean = fgr_lab[valid].reshape(-1, 3).mean(axis=0)
                else:
                    frame_mean = target_lab

                if smoothed_mean is None:
                    smoothed_mean = frame_mean
                else:
                    s = args.relight_smoothing
                    smoothed_mean = s * smoothed_mean + (1 - s) * frame_mean

                delta = (target_lab - smoothed_mean) * args.relight_strength
                delta[0] *= args.relight_luma_weight  # dampen brightness shift; keep mostly color/tone
                fgr_lab = fgr_lab + delta.reshape(1, 1, 3).astype(np.float32) * interior_weight
                fgr_lab[:, :, 0] = np.clip(fgr_lab[:, :, 0], 0, 100)
                fgr_lab[:, :, 1:] = np.clip(fgr_lab[:, :, 1:], -127, 127)
                fgr_np = cv2.cvtColor(fgr_lab, cv2.COLOR_LAB2RGB)
                fgr = torch.from_numpy(np.clip(fgr_np, 0, 1)).permute(2, 0, 1).unsqueeze(0).to(device)

            if args.fg_sharpen > 0:
                fgr_np = fgr[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy().astype(np.float32)
                blurred = cv2.GaussianBlur(fgr_np, (0, 0), 1.5)
                sharpened = fgr_np + (fgr_np - blurred) * args.fg_sharpen * interior_weight
                fgr = torch.from_numpy(np.clip(sharpened, 0, 1)).permute(2, 0, 1).unsqueeze(0).to(device)

            if args.light_wrap_strength > 0:
                # Blend a touch of the (diffuse, heavily-blurred) new
                # background's color into the subject right at the alpha
                # edge -- mimics ambient bounce light from the new
                # environment, which is what makes a real composite's edge
                # against a high-contrast backdrop read as natural rather
                # than a "ring" (see --light-wrap-strength help).
                fgr_np = fgr[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy().astype(np.float32)
                wrap_weight = np.clip(1.0 - interior_weight, 0.0, 1.0) * args.light_wrap_strength
                wrapped = fgr_np * (1 - wrap_weight) + light_wrap_glow_np * wrap_weight
                fgr = torch.from_numpy(np.clip(wrapped, 0, 1)).permute(2, 0, 1).unsqueeze(0).to(device)

            if use_blur_ramp:
                mask_np = pha[0, 0].cpu().numpy()
                fg_mask = (mask_np > 0.5).astype(np.uint8)
                dist_px = cv2.distanceTransform(1 - fg_mask, cv2.DIST_L2, 5)
                t = np.clip(dist_px / args.bg_blur_ramp_px, 0, 1).astype(np.float32)
                t_tensor = torch.from_numpy(t).to(device).unsqueeze(0).unsqueeze(0)
                frame_bg_tensor = bg_near_tensor * (1 - t_tensor) + bg_far_tensor * t_tensor
            else:
                frame_bg_tensor = bg_tensor

            comp = fgr * pha + frame_bg_tensor * (1 - pha)
            comp_np = comp[0].clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
            comp_bgr = cv2.cvtColor(comp_np, cv2.COLOR_RGB2BGR)
            ffmpeg_proc.stdin.write(comp_bgr.tobytes())

            if args.diagnostics_out and frame_idx % diag_sample_every == 0:
                mask_np = pha[0, 0].cpu().numpy()
                solid = mask_np > 0.7
                if solid.sum() > 200:
                    gray = cv2.cvtColor(comp_bgr, cv2.COLOR_BGR2GRAY)
                    ys, xs = np.where(solid)
                    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
                    patch = gray[y0:y1, x0:x1]
                    if patch.size > 0:
                        diag["fg_lap_var_samples"].append(float(cv2.Laplacian(patch, cv2.CV_64F).var()))
                    hsv = cv2.cvtColor(comp_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
                    diag["fg_luma_samples"].append(float(hsv[..., 2][solid].mean()))
                    edge_band = (mask_np > 0.05) & (mask_np < 0.95)
                    if edge_band.sum() > 50:
                        diag["edge_chroma_samples"].append(float(hsv[..., 1][edge_band].mean()))
                        # Real-alpha brightness halo check: a boundary pixel
                        # is expected to land BETWEEN the subject's local
                        # interior tone and the new background's brightness
                        # at that spot (ordinary alpha blending). Flag only
                        # genuine overshoot past both -- the signature of
                        # contaminated/extrapolated matte color, not normal
                        # blending against a bright wall.
                        ref_fg_v = local_interior_reference(hsv[..., 2], solid.astype(np.float32), ksize=31)
                        blend_ceiling = np.maximum(ref_fg_v, bg_v_full)
                        excess = hsv[..., 2][edge_band] - blend_ceiling[edge_band]
                        diag["edge_brightness_halo_samples"].append(float((excess > 18.0).mean()))
                    interior = mask_np > 0.85
                    if interior.sum() > 50:
                        diag["interior_chroma_samples"].append(float(hsv[..., 1][interior].mean()))

            frame_idx += 1
            if frame_idx % 60 == 0 or frame_idx == total_frames:
                print(f"[info] processed {frame_idx}/{total_frames} frames", end="\r")

    print()
    cap.release()
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    if ffmpeg_proc.returncode != 0:
        sys.exit(f"ffmpeg encode failed (exit {ffmpeg_proc.returncode})")
    print(f"[done] wrote {args.output}")

    if args.diagnostics_out:
        def mean_or_none(xs):
            return float(np.mean(xs)) if xs else None

        fg_lap_var = mean_or_none(diag["fg_lap_var_samples"])
        summary = {
            "bg_lap_var": diag["bg_lap_var"],
            "fg_lap_var": fg_lap_var,
            "sharpness_ratio_fg_over_bg": (fg_lap_var / diag["bg_lap_var"]) if fg_lap_var and diag["bg_lap_var"] else None,
            "fg_mean_luma_0_255": mean_or_none(diag["fg_luma_samples"]),
            "edge_mean_saturation_0_255": mean_or_none(diag["edge_chroma_samples"]),
            "interior_mean_saturation_0_255": mean_or_none(diag["interior_chroma_samples"]),
            "edge_to_interior_saturation_ratio": (
                mean_or_none(diag["edge_chroma_samples"]) / mean_or_none(diag["interior_chroma_samples"])
                if diag["edge_chroma_samples"] and diag["interior_chroma_samples"] and mean_or_none(diag["interior_chroma_samples"]) > 0
                else None
            ),
            # WORST sampled frame, not the mean -- a halo confined to one
            # limb/edge (present in every frame, but a small fraction of
            # total boundary pixels) gets averaged away to nothing if you
            # mean across frames. This is computed from the REAL alpha matte
            # (this script has it; verify_quality.py only approximates it
            # from the flat output video, which is far noisier -- prefer
            # this number when both are available).
            "edge_brightness_halo_worst_frame_fraction": (
                float(max(diag["edge_brightness_halo_samples"])) if diag["edge_brightness_halo_samples"] else None
            ),
            "edge_brightness_halo_mean_frame_fraction": mean_or_none(diag["edge_brightness_halo_samples"]),
            "params": {k: v for k, v in vars(args).items()},
        }
        Path(args.diagnostics_out).write_text(json.dumps(summary, indent=2))
        print(f"[info] wrote diagnostics to {args.diagnostics_out}")


if __name__ == "__main__":
    main()
