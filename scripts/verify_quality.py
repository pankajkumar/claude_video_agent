#!/usr/bin/env python3
"""
Agentic quality verifier for background-replaced video.

Samples frames from the output video and checks for the failure modes that came
up repeatedly while building this pipeline:
  - edge fringe / color halo around the subject (warm tint bleeding through
    semi-transparent hair/edge pixels)
  - foreground noticeably softer or harsher than the background (mismatched
    sharpness makes the composite look pasted-in)
  - subject too dark/bright relative to a natural-looking frame
  - basic file sanity (resolution, duration, fps vs the source)

It does NOT have access to the matting alpha (the final video is flat RGB), so
the subject mask is approximated by diffing each frame against the known
background image. This makes it reusable as a generic post-hoc check on any
background-replacement output, not just ones produced by replace_background.py.

Usage:
    python3 verify_quality.py --video final.mp4 --background bg.jpg --report report.json
    python3 verify_quality.py --video final.mp4 --background bg.jpg --source-video original.mp4

Exit code 0 = all checks passed, 1 = at least one check failed (see report for which).
"""
import argparse
import json
import sys

import cv2
import numpy as np


def cover_resize(image, target_w, target_h, anchor_x=0.5, anchor_y=0.5):
    """Mirrors replace_background.py's cover_resize -- same anchor semantics
    (0=top/left .. 1=bottom/right) so this samples the SAME region of the
    background that was actually composited, not just a centered crop."""
    h, w = image.shape[:2]
    scale = max(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    x0 = int(round((new_w - target_w) * anchor_x))
    y0 = int(round((new_h - target_h) * anchor_y))
    x0 = max(0, min(x0, new_w - target_w))
    y0 = max(0, min(y0, new_h - target_h))
    return resized[y0:y0 + target_h, x0:x0 + target_w]


def approx_subject_mask(frame_bgr, bg_bgr, thresh=28):
    diff = cv2.absdiff(frame_bgr, bg_bgr).astype(np.int32).sum(axis=2)
    mask = (diff > thresh).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return mask


def sample_frame_indices(total_frames, n=12):
    if total_frames <= n:
        return list(range(total_frames))
    return [int(i * total_frames / n) for i in range(n)]


def local_interior_reference(value_channel, solid_mask, ksize=31):
    """Box-filtered mean of value_channel over solid_mask, evaluated at every
    pixel -- a spatially-local "what should this boundary pixel's brightness
    be" reference that follows e.g. skin vs. sleeve vs. hair, unlike a single
    global interior mean. Used to catch a brightness halo confined to one
    limb/edge, which a whole-frame average dilutes into nothing."""
    solid_f = (solid_mask > 0).astype(np.float32)
    k = (ksize, ksize)
    sum_v = cv2.boxFilter(value_channel * solid_f, -1, k, normalize=False)
    sum_w = cv2.boxFilter(solid_f, -1, k, normalize=False)
    return sum_v / np.maximum(sum_w, 1e-3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--background", required=True, help="The background image used for compositing")
    p.add_argument("--source-video", default=None, help="Original pre-composite video, for resolution/duration/fps sanity check")
    p.add_argument("--samples", type=int, default=12)
    p.add_argument("--report", default=None, help="Path to write JSON report")
    p.add_argument("--fringe-edge-ratio-max", type=float, default=1.8,
                    help="edge_fringe fails above this edge/interior saturation ratio")
    p.add_argument("--fringe-warm-excess-max", type=float, default=0.35,
                    help="edge_fringe fails above this excess warm-hue fraction at the boundary")
    p.add_argument("--halo-brightness-delta", type=float, default=18.0,
                    help="edge_brightness_halo: a boundary pixel counts as 'haloed' if its V (0-255) "
                         "exceeds the LOCAL interior reference brightness by more than this")
    p.add_argument("--halo-fraction-max", type=float, default=0.12,
                    help="edge_brightness_halo fails if more than this fraction of boundary pixels "
                         "are haloed, taking the WORST sampled frame (not averaged across frames -- "
                         "a halo confined to one limb in every frame must not be diluted away by frames "
                         "where that limb isn't even in view)")
    p.add_argument("--halo-ref-ksize", type=int, default=31,
                    help="box-filter kernel size (px) for the local interior brightness reference")
    p.add_argument("--sharpness-ratio-min", type=float, default=0.08,
                    help="sharpness_match fails below this fg/bg Laplacian-variance ratio (subject too soft)")
    p.add_argument("--sharpness-ratio-max", type=float, default=6.0,
                    help="sharpness_match fails above this fg/bg Laplacian-variance ratio (subject oversharpened/noisy)")
    p.add_argument("--brightness-min", type=float, default=45.0,
                    help="brightness check fails below this mean subject luma (0-255)")
    p.add_argument("--brightness-max", type=float, default=235.0,
                    help="brightness check fails above this mean subject luma (0-255)")
    p.add_argument("--duration-tolerance", type=float, default=0.5,
                    help="duration_match fails if output/source duration differ by more than this many seconds")
    p.add_argument("--bg-blur", type=float, default=0.0,
                    help="Gaussian blur sigma to replicate on --background before sampling it for sharpness_match, "
                         "matching whatever --bg-blur replace_background.py actually used for this render -- "
                         "without this, sharpness_match always compares against the ORIGINAL unblurred image "
                         "and increasing --bg-blur in the render has no effect on the check.")
    p.add_argument("--bg-anchor-x", type=float, default=0.5,
                    help="must match replace_background.py's --bg-anchor-x for this render, else the sampled "
                         "background patch is the wrong region of the image")
    p.add_argument("--bg-anchor-y", type=float, default=0.5,
                    help="must match replace_background.py's --bg-anchor-y for this render")
    p.add_argument("--bg-crop-top", type=float, default=0.0,
                    help="must match replace_background.py's --bg-crop-top for this render")
    p.add_argument("--bg-crop-bottom", type=float, default=0.0,
                    help="must match replace_background.py's --bg-crop-bottom for this render")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Could not open video: {args.video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    bg = cv2.imread(args.background)
    if bg is None:
        sys.exit(f"Could not read background: {args.background}")
    if args.bg_crop_top > 0 or args.bg_crop_bottom > 0:
        bh = bg.shape[0]
        y0 = int(bh * args.bg_crop_top)
        y1 = bh - int(bh * args.bg_crop_bottom)
        bg = bg[y0:y1]
    bg = cover_resize(bg, w, h, args.bg_anchor_x, args.bg_anchor_y)
    if args.bg_blur > 0:
        k = max(3, int(args.bg_blur * 6) | 1)
        bg = cv2.GaussianBlur(bg, (k, k), args.bg_blur)

    idxs = sample_frame_indices(total, args.samples)
    fringe_scores, sharp_ratios, lumas = [], [], []
    halo_fractions = []

    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        mask = approx_subject_mask(frame, bg)
        if mask.sum() < 255 * 500:
            continue  # subject not detectable in this frame, skip

        solid = cv2.erode(mask, np.ones((9, 9), np.uint8))
        boundary = cv2.subtract(cv2.dilate(mask, np.ones((9, 9), np.uint8)), solid)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        bg_hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV).astype(np.float32)
        if (boundary > 0).sum() > 50 and (solid > 0).sum() > 200:
            edge_sat = hsv[..., 1][boundary > 0].mean()
            interior_sat = hsv[..., 1][solid > 0].mean()
            edge_hue = hsv[..., 0][boundary > 0]
            warm_fraction = float(((edge_hue > 12) & (edge_hue < 45)).mean())  # yellow/orange band
            # The background itself may legitimately be warm-toned (e.g. warm
            # gym lights) — only flag fringe if the edge is warmer than the
            # background already is at those same pixel locations.
            bg_edge_hue = bg_hsv[..., 0][boundary > 0]
            bg_warm_fraction = float(((bg_edge_hue > 12) & (bg_edge_hue < 45)).mean())
            fringe_scores.append({
                "edge_to_interior_saturation_ratio": float(edge_sat / max(interior_sat, 1e-3)),
                "warm_hue_fraction_at_edge": warm_fraction,
                "warm_hue_fraction_excess": float(max(0.0, warm_fraction - bg_warm_fraction)),
            })
            lumas.append(float(hsv[..., 2][solid > 0].mean()))

            # A boundary pixel is semi-transparent by construction, so it's
            # expected to read SOMEWHERE BETWEEN the subject's local skin/
            # fabric tone and the new background's local brightness at that
            # spot (that's just normal alpha blending, not a defect, and is
            # very common e.g. dark hair against a bright wall). Only flag
            # genuine overshoot: brighter than BOTH endpoints of that blend
            # range, which is what contaminated/extrapolated matte color
            # looks like once composited, not what correct blending produces.
            ref_fg_v = local_interior_reference(hsv[..., 2], solid, ksize=args.halo_ref_ksize)
            ref_bg_v = cv2.boxFilter(bg_hsv[..., 2], -1, (args.halo_ref_ksize, args.halo_ref_ksize))
            blend_ceiling = np.maximum(ref_fg_v, ref_bg_v)
            boundary_excess = hsv[..., 2][boundary > 0] - blend_ceiling[boundary > 0]
            halo_fractions.append(float((boundary_excess > args.halo_brightness_delta).mean()))

        ys, xs = np.where(solid > 0)
        if len(ys) > 200:
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            # Compare against the LOCAL background patch behind the subject
            # (same coordinates), not the whole image — a face is naturally
            # lower-frequency than e.g. gym equipment, so a whole-image
            # comparison unfairly flags every face as "too soft".
            fg_patch = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            bg_patch = cv2.cvtColor(bg[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            fg_var = cv2.Laplacian(fg_patch, cv2.CV_64F).var()
            bg_var = cv2.Laplacian(bg_patch, cv2.CV_64F).var()
            if bg_var > 1:
                sharp_ratios.append(float(fg_var / bg_var))

    cap.release()

    def mean_or_none(xs):
        return float(np.mean(xs)) if xs else None

    edge_ratio = mean_or_none([f["edge_to_interior_saturation_ratio"] for f in fringe_scores])
    warm_excess = mean_or_none([f["warm_hue_fraction_excess"] for f in fringe_scores])
    sharp_ratio = mean_or_none(sharp_ratios)
    mean_luma = mean_or_none(lumas)

    checks = {}
    checks["edge_fringe"] = {
        "metric": edge_ratio,
        "warm_hue_fraction_excess_vs_background": warm_excess,
        "pass": edge_ratio is None or (edge_ratio < args.fringe_edge_ratio_max and (warm_excess or 0) < args.fringe_warm_excess_max),
        "suggestion": "increase --edge-despill (try +0.2) and/or --edge-feather 1-2" if (edge_ratio and edge_ratio >= args.fringe_edge_ratio_max) or (warm_excess and warm_excess >= args.fringe_warm_excess_max) else None,
    }
    worst_halo_fraction = float(max(halo_fractions)) if halo_fractions else None
    checks["edge_brightness_halo"] = {
        "metric": worst_halo_fraction,
        "worst_frame_fraction": worst_halo_fraction,
        "mean_frame_fraction": mean_or_none(halo_fractions),
        "pass": worst_halo_fraction is None or worst_halo_fraction < args.halo_fraction_max,
        "suggestion": (
            "localized bright rim at the matte boundary (e.g. an elbow/limb edge): increase "
            "--edge-despill, or if it persists, the new background's own content directly behind "
            "that limb may be bright/light-colored there -- try a different --bg-anchor-x/y or crop"
            if worst_halo_fraction and worst_halo_fraction >= args.halo_fraction_max else None
        ),
    }
    checks["sharpness_match"] = {
        "metric": sharp_ratio,
        # Wide tolerance by default: this is a noisy proxy (a face is
        # inherently different frequency content than e.g. gym equipment),
        # meant to catch gross mismatches (near-zero detail, or oversharpen
        # halos), not to force a pixel-perfect match.
        "pass": sharp_ratio is None or (args.sharpness_ratio_min <= sharp_ratio <= args.sharpness_ratio_max),
        "suggestion": (
            "subject much softer than background: increase --bg-blur to match (prefer this over --fg-sharpen, which reads as pixelation on a face)"
            if sharp_ratio and sharp_ratio < args.sharpness_ratio_min else
            "subject much sharper/noisier than background (possible halo): reduce --fg-sharpen"
            if sharp_ratio and sharp_ratio > args.sharpness_ratio_max else None
        ),
    }
    checks["brightness"] = {
        "metric": mean_luma,
        "pass": mean_luma is None or (args.brightness_min <= mean_luma <= args.brightness_max),
        "suggestion": (
            "subject too dark: lower --relight-strength or --relight-luma-weight, or raise grade --brightness"
            if mean_luma and mean_luma < args.brightness_min else
            "subject too bright/blown out: raise --relight-luma-weight toward target or lower grade --brightness"
            if mean_luma and mean_luma > args.brightness_max else None
        ),
    }

    if args.source_video:
        scap = cv2.VideoCapture(args.source_video)
        sw, sh = int(scap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(scap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        sfps = scap.get(cv2.CAP_PROP_FPS)
        sdur = scap.get(cv2.CAP_PROP_FRAME_COUNT) / sfps if sfps else None
        scap.release()
        dur = total / fps if fps else None
        checks["duration_match"] = {
            "output_duration": dur, "source_duration": sdur,
            "pass": sdur is None or dur is None or abs(dur - sdur) < args.duration_tolerance,
            "suggestion": "output duration drifted from source; check audio mux step (--shortest / frame drop)" if sdur and dur and abs(dur - sdur) >= args.duration_tolerance else None,
        }

    report = {
        "video": args.video, "resolution": [w, h], "fps": fps, "total_frames": total,
        "samples_used": len(idxs), "checks": checks,
        "overall_pass": all(c["pass"] for c in checks.values()),
    }

    print(json.dumps(report, indent=2))
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)

    sys.exit(0 if report["overall_pass"] else 1)


if __name__ == "__main__":
    main()
