#!/usr/bin/env python3
"""
Check whether a background image's native resolution is high enough for a
given output canvas, and if not, AI-upscale it (Real-ESRGAN) before it ever
reaches replace_background.py.

Why this matters: replace_background.py's cover_resize scales a background
to *fill* the output canvas (CSS background-size: cover semantics) --
`scale = max(canvas_w / img_w, canvas_h / img_h)`. If a background's native
resolution is smaller than what that scale requires, cover_resize UPSCALES
it, which faithfully magnifies whatever compression artifacts the source
JPEG/WebP already had (blocky edges, soft detail) -- this is what "the
background looks pixelated even in the sharp version, but the original
photo looks fine" actually is: the original looks fine at its own (smaller)
display size, but the portrait-video canvas here needs more pixels than the
source has, so something has to invent detail. Plain Lanczos upscaling
can't invent detail; Real-ESRGAN is trained to.

This only touches a handful of *still images* (the background candidates),
not video frames -- Real-ESRGAN is too slow for the latter (see README) but
takes seconds for the former.

Usage:
    python3 prepare_background.py --image "bg.jpg" --canvas-width 608 --canvas-height 1080
    # or derive canvas size from a --content-crop string directly:
    python3 prepare_background.py --image "bg.jpg" --content-crop 608:1080:656:0

Prints the path to use as --background: the original path unchanged if it's
already high enough resolution, or the path to a cached upscaled copy
(written next to the original, suffixed _upscaled_<N>x.png) otherwise.
"""
import argparse
import math
import subprocess
import sys
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).parent


def needed_upscale_factor(img_w, img_h, canvas_w, canvas_h):
    """The cover_resize scale factor a background of this native size would
    need to undergo to fill this canvas. <=1.0 means no upscale needed."""
    return max(canvas_w / img_w, canvas_h / img_h)


def prepare_background(image_path, canvas_w, canvas_h, *, margin=1.0, min_scale_step=2,
                         max_scale_step=4, model="realesrgan-x4plus", force=False,
                         bg_crop_top=0.0, bg_crop_bottom=0.0):
    """Returns the path to use as --background. If the image is already
    high-res enough (factor <= margin), returns image_path unchanged.
    Otherwise upscales by the smallest of {2,3,4}x that clears the
    requirement (capped at max_scale_step) and caches the result next to the
    original. margin lets you require some headroom above the bare minimum
    (e.g. margin=1.5 means "at least 1.5x more native resolution than the
    bare minimum needed").

    bg_crop_top/bg_crop_bottom MUST match whatever replace_background.py's
    own --bg-crop-top/--bg-crop-bottom will be for this background -- those
    discard rows from the source BEFORE cover_resize runs, so the effective
    source for the upscale-need calculation is smaller than the raw file,
    sometimes significantly (e.g. a 0.3 top-crop on an already-short image
    can turn a fine 1.08x cover-fit into a much worse 1.55x one). Passing the
    full uncropped image's size here when a crop is actually in effect
    silently understates how much upscale is really needed."""
    img = cv2.imread(str(image_path))
    if img is None:
        sys.exit(f"could not read image: {image_path}")
    h, w = img.shape[:2]
    cropped_h = h - int(h * bg_crop_top) - int(h * bg_crop_bottom)
    factor = needed_upscale_factor(w, cropped_h, canvas_w, canvas_h)

    crop_note = f" ({w}x{cropped_h} after bg-crop-top/bottom)" if (bg_crop_top > 0 or bg_crop_bottom > 0) else ""
    if factor <= margin and not force:
        print(f"[info] {image_path}: native {w}x{h}{crop_note} already sufficient for "
              f"{canvas_w}x{canvas_h} canvas (would need {factor:.2f}x upscale, margin {margin}x) -- using as-is")
        return str(image_path)

    # realesrgan-ncnn-vulkan only supports scale in {2,3,4}; pick the
    # smallest of those that clears the required factor, capped at 4x.
    available = [s for s in (2, 3, 4) if min_scale_step <= s <= max_scale_step]
    scale_step = next((s for s in available if s >= factor), max(available))

    cache_path = Path(image_path).with_name(f"{Path(image_path).stem}_upscaled_{scale_step}x.png")
    if cache_path.exists() and not force:
        print(f"[info] {image_path}: needs {factor:.2f}x upscale -- using cached {cache_path}")
        return str(cache_path)

    print(f"[info] {image_path}: native {w}x{h}{crop_note} needs {factor:.2f}x upscale for "
          f"{canvas_w}x{canvas_h} canvas -- running Real-ESRGAN {scale_step}x...")
    cmd = [sys.executable, str(SCRIPT_DIR / "upscale_realesrgan.py"),
           "--input", str(image_path), "--output", str(cache_path),
           "--scale", str(scale_step), "--model", model]
    subprocess.run(cmd, check=True)
    print(f"[done] wrote {cache_path}")
    return str(cache_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--canvas-width", type=int, default=None)
    p.add_argument("--canvas-height", type=int, default=None)
    p.add_argument("--content-crop", default=None, help="w:h:x:y -- alternative to --canvas-width/height")
    p.add_argument("--margin", type=float, default=1.0,
                    help="require native resolution to clear the bare-minimum cover-fit scale by this factor")
    p.add_argument("--bg-crop-top", type=float, default=0.0,
                    help="MUST match replace_background.py's --bg-crop-top for this background, else the upscale-need "
                         "calculation is computed against the wrong (too-large) effective source size")
    p.add_argument("--bg-crop-bottom", type=float, default=0.0,
                    help="MUST match replace_background.py's --bg-crop-bottom for this background")
    p.add_argument("--model", default="realesrgan-x4plus")
    p.add_argument("--force", action="store_true", help="re-run even if already sufficient or already cached")
    args = p.parse_args()

    if args.content_crop:
        cw, ch, _, _ = (int(v) for v in args.content_crop.split(":"))
    elif args.canvas_width and args.canvas_height:
        cw, ch = args.canvas_width, args.canvas_height
    else:
        sys.exit("pass --content-crop, or both --canvas-width and --canvas-height")

    result = prepare_background(args.image, cw, ch, margin=args.margin, model=args.model, force=args.force,
                                 bg_crop_top=args.bg_crop_top, bg_crop_bottom=args.bg_crop_bottom)
    print(result)


if __name__ == "__main__":
    main()
