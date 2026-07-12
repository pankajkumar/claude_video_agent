#!/usr/bin/env python3
"""
Build a labeled side-by-side frame grid from multiple candidate composited
videos (e.g. different background-removal models/settings, different
background images, or "original source vs. composite" run against the same
clip), so they can be eyeballed together instead of one at a time.

Two modes:
  - Whole-frame comparison (default): each video's frame is scaled to
    --height and placed side by side.
  - Region comparison (--region and/or per-video region): crop to a region
    in that video's own pixel coordinates and zoom it, e.g. to compare matte
    edge quality or detail/pixelation at the same spot across renders, or
    against the un-composited original.

Usage:
    # whole frames
    python3 compare_renders.py \
        --video A:out_candidate_a.mp4 --video B:out_candidate_b.mp4 \
        --frame 90 --output compare.jpg

    # same region, same coordinates, across all videos
    python3 compare_renders.py \
        --video Original:source_cropped.mp4 --video Composite:out.mp4 \
        --frame 90 --region 220,280,260,200 --zoom 4 --output face_zoom_compare.jpg

    # per-video region override (e.g. original video isn't content-cropped
    # the same way the composite is, so its silhouette sits at different
    # coordinates) -- append :x,y,w,h to that --video entry
    python3 compare_renders.py \
        --video Original:source_full.mp4:876,280,260,200 \
        --video Composite:out.mp4:220,280,260,200 \
        --frame 90 --zoom 4 --output face_zoom_compare.jpg
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def grab_frame(video_path, frame_idx, time_s):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"could not open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = int(time_s * fps) if time_s is not None else (frame_idx if frame_idx is not None else total // 2)
    idx = max(0, min(idx, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {idx} from {video_path}")
    return frame


def parse_video_arg(v):
    """label:path or label:path:x,y,w,h (per-video region override)."""
    parts = v.split(":")
    if len(parts) == 2:
        label, path = parts
        return label, path, None
    if len(parts) == 3:
        label, path, region = parts
        x, y, w, h = (int(n) for n in region.split(","))
        return label, path, (x, y, w, h)
    raise SystemExit(f"--video must be label:path or label:path:x,y,w,h, got {v!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", action="append", required=True,
                    help="label:path, or label:path:x,y,w,h for a per-video region override; repeatable")
    p.add_argument("--frame", type=int, default=None)
    p.add_argument("--time", type=float, default=None, help="seconds, alternative to --frame")
    p.add_argument("--height", type=int, default=640,
                    help="whole-frame mode: row height in px, width auto-scaled (ignored in region mode)")
    p.add_argument("--region", default=None,
                    help="x,y,w,h applied to every video unless it has its own per-video region; switches to region mode")
    p.add_argument("--zoom", type=float, default=3.0, help="region mode: nearest-neighbor zoom factor")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    entries = [parse_video_arg(v) for v in args.video]
    global_region = None
    if args.region:
        x, y, w, h = (int(n) for n in args.region.split(","))
        global_region = (x, y, w, h)
    region_mode = global_region is not None or any(r for _, _, r in entries)

    bar_h = 28
    tiles = []
    for label, path, region in entries:
        frame = grab_frame(path, args.frame, args.time)
        region = region or global_region

        if region_mode:
            x, y, w, h = region
            fh, fw = frame.shape[:2]
            x, y = max(0, min(x, fw - 1)), max(0, min(y, fh - 1))
            w, h = min(w, fw - x), min(h, fh - y)
            crop = frame[y:y + h, x:x + w]
            resized = cv2.resize(crop, (int(w * args.zoom), int(h * args.zoom)), interpolation=cv2.INTER_NEAREST)
        else:
            fh, fw = frame.shape[:2]
            scale = args.height / fh
            resized = cv2.resize(frame, (int(fw * scale), args.height), interpolation=cv2.INTER_AREA)

        tile = np.full((resized.shape[0] + bar_h, resized.shape[1], 3), 255, dtype=np.uint8)
        tile[bar_h:] = resized
        cv2.putText(tile, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        tiles.append(tile)

    max_h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, max_h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
             for t in tiles]
    montage = np.hstack(tiles)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, montage)
    print(f"[done] wrote {args.output} ({len(tiles)} clip(s){', region mode' if region_mode else ''})")


if __name__ == "__main__":
    main()
