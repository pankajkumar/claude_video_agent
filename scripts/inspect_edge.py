#!/usr/bin/env python3
"""
Pull one frame out of a composited video and zoom into specific regions
(e.g. around the head/shoulder silhouette) to inspect the matte boundary for
visible seams, halos, or color fringe — the things that are easy to miss at
full-frame/full-speed but jump out once magnified.

Usage:
    python3 inspect_edge.py --video out.mp4 --frame 90 \
        --region 190,0,180,220 --region 40,150,200,300 \
        --output edge_zoom.jpg

    # no --region given -> defaults to a head-and-shoulders strip down the
    # vertical center third of the frame, which is where a standing
    # camera-facing subject's silhouette usually falls
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_region(s, frame_w, frame_h):
    x, y, w, h = (int(v) for v in s.split(","))
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = min(w, frame_w - x)
    h = min(h, frame_h - y)
    return x, y, w, h


def default_regions(frame_w, frame_h):
    # center-third vertical strip, split into head / torso / waist bands
    cx0, cx1 = int(frame_w * 0.30), int(frame_w * 0.70)
    band_h = frame_h // 3
    return [
        (cx0, 0, cx1 - cx0, band_h),
        (cx0, band_h, cx1 - cx0, band_h),
        (cx0, band_h * 2, cx1 - cx0, frame_h - band_h * 2),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--frame", type=int, default=None, help="frame index (default: middle frame)")
    p.add_argument("--time", type=float, default=None, help="alternative to --frame, seconds into the clip")
    p.add_argument("--region", action="append", default=[], help="x,y,w,h in source pixels; repeatable")
    p.add_argument("--zoom", type=float, default=3.0)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if args.time is not None:
        idx = int(args.time * fps)
    elif args.frame is not None:
        idx = args.frame
    else:
        idx = total // 2
    idx = max(0, min(idx, total - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {idx}")

    h, w = frame.shape[:2]
    regions = [parse_region(r, w, h) for r in args.region] if args.region else default_regions(w, h)

    tiles = []
    for (x, y, rw, rh) in regions:
        crop = frame[y:y + rh, x:x + rw]
        zoomed = cv2.resize(crop, (int(rw * args.zoom), int(rh * args.zoom)), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(zoomed, (0, 0), (zoomed.shape[1] - 1, zoomed.shape[0] - 1), (0, 0, 255), 2)
        label = f"({x},{y},{rw}x{rh})"
        cv2.putText(zoomed, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        tiles.append(zoomed)

    max_h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, max_h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
             for t in tiles]
    montage = np.hstack(tiles)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, montage)
    print(f"[done] frame {idx}/{total - 1} -> wrote {args.output} ({len(tiles)} region(s) at {args.zoom}x zoom)")


if __name__ == "__main__":
    main()
