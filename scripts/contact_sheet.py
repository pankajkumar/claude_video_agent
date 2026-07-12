#!/usr/bin/env python3
"""
Build a labeled thumbnail grid from a folder of candidate background images
(or any folder of images), for quickly eyeballing options before picking one
to composite against. See README's "Background images" section for what to
look for: real photo (not CG render), plain area where the subject's head
will land, no text/logos/people, lighting not wildly different from source.

Usage:
    python3 contact_sheet.py --images-dir "/path/to/gym images" \
        --output contact_sheet.jpg
    python3 contact_sheet.py --images "a.jpg" "b.jpg" --output sheet.jpg --cols 3
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def label(img, text, cell_w):
    bar_h = 22
    canvas = np.full((img.shape[0] + bar_h, cell_w, 3), 255, dtype=np.uint8)
    canvas[bar_h:, : img.shape[1]] = img
    cv2.putText(canvas, text[: cell_w // 7], (3, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", help="folder of images to include (non-recursive)")
    p.add_argument("--images", nargs="*", help="explicit list of image paths instead of --images-dir")
    p.add_argument("--output", required=True)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--cell-size", type=int, default=320, help="thumbnail box side in px")
    args = p.parse_args()

    if args.images:
        paths = [Path(f) for f in args.images]
    elif args.images_dir:
        paths = sorted(f for f in Path(args.images_dir).iterdir() if f.suffix.lower() in IMAGE_EXTS)
    else:
        raise SystemExit("pass --images-dir or --images")
    if not paths:
        raise SystemExit("no images found")

    cell = args.cell_size
    bar_h = 22
    cols = args.cols
    rows = (len(paths) + cols - 1) // cols
    sheet = np.full((rows * (cell + bar_h), cols * cell, 3), 255, dtype=np.uint8)

    for i, path in enumerate(paths):
        img = cv2.imread(str(path))
        if img is None:
            print(f"[warn] could not read {path}, skipping")
            continue
        h, w = img.shape[:2]
        scale = min(cell / w, cell / h)
        thumb = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        tile = np.full((cell, cell, 3), 230, dtype=np.uint8)
        y0 = (cell - thumb.shape[0]) // 2
        x0 = (cell - thumb.shape[1]) // 2
        tile[y0:y0 + thumb.shape[0], x0:x0 + thumb.shape[1]] = thumb
        tile = label(tile, path.name, cell)

        r, c = divmod(i, cols)
        sheet[r * (cell + bar_h):(r + 1) * (cell + bar_h), c * cell:(c + 1) * cell] = tile

    cv2.imwrite(args.output, sheet)
    print(f"[done] wrote {args.output} ({len(paths)} images, {cols}x{rows} grid)")


if __name__ == "__main__":
    main()
