#!/usr/bin/env python3
"""
Detect the real content box of a video that has black pillarbox/letterbox
bars baked into the frame (common with vertical phone footage embedded in a
landscape canvas), and print it as a --content-crop W:H:X:Y string ready to
paste into replace_background.py / pipeline_agent.py.

Wraps ffmpeg's cropdetect filter and takes the most frequently reported
crop= line across the whole video (a single frame's detection can be noisy).

Usage:
    python3 detect_content_crop.py --input video.mp4
    python3 detect_content_crop.py --input video.mp4 --limit-seconds 5
"""
import argparse
import re
import subprocess
import sys
from collections import Counter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--limit-seconds", type=float, default=None,
                    help="only scan the first N seconds (faster on long videos)")
    p.add_argument("--threshold", type=int, default=24, help="cropdetect black-level threshold")
    args = p.parse_args()

    cmd = ["ffmpeg"]
    if args.limit_seconds:
        cmd += ["-t", str(args.limit_seconds)]
    cmd += ["-i", args.input, "-vf", f"cropdetect={args.threshold}:2:0", "-f", "null", "-"]

    result = subprocess.run(cmd, capture_output=True, text=True)
    crops = re.findall(r"crop=(\d+:\d+:\d+:\d+)", result.stderr)
    if not crops:
        sys.exit("No crop detected (ffmpeg cropdetect found no black bars, or input/ffmpeg error)")

    most_common, count = Counter(crops).most_common(1)[0]
    print(f"--content-crop {most_common}")
    print(f"# (seen in {count}/{len(crops)} sampled frames)", file=sys.stderr)


if __name__ == "__main__":
    main()
