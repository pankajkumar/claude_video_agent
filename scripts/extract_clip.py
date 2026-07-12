#!/usr/bin/env python3
"""
Cut a short test clip out of a longer video, always re-encoding (never
stream-copying). Stream-copying a cut at a non-keyframe boundary can scramble
frame order on decode (B-frame reordering) — this looks like a pipeline bug
downstream but isn't, so this tool always re-encodes to avoid it.

Usage:
    python3 extract_clip.py --input video.mp4 --start 5 --duration 3 \
        --output clip3s.mp4
"""
import argparse
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--start", type=float, default=0.0, help="seconds")
    p.add_argument("--duration", type=float, required=True, help="seconds")
    p.add_argument("--crf", type=int, default=18)
    args = p.parse_args()

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(args.start),
        "-i", args.input,
        "-t", str(args.duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", str(args.crf),
        "-c:a", "aac",
        args.output,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"ffmpeg failed:\n{result.stderr[-2000:]}")
    print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()
