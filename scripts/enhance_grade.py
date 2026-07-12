#!/usr/bin/env python3
"""
Post-process a composited video: denoise/detail-enhance + cinematic color grade.

Both stages are applied as a single chained ffmpeg filtergraph in one encode
pass (each can still be toggled/tuned independently via flags) rather than as
two separate ffmpeg runs — every extra encode generation compounds lossy
compression artifacts, which is exactly what produces a visible "stair-step"
border right at the composited subject's edge:
  1. enhance: hqdn3d (denoise) + unsharp (detail) — cleans up matting/compression
     softness without touching color.
  2. grade: filmic contrast curve, saturation, warm/cool color balance, vignette,
     fine grain — gives the "shot on camera" cinematic look.

Usage:
    python3 enhance_grade.py --input comp.mp4 --output final.mp4
    python3 enhance_grade.py --input comp.mp4 --output final.mp4 --no-grade
    python3 enhance_grade.py --input comp.mp4 --output final.mp4 \
        --denoise 2.0 --sharpen 0.8 --brightness 0.06 --vignette 0
"""
import argparse
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)

    p.add_argument("--enhance", action="store_true", default=True)
    p.add_argument("--no-enhance", dest="enhance", action="store_false")
    p.add_argument("--denoise", type=float, default=1.5, help="hqdn3d luma spatial strength")
    p.add_argument("--sharpen", type=float, default=0.6, help="unsharp amount")

    p.add_argument("--grade", action="store_true", default=True)
    p.add_argument("--no-grade", dest="grade", action="store_false")
    p.add_argument("--contrast", type=float, default=1.06)
    p.add_argument("--saturation", type=float, default=1.10)
    p.add_argument("--brightness", type=float, default=0.04, help="-1..1, eq= brightness offset")
    p.add_argument("--warmth", type=float, default=0.03, help="shifts red up / blue down in shadows+mids; negative = cooler")
    p.add_argument("--vignette", type=float, default=1.0, help="0=off, 1=normal strength (PI/7), 2=stronger")
    p.add_argument("--grain", type=float, default=3.0, help="0=off, noise=alls=<grain>")
    p.add_argument("--crf", type=int, default=14)
    args = p.parse_args()

    if not args.enhance and not args.grade:
        sys.exit("Nothing to do: both --no-enhance and --no-grade set")

    filters = []
    if args.enhance:
        filters.append(f"hqdn3d={args.denoise}:{args.denoise}:3:3")
        filters.append(f"unsharp=5:5:{args.sharpen}:5:5:0.0")

    if args.grade:
        filters.append("curves=master='0/0.02 0.25/0.27 0.5/0.54 0.75/0.80 1/1'")
        filters.append(f"eq=contrast={args.contrast}:saturation={args.saturation}:brightness={args.brightness}")
        if args.warmth != 0:
            w = args.warmth
            filters.append(f"colorbalance=rs={w:.3f}:bs={-w*1.3:.3f}:rm={w*0.6:.3f}:bm={-w*0.6:.3f}:rh={-w*0.3:.3f}:bh={w*0.9:.3f}")
        if args.vignette > 0:
            filters.append(f"vignette=PI/{max(2.0, 7/args.vignette):.2f}:mode=forward")
        if args.grain > 0:
            filters.append(f"noise=alls={args.grain}:allf=t+u")

    run_ffmpeg(args.input, args.output, ",".join(filters), args.crf)
    print(f"[done] wrote {args.output}")


def run_ffmpeg(src, dst, vf, crf):
    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-vf", vf,
        # Force yuv420p explicitly: without it, libx264 auto-negotiates the
        # pixel format from whatever the filtergraph happens to output --
        # colorbalance (used for --warmth) bumps chroma precision, which
        # without an explicit override here silently produced yuv444p /
        # "High 4:4:4 Predictive" profile. That's essentially undecodable on
        # phone hardware decoders (iOS and Android both expect yuv420p /
        # High-or-lower profile) even though it plays fine on a desktop with
        # ffmpeg's own software decoder -- the file looks fine until someone
        # actually tries to play it on a phone. +faststart moves the moov
        # atom to the front of the file so mobile players can start playback
        # before the whole file downloads, instead of needing it fully local
        # first.
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "slow", "-crf", str(crf),
        "-movflags", "+faststart",
        "-c:a", "copy",
        dst,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


if __name__ == "__main__":
    main()
