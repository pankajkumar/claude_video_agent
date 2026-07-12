#!/usr/bin/env python3
"""
Audio-only processing on a video file: the video stream is always stream-
copied untouched (`-c:v copy`), so this is safe to run on a final, already-
graded video without re-encoding a single video frame.

Operations (all optional, chained in one ffmpeg pass when applied to the
existing track):
  --normalize             loudness-normalize to --target-lufs (EBU R128 via
                           ffmpeg's loudnorm filter)
  --denoise F              afftdn noise reduction strength, 0=off
  --volume-db F            gain in dB, 0=off
  --replace-audio PATH     swap in a completely different audio file instead
                           of filtering the original track (e.g. a cleaned
                           voiceover) -- mutually exclusive with the filters
                           above, since there's no "original track" to filter

Usage:
    python3 process_audio.py --input video.mp4 --output video_audio.mp4 --normalize
    python3 process_audio.py --input video.mp4 --output out.mp4 --denoise 12 --volume-db 2
    python3 process_audio.py --input video.mp4 --output out.mp4 --replace-audio voiceover.wav
"""
import argparse
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--normalize", action="store_true", help="EBU R128 loudness normalize")
    p.add_argument("--target-lufs", type=float, default=-16.0, help="loudnorm target integrated loudness")
    p.add_argument("--loudnorm-tp", type=float, default=-1.5, help="loudnorm true-peak ceiling (dBTP)")
    p.add_argument("--loudnorm-lra", type=float, default=11.0, help="loudnorm target loudness range")
    p.add_argument("--denoise", type=float, default=0.0, help="afftdn noise reduction in dB, 0=off, ~10-25 typical")
    p.add_argument("--volume-db", type=float, default=0.0, help="gain in dB, 0=off")
    p.add_argument("--replace-audio", default=None,
                    help="path to a replacement audio file; replaces the track entirely instead of filtering it")
    args = p.parse_args()

    if args.replace_audio:
        if args.normalize or args.denoise or args.volume_db:
            print("[warn] --replace-audio ignores --normalize/--denoise/--volume-db "
                  "(no original track left to filter); apply those to the replacement file directly if needed",
                  file=sys.stderr)
        cmd = [
            "ffmpeg", "-y",
            "-i", args.input, "-i", args.replace_audio,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            args.output,
        ]
    else:
        filters = []
        if args.denoise > 0:
            filters.append(f"afftdn=nr={args.denoise}")
        if args.volume_db != 0:
            filters.append(f"volume={args.volume_db}dB")
        if args.normalize:
            filters.append(f"loudnorm=I={args.target_lufs}:TP={args.loudnorm_tp}:LRA={args.loudnorm_lra}")
        if not filters:
            sys.exit("nothing to do: pass --normalize, --denoise, --volume-db, or --replace-audio")
        cmd = [
            "ffmpeg", "-y", "-i", args.input,
            "-c:v", "copy",
            "-af", ",".join(filters),
            "-c:a", "aac", "-b:a", "192k",
            args.output,
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"ffmpeg failed:\n{result.stderr[-2000:]}")
    print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()
