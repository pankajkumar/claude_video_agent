#!/usr/bin/env python3
"""
Optional AI super-resolution pass using Real-ESRGAN (ncnn-vulkan binary).

NOT part of the default pipeline_agent.py flow — benchmarked at ~8-10s/frame
on Apple Silicon (Vulkan/MoltenVK), i.e. hours for anything beyond a few
seconds of footage. Useful for:
  - short hero clips / thumbnails where quality matters more than turnaround
  - a handful of still frames
  - re-evaluating on a machine with a strong discrete GPU, where this may be
    fast enough to use by default

Usage:
    python3 upscale_realesrgan.py --input video.mp4 --output upscaled.mp4 --scale 2
    python3 upscale_realesrgan.py --input frame.png --output frame_up.png --scale 2

Extracts frames -> runs the ncnn-vulkan binary in batch-folder mode (much
faster than invoking it per-frame, which pays process/model-load startup
cost every time) -> re-muxes with the original audio/fps if input was video.

IMPORTANT: realesrgan-x4plus / realesrgan-x4plus-anime are FIXED 4x networks
(there's no separately-trained 2x/3x weight file for them, unlike the
realesr-animevideov3 family which ships -x2/-x3/-x4 variants). Asking the
ncnn-vulkan binary for `-s 2` or `-s 3` with an x4plus model forces it to
do an internal 4x-then-downsample step that is broken in this binary build
-- it produces a corrupted, mirrored/tiled "kaleidoscope" mosaic instead of
a clean image (confirmed: identical artifact in both single-file and batch-
folder mode, and with an explicit `-t` tile-size override, so it isn't a
tiling-stitch bug -- only `-s 4` with this model is clean). This script
always runs the binary at the model's true native scale and does any
further downsampling itself afterward with a normal high-quality resize.
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

TOOLS_DIR = Path(__file__).parent.parent / "tools" / "realesrgan-ncnn"
BINARY = TOOLS_DIR / "realesrgan-ncnn-vulkan"

# Models with exactly one trained scale (no per-scale weight files) --
# always invoke the binary at this scale, then downsample ourselves if the
# user asked for less. realesr-animevideov3 is NOT here: it ships separate
# -x2/-x3/-x4 weight files, so asking the binary for a different -s value
# correctly loads the matching, properly-trained weights instead.
FIXED_SCALE_MODELS = {"realesrgan-x4plus": 4, "realesrgan-x4plus-anime": 4}


def native_scale_for(model, requested_scale):
    return FIXED_SCALE_MODELS.get(model, requested_scale)


def downsample_to_scale(image_path, orig_w, orig_h, target_scale):
    img = cv2.imread(str(image_path))
    target_w, target_h = round(orig_w * target_scale), round(orig_h * target_scale)
    if (img.shape[1], img.shape[0]) != (target_w, target_h):
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(image_path), img)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--scale", type=int, default=2, choices=[2, 3, 4])
    p.add_argument("--model", default="realesrgan-x4plus",
                   help="model name in tools/realesrgan-ncnn/models/ (x4plus = photo-realistic content, x4plus-anime = illustrations)")
    args = p.parse_args()

    if not BINARY.exists():
        sys.exit(f"realesrgan-ncnn-vulkan binary not found at {BINARY}")

    native_scale = native_scale_for(args.model, args.scale)
    if native_scale != args.scale:
        print(f"[info] {args.model} is a fixed {native_scale}x network -- running at {native_scale}x "
              f"then downsampling to the requested {args.scale}x (see module docstring for why)")

    in_path = Path(args.input)
    is_video = in_path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}

    if not is_video:
        orig = cv2.imread(str(in_path))
        if orig is None:
            sys.exit(f"could not read image: {in_path}")
        orig_h, orig_w = orig.shape[:2]
        cmd = [str(BINARY), "-i", str(in_path.resolve()), "-o", str(Path(args.output).resolve()),
               "-m", str(TOOLS_DIR / "models"), "-n", args.model, "-s", str(native_scale)]
        subprocess.run(cmd, check=True)
        downsample_to_scale(args.output, orig_w, orig_h, args.scale)
        print(f"[done] wrote {args.output}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        frames_in, frames_out = tmp / "in", tmp / "out"
        frames_in.mkdir()
        frames_out.mkdir()

        fps = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=r_frame_rate", "-of", "csv=p=0", str(in_path)],
            capture_output=True, text=True, check=True
        ).stdout.strip()

        print("[info] extracting frames...")
        subprocess.run(["ffmpeg", "-y", "-i", str(in_path), str(frames_in / "f%06d.png")],
                        check=True, capture_output=True)
        n_frames = len(list(frames_in.glob("*.png")))
        print(f"[info] {n_frames} frames extracted; this will take roughly "
              f"{n_frames * 8 / 60:.1f}-{n_frames * 10 / 60:.1f} minutes at ~8-10s/frame")

        first_frame = sorted(frames_in.glob("*.png"))[0]
        first_frame_img = cv2.imread(str(first_frame))
        frame_h, frame_w = first_frame_img.shape[:2]

        print("[info] running realesrgan-ncnn-vulkan (batch mode)...")
        subprocess.run([str(BINARY), "-i", str(frames_in), "-o", str(frames_out),
                         "-m", str(TOOLS_DIR / "models"),
                         "-n", args.model, "-s", str(native_scale), "-f", "png"], check=True)

        if native_scale != args.scale:
            print(f"[info] downsampling frames from {native_scale}x to requested {args.scale}x...")
            for frame in frames_out.glob("*.png"):
                downsample_to_scale(frame, frame_w, frame_h, args.scale)

        print("[info] re-encoding + muxing original audio...")
        tmp_video = tmp / "video_noaudio.mp4"
        subprocess.run(["ffmpeg", "-y", "-framerate", fps, "-i", str(frames_out / "f%06d.png"),
                         "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
                         str(tmp_video)], check=True, capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-i", str(tmp_video), "-i", str(in_path),
                         "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac",
                         "-shortest", args.output], check=True, capture_output=True)

    print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()
