#!/usr/bin/env python3
"""
Run a saved workflow (workflows/*.md, see workflow_template.md) against a
video. A workflow is an ordered list of STAGES -- each stage is one of the
pipeline's building blocks -- so the same runner handles every shape of job
without special-casing any of them:
    - background replacement only            : [bg_replace]
    - background replacement, blurred bg      : [bg_replace] with bg_blur set
    - background removal only (flat color)    : [bg_remove]
    - background removal + grade, no bg image : [bg_remove, enhance]
    - enhance/grade only, no background work  : [enhance]
    - audio only, no video work at all        : [audio]
    - any combination, in any order           : [bg_replace, audio], etc.

Stage output chains into the next stage's input automatically. A workflow
with zero video-touching stages (just [audio]) never invokes the matting
model at all -- each stage is a thin dispatch to one existing script
(pipeline_agent.py / replace_background.py / enhance_grade.py /
process_audio.py / upscale_realesrgan.py), so adding a new stage type later
is a small addition here, not a redesign.

IMPORTANT: a `bg_replace` stage's params include enhance_grade.py's params
too (denoise/sharpen/grain/contrast/saturation/brightness/warmth/vignette),
not a separate trailing `enhance` stage -- pipeline_agent.py grades every
auto-tune iteration itself (the verifier checks the GRADED output), so a
separate `enhance` stage after `bg_replace` would grade the video twice. Use
a standalone `enhance` stage only when there's no `bg_replace` stage (e.g.
after `bg_remove`, or as the only stage in an enhance-only workflow).

Usage:
    python3 run_workflow.py --workflow workflows/gym_dark_moody.md \
        --input /path/to/full_video.mp4 --output /path/to/final.mp4

    # override one stage's one param without editing the .md
    python3 run_workflow.py --workflow workflows/gym_dark_moody.md \
        --input video.mp4 --output final.mp4 \
        --stage-param bg_replace.bg_blur=4.0

    # if the workflow has more than one stage of the same type, target by
    # position (0-based) instead of type:
        --stage-param 0.bg_blur=4.0
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).parent


def load_workflow(path):
    text = Path(path).read_text()
    if not text.startswith("---"):
        sys.exit(f"{path}: expected YAML frontmatter (file must start with '---')")
    _, frontmatter, _body = text.split("---", 2)
    workflow = yaml.safe_load(frontmatter)
    if "stages" not in workflow or not isinstance(workflow["stages"], list):
        sys.exit(f"{path}: frontmatter must have a 'stages' list")
    return workflow


def apply_stage_overrides(stages, overrides):
    """overrides: list of 'target.key=value' where target is a stage type
    name (applies to the first stage of that type) or a 0-based index."""
    for ov in overrides:
        target_key, value = ov.split("=", 1)
        target, key = target_key.rsplit(".", 1)
        try:
            value = float(value) if "." in value else int(value)
        except ValueError:
            pass

        if target.isdigit():
            idx = int(target)
            if idx >= len(stages):
                sys.exit(f"--stage-param target index {idx} out of range ({len(stages)} stages)")
            stages[idx].setdefault("params", {})[key] = value
        else:
            matches = [s for s in stages if s.get("type") == target]
            if not matches:
                sys.exit(f"--stage-param target type {target!r} matches no stage")
            matches[0].setdefault("params", {})[key] = value
    return stages


def video_dimensions(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"could not open {path} to read dimensions")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


# BGR tuples (OpenCV order), not RGB.
COLOR_NAMES_BGR = {
    "green": (0, 255, 0), "black": (0, 0, 0), "white": (255, 255, 255),
    "gray": (128, 128, 128), "blue": (255, 0, 0),
}


def resolve_color_bgr(name_or_hex):
    if name_or_hex in COLOR_NAMES_BGR:
        return COLOR_NAMES_BGR[name_or_hex]
    hexs = name_or_hex.lstrip("#")
    r, g, b = int(hexs[0:2], 16), int(hexs[2:4], 16), int(hexs[4:6], 16)
    return (b, g, r)


def to_cli(params):
    args = []
    for k, v in params.items():
        args += ["--" + k.replace("_", "-"), str(v)]
    return args


def resolve_background_ref(background, context, idx):
    """A stage's 'background' param can be a literal path, or '$name' to use
    whatever an earlier upscale_background stage (with matching 'name',
    default 'background') stored in the shared context -- this is what
    makes upscaling optional/composable: a workflow either includes an
    upscale_background stage feeding '$name' into a later bg_replace/
    bg_remove stage, or just uses a literal path and skips it entirely."""
    if isinstance(background, str) and background.startswith("$"):
        key = background[1:]
        if key not in context:
            sys.exit(f"stage {idx}: background references '${key}' but no earlier stage stored that name "
                      f"(add an upscale_background stage with name: {key}, or use a literal path)")
        return context[key]
    return background


def run_upscale_background(current_input, params, work_dir, idx, context):
    """Resolve (and AI-upscale via Real-ESRGAN if needed) a background image
    for a LATER stage to consume -- does not touch the video at all, so
    `current_input` passes through unchanged. Stores its result in `context`
    under `params['name']` (default 'background'); a later bg_replace/
    bg_remove stage references it via `background: "$<name>"`.
    See prepare_background.py for the actual resolution logic and why this
    is needed (README lesson #11) -- and its own docstring / the fixed
    upscale_realesrgan.py for why naive `-s 2/3` on a fixed-4x model used to
    silently corrupt the image (now fixed at the root, but this stage stays
    explicit/opt-in regardless, since it needs the Real-ESRGAN binary and
    takes a few extra seconds)."""
    params = dict(params)
    image = params.pop("image", None)
    if not image:
        sys.exit(f"stage {idx} (upscale_background): missing required 'image' param")
    name = params.pop("name", "background")
    content_crop = params.pop("content_crop", None)
    margin = params.pop("margin", 1.0)
    model = params.pop("model", "realesrgan-x4plus")
    force = params.pop("force", False)
    # Must match the LATER bg_replace/bg_remove stage's bg_crop_top/bottom
    # for this same background, else the upscale-need check is computed
    # against the wrong (too-large) effective source size -- see
    # prepare_background.py's docstring.
    bg_crop_top = params.pop("bg_crop_top", 0.0)
    bg_crop_bottom = params.pop("bg_crop_bottom", 0.0)

    if content_crop:
        canvas_w, canvas_h = (int(v) for v in content_crop.split(":")[:2])
    else:
        canvas_w, canvas_h = video_dimensions(current_input)

    sys.path.insert(0, str(SCRIPT_DIR))
    from prepare_background import prepare_background  # noqa: E402
    resolved = prepare_background(image, canvas_w, canvas_h, margin=margin, model=model, force=force,
                                   bg_crop_top=bg_crop_top, bg_crop_bottom=bg_crop_bottom)
    context[name] = resolved
    print(f"[workflow] stage {idx} (upscale_background): ${name} -> {resolved}")
    return current_input


def run_bg_replace(current_input, params, work_dir, idx, context):
    """Full background-replacement stage: estimate (unless params already
    pin the scene-dependent values and estimate=false) -> render -> verify ->
    auto-tune -> retry. Delegates entirely to pipeline_agent.py."""
    params = dict(params)
    background = resolve_background_ref(params.pop("background", None), context, idx)
    if not background:
        sys.exit(f"stage {idx} (bg_replace): missing required 'background' param (path to a background image)")
    content_crop = params.pop("content_crop", None)
    max_iterations = params.pop("max_iterations", 3)
    estimate = params.pop("estimate", None)  # None = auto-decide
    verify_overrides = params.pop("verify", {}) or {}
    tune_overrides = params.pop("tune", {}) or {}
    estimate_overrides = params.pop("estimate_overrides", {}) or {}

    scene_keys = {"bg_blur", "fg_sharpen", "relight_strength"}
    if estimate is None:
        estimate = not scene_keys.issubset(params.keys())

    out_path = work_dir / f"stage{idx}_bg_replace.mp4"
    cmd = [sys.executable, str(SCRIPT_DIR / "pipeline_agent.py"),
           "--input", str(current_input), "--background", background,
           "--output", str(out_path), "--max-iterations", str(max_iterations)]
    if content_crop:
        cmd += ["--content-crop", content_crop]
    if not estimate:
        cmd += ["--no-estimate"]
    for k, v in params.items():
        cmd += ["--param", f"{k}={v}"]
    for k, v in tune_overrides.items():
        cmd += ["--tune-param", f"{k}={v}"]
    for k, v in verify_overrides.items():
        cmd += ["--verify-param", f"{k}={v}"]
    for k, v in estimate_overrides.items():
        cmd += ["--estimate-param", f"{k}={v}"]
    print(f"[workflow] stage {idx} (bg_replace): {' '.join(cmd[1:])}")
    subprocess.run(cmd, check=True)
    return out_path


def run_bg_remove(current_input, params, work_dir, idx, context):
    """Background-removal-only stage: matte + composite onto a flat color,
    one direct render (no estimate/verify/auto-tune loop -- there's no
    'natural-looking scene' to match against a solid color)."""
    params = dict(params)
    color = params.pop("color", "green")
    content_crop = params.pop("content_crop", None)

    if content_crop:
        w, h = (int(v) for v in content_crop.split(":")[:2])
    else:
        w, h = video_dimensions(current_input)
    solid = np.full((h, w, 3), resolve_color_bgr(color), dtype=np.uint8)
    solid_path = work_dir / f"stage{idx}_solid_{color.lstrip('#')}.png"
    cv2.imwrite(str(solid_path), solid)

    # relight makes no sense against a flat color (nothing to color-match to)
    params.setdefault("no_relight", True)
    no_relight = params.pop("no_relight")

    out_path = work_dir / f"stage{idx}_bg_remove.mp4"
    diag_path = work_dir / f"stage{idx}_bg_remove_diagnostics.json"
    cmd = [sys.executable, str(SCRIPT_DIR / "replace_background.py"),
           "--input", str(current_input), "--background", str(solid_path),
           "--output", str(out_path), "--diagnostics-out", str(diag_path)]
    if content_crop:
        cmd += ["--content-crop", content_crop]
    if no_relight:
        cmd += ["--no-relight"]
    cmd += to_cli(params)
    print(f"[workflow] stage {idx} (bg_remove): {' '.join(cmd[1:])}")
    subprocess.run(cmd, check=True)
    return out_path


def run_enhance(current_input, params, work_dir, idx, context):
    out_path = work_dir / f"stage{idx}_enhance.mp4"
    cmd = [sys.executable, str(SCRIPT_DIR / "enhance_grade.py"),
           "--input", str(current_input), "--output", str(out_path)]
    cmd += to_cli(params)
    print(f"[workflow] stage {idx} (enhance): {' '.join(cmd[1:])}")
    subprocess.run(cmd, check=True)
    return out_path


def run_audio(current_input, params, work_dir, idx, context):
    out_path = work_dir / f"stage{idx}_audio.mp4"
    cmd = [sys.executable, str(SCRIPT_DIR / "process_audio.py"),
           "--input", str(current_input), "--output", str(out_path)]
    bool_flags = {"normalize"}
    rest = {}
    for k, v in params.items():
        if k in bool_flags:
            if v:
                cmd += ["--" + k.replace("_", "-")]
        else:
            rest[k] = v
    cmd += to_cli(rest)
    print(f"[workflow] stage {idx} (audio): {' '.join(cmd[1:])}")
    subprocess.run(cmd, check=True)
    return out_path


def run_upscale(current_input, params, work_dir, idx, context):
    out_path = work_dir / f"stage{idx}_upscale.mp4"
    cmd = [sys.executable, str(SCRIPT_DIR / "upscale_realesrgan.py"),
           "--input", str(current_input), "--output", str(out_path)]
    cmd += to_cli(params)
    print(f"[workflow] stage {idx} (upscale): {' '.join(cmd[1:])}")
    subprocess.run(cmd, check=True)
    return out_path


def run_crop_video(current_input, params, work_dir, idx, context):
    """Crop and re-encode the input video using ffmpeg's crop filter.

    Required param: content_crop "W:H:X:Y" — the same format as
    replace_background.py's --content-crop.  This lets you extract just
    the subject area BEFORE a subsequent upscale stage so Real-ESRGAN
    processes only the content pixels, not the full frame with black bars.

    Example workflow order:
      crop_video  (1920x1080 -> 608x1080, content area only)
      upscale     (608x1080 -> 1216x2160, AI upscale)
      bg_replace  (runs at 1216x2160, no --content-crop needed)
    """
    content_crop = params.get("content_crop")
    if not content_crop:
        sys.exit(f"stage {idx} (crop_video): missing required 'content_crop' param")
    cw, ch, cx, cy = (int(v) for v in content_crop.split(":"))
    out_path = work_dir / f"stage{idx}_crop_video.mp4"
    cmd = [
        "ffmpeg", "-y", "-i", str(current_input),
        "-vf", f"crop={cw}:{ch}:{cx}:{cy}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "10",
        "-movflags", "+faststart", "-c:a", "copy",
        str(out_path),
    ]
    print(f"[workflow] stage {idx} (crop_video): crop {cw}x{ch} at ({cx},{cy})")
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


STAGE_RUNNERS = {
    "bg_replace": run_bg_replace,
    "bg_remove": run_bg_remove,
    "enhance": run_enhance,
    "audio": run_audio,
    "upscale": run_upscale,
    "crop_video": run_crop_video,
    "upscale_background": run_upscale_background,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workflow", required=True, help="path to a workflows/*.md file")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stage-param", action="append", default=[],
                    help="override one stage's param: <type-or-index>.<key>=<value>, repeatable, "
                         "e.g. --stage-param bg_replace.bg_blur=4.0 or --stage-param 0.bg_blur=4.0")
    p.add_argument("--work-dir", default=None, help="default: <output>.workflow_work/")
    args = p.parse_args()

    workflow = load_workflow(args.workflow)
    stages = apply_stage_overrides(workflow["stages"], args.stage_param)
    if not stages:
        sys.exit("workflow has no stages -- nothing to do")

    output = Path(args.output)
    work_dir = Path(args.work_dir) if args.work_dir else output.parent / f"{output.stem}.workflow_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"[workflow] running '{workflow.get('name', Path(args.workflow).stem)}' "
          f"({len(stages)} stage(s)) on {args.input}")

    current = Path(args.input)
    context = {}  # shared across stages, e.g. upscale_background's resolved image paths

    # Load existing manifest to restore context for skipped stages (e.g. upscale_background
    # stores a resolved image path in context that later stages reference via "$name").
    manifest_path = work_dir / "manifest.json"
    existing_manifest = {}
    if manifest_path.exists():
        try:
            existing_manifest = json.loads(manifest_path.read_text())
        except Exception:
            pass

    manifest = {"workflow": str(args.workflow), "name": workflow.get("name"), "stages_run": []}
    for idx, stage in enumerate(stages):
        stage_type = stage.get("type")
        runner = STAGE_RUNNERS.get(stage_type)
        if runner is None:
            sys.exit(f"stage {idx}: unknown type {stage_type!r}, expected one of {list(STAGE_RUNNERS)}")
        params = stage.get("params", {}) or {}

        # Resume: if this stage's expected output already exists and is non-empty, skip it.
        # upscale_background doesn't produce a video output (it returns current_input
        # unchanged and stores into context) -- handle it separately.
        if stage_type == "upscale_background":
            prev = next((s for s in existing_manifest.get("stages_run", []) if s["index"] == idx), None)
            if prev:
                name = params.get("name", "background")
                # Restore what the stage stored in context so later stages can reference it.
                for s in existing_manifest.get("stages_run", []):
                    if s["index"] == idx:
                        # The stage runner stores context[name]=resolved_path; recover it from
                        # the bg_replace stage's params in the manifest isn't reliable -- instead
                        # re-run the stage (it's fast, just a file check / small upscale).
                        break
        else:
            # Determine expected output file name (mirrors naming in each run_* function).
            suffix_map = {
                "crop_video": f"stage{idx}_crop_video.mp4",
                "upscale":    f"stage{idx}_upscale.mp4",
                "bg_replace": f"stage{idx}_bg_replace.mp4",
                "bg_remove":  f"stage{idx}_bg_remove.mp4",
                "enhance":    f"stage{idx}_enhance.mp4",
                "audio":      f"stage{idx}_audio.mp4",
            }
            cached = work_dir / suffix_map.get(stage_type, "")
            if cached.exists() and cached.stat().st_size > 0:
                print(f"[workflow] stage {idx} ({stage_type}): CACHED — reusing {cached} ({cached.stat().st_size // 1_000_000}MB)")
                current = cached
                manifest["stages_run"].append({"index": idx, "type": stage_type, "params": params, "output": str(current), "cached": True})
                continue

        current = runner(current, params, work_dir, idx, context)
        manifest["stages_run"].append({"index": idx, "type": stage_type, "params": params, "output": str(current)})
        # Write manifest after every stage so a mid-run kill preserves progress for resume.
        manifest_path.write_text(json.dumps(manifest, indent=2))

    manifest_path.write_text(json.dumps(manifest, indent=2))
    shutil.copy(current, output)
    print(f"[workflow] final output: {output}")
    print(f"[workflow] stage manifest: {work_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
