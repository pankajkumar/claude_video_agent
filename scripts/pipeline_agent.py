#!/usr/bin/env python3
"""
End-to-end agentic pipeline: estimate starting params from the actual footage
-> background replacement -> enhance -> cinematic grade -> verify -> auto-tune
and retry on failure.

This is the entry point an AI agent should call for "replace the background
of this video" requests. It wraps estimate_params.py, replace_background.py,
enhance_grade.py, and verify_quality.py, and closes the loop: after each
render it runs the quality verifier, and if a check fails, nudges the
relevant parameter(s) proportionally to how far off the metric is and
re-renders (up to --max-iterations times). Every iteration's params +
verifier report are logged to <output>.agent_work/iterations.json so the
tuning history is inspectable.

Usage:
    python3 pipeline_agent.py \
        --input /path/to/video.mp4 \
        --background "/path/to/bg.jpg" \
        --output /path/to/final.mp4 \
        --content-crop 608:1080:656:0   # optional, strip pillarbox bars first

Run `python3 detect_content_crop.py --input video.mp4` to get --content-crop
if the source has black bars.

Three params are scene-dependent (how soft the source camera is, how
bright/sharp the background photo is, how different its color temperature is
from the subject) and are NOT a fixed preset here: bg_blur, fg_sharpen,
relight_strength are estimated per-(clip, background) pair by
estimate_params.py before the first render. Pass --no-estimate to fall back
to DEFAULT_PARAMS' static values instead, or override any of them directly:
    --param bg_blur=2.0 --param fg_sharpen=0.5 --param relight_strength=0.15

Everything the auto-tune loop does is controlled by TUNING_DEFAULTS below,
itself overridable via --tune-param (same key=value syntax as --param) --
there are no nudge amounts or pass/fail thresholds buried inline in the
adjust_params() logic.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

# Params that are genuinely subjective creative choices, not measurable
# footage properties -- see estimate_params.py's module docstring for the
# full rationale on what's covered there vs. left static here.
# downsample_ratio is intentionally absent: leaving it unset lets
# replace_background.py auto-scale it from the video's own resolution.
DEFAULT_PARAMS = {
    # replace_background.py
    "variant": "resnet50",
    "device": "auto",
    "bg_crop_bottom": 0.0,  # rarely needed; bg_crop_top is estimated, this isn't
    # enhance_grade.py -- color-grade taste, not derived from the footage
    "contrast": 1.06,
    "saturation": 1.10,
    "warmth": 0.03,
    "vignette": 1.0,
}

# Used only when --no-estimate is passed (estimate_params.py skipped) --
# every key estimate_params.py would otherwise compute, frozen at the values
# this project's original tuning converged on for one specific video. Not a
# universal preset; --no-estimate exists for speed (skips the matting-model
# sampling pass) or as a fallback if the estimator's own sampling fails.
FALLBACK_SCENE_PARAMS = {
    "bg_blur": 1.3,
    "fg_sharpen": 0.5,
    "relight_strength": 0.18,
    "relight_luma_weight": 0.25,
    "relight_smoothing": 0.9,
    "edge_feather": 1,
    "edge_despill": 0.6,
    "bg_anchor_x": 0.5,
    "bg_anchor_y": 0.3,
    "bg_crop_top": 0.0,
    "denoise": 1.0,
    "sharpen": 0.4,
    "grain": 1.5,
    "brightness": 0.04,
}

# Every gain/bound/threshold the auto-tune loop uses. Override any of them
# with --tune-param key=value. The *_low/*_high pairs are kept in sync with
# verify_quality.py's own --sharpness-ratio-min/max and --brightness-min/max
# defaults by default -- pass matching --verify-param overrides if you change
# one side, so the tuner is chasing the same target the verifier checks.
TUNING_DEFAULTS = {
    "sharpness_ratio_low": 0.08,     # below -> subject reads softer than bg
    "sharpness_ratio_high": 6.0,     # above -> subject oversharpened/noisy vs bg
    "bg_blur_gain": 0.6,             # bg_blur increment per failing iteration (too-soft case)
    "bg_blur_max": 8.0,
    "fg_sharpen_gain": 0.15,         # small: oversharpening a face reads as "pixelated"
    "fg_sharpen_max": 1.5,
    "fg_sharpen_relief": 0.5,        # decrement when subject is oversharpened/noisy
    "edge_despill_gain": 0.2,
    "edge_despill_max": 1.0,
    "edge_feather_gain": 1,
    "edge_feather_max": 3,
    "brightness_low": 45.0,
    "brightness_high": 235.0,
    "relight_luma_weight_relief": 0.1,
    "grade_brightness_gain": 0.04,
}

# Which script each param belongs to.
REPLACE_BG_PARAMS = {
    "variant", "device", "downsample_ratio", "edge_feather", "edge_despill",
    "relight_strength", "relight_luma_weight", "relight_smoothing",
    "bg_anchor_x", "bg_anchor_y", "bg_crop_top", "bg_crop_bottom", "bg_blur", "fg_sharpen",
    # Gentle depth-of-field gradient, off by default (bg_blur_ramp_px=0). Opt
    # in via --param; see replace_background.py --help and README lesson #9
    # for why near/far should stay close together with a WIDE ramp.
    "bg_blur_near", "bg_blur_ramp_px",
}
ENHANCE_GRADE_PARAMS = {
    "denoise", "sharpen", "contrast", "saturation", "brightness", "warmth", "vignette", "grain",
}


def parse_kv_overrides(kv_list, base):
    """--key=value list -> a copy of base with numeric-typed overrides applied."""
    out = dict(base)
    for kv in kv_list:
        k, v = kv.split("=", 1)
        try:
            v = float(v) if "." in v else int(v)
        except ValueError:
            pass
        out[k] = v
    return out


def to_cli(params):
    args = []
    for k in params:
        flag = "--" + k.replace("_", "-")
        args += [flag, str(params[k])]
    return args


def estimate_scene_params(input_video, background, content_crop, estimate_overrides):
    """Call estimate_params.py's library function for this (clip, background)
    pair and return every footage-dependent param it computes (see its module
    docstring for the full list and rationale). estimate_overrides is a dict
    of its keyword-arg names (e.g. num_samples, sigma_max, fg_sharpen_floor --
    see estimate_params.py's estimate_params() signature) to override its own
    defaults."""
    sys.path.insert(0, str(SCRIPT_DIR))
    from estimate_params import estimate_params  # noqa: E402

    kwargs = dict(content_crop=content_crop)
    kwargs.update(estimate_overrides)
    result = estimate_params(input_video, background, **kwargs)
    scene_params = {k: v for k, v in result.items() if not k.startswith("_")}
    print(f"[agent] estimated scene params from footage: "
          + ", ".join(f"{k}={v}" for k, v in scene_params.items()))
    return scene_params


def run_iteration(input_video, background, content_crop, work_dir, iteration, params, verify_params):
    comp_path = work_dir / f"iter{iteration}_composite.mp4"
    final_path = work_dir / f"iter{iteration}_final.mp4"
    diag_path = work_dir / f"iter{iteration}_diagnostics.json"

    rb_params = {k: v for k, v in params.items() if k in REPLACE_BG_PARAMS}
    cmd = [sys.executable, str(SCRIPT_DIR / "replace_background.py"),
           "--input", str(input_video), "--background", str(background),
           "--output", str(comp_path), "--diagnostics-out", str(diag_path)]
    if content_crop:
        cmd += ["--content-crop", content_crop]
    cmd += to_cli(rb_params)
    print(f"[agent] iteration {iteration}: replace_background.py")
    subprocess.run(cmd, check=True)

    eg_params = {k: v for k, v in params.items() if k in ENHANCE_GRADE_PARAMS}
    cmd = [sys.executable, str(SCRIPT_DIR / "enhance_grade.py"),
           "--input", str(comp_path), "--output", str(final_path)]
    cmd += to_cli(eg_params)
    print(f"[agent] iteration {iteration}: enhance_grade.py")
    subprocess.run(cmd, check=True)

    report_path = work_dir / f"iter{iteration}_report.json"
    # verify_quality.py needs to sample the SAME region/blur of the background
    # that was actually composited, else its sharpness_match metric won't
    # respond to changes in bg_blur/bg_anchor_*/bg_crop_* at all.
    bg_render_params = {k: v for k in ("bg_blur", "bg_anchor_x", "bg_anchor_y", "bg_crop_top", "bg_crop_bottom")
                         if (v := params.get(k)) is not None}
    cmd = [sys.executable, str(SCRIPT_DIR / "verify_quality.py"),
           "--video", str(final_path), "--background", str(background),
           "--source-video", str(input_video), "--report", str(report_path)]
    cmd += to_cli(bg_render_params)
    cmd += to_cli(verify_params)
    print(f"[agent] iteration {iteration}: verify_quality.py")
    subprocess.run(cmd, capture_output=True, text=True)
    report = json.loads(report_path.read_text())

    diagnostics = json.loads(diag_path.read_text()) if diag_path.exists() else None

    # replace_background.py computes edge_brightness_halo from the REAL alpha
    # matte during rendering; verify_quality.py only approximates the subject
    # mask post-hoc by diffing the flat output against the background image,
    # which is far noisier (validated: it scored a known-buggy render and its
    # fix almost identically). Prefer the render-time number when present.
    if diagnostics and diagnostics.get("edge_brightness_halo_worst_frame_fraction") is not None:
        halo_max = verify_params.get("halo-fraction-max", 0.12)
        worst = diagnostics["edge_brightness_halo_worst_frame_fraction"]
        report["checks"]["edge_brightness_halo"] = {
            "metric": worst,
            "worst_frame_fraction": worst,
            "mean_frame_fraction": diagnostics.get("edge_brightness_halo_mean_frame_fraction"),
            "pass": worst < halo_max,
            "source": "replace_background.py (real alpha)",
            "suggestion": (
                "localized bright rim at the matte boundary: increase --edge-despill, or if it "
                "persists, the background photo's own content directly behind a limb may be "
                "bright/light there -- try a different --bg-anchor-x/y or crop"
                if worst >= halo_max else None
            ),
        }
        report["overall_pass"] = all(c["pass"] for c in report["checks"].values())

    return final_path, report, diagnostics


def adjust_params(params, report, tuning):
    """Apply the verifier's suggestions to params for the next iteration,
    nudging proportionally to how far off each metric is from its pass
    boundary (in tuning), scaled by that param's gain (also in tuning).
    Returns a new params dict; mutates nothing in place."""
    new_params = dict(params)
    checks = report["checks"]

    if not checks.get("edge_fringe", {}).get("pass", True):
        new_params["edge_despill"] = min(tuning["edge_despill_max"], params["edge_despill"] + tuning["edge_despill_gain"])
        new_params["edge_feather"] = min(tuning["edge_feather_max"], params["edge_feather"] + tuning["edge_feather_gain"])

    if not checks.get("edge_brightness_halo", {}).get("pass", True):
        # Same remedy as edge_fringe (stronger despill/feather catches most
        # matte-side halos), but this check can also fail because the
        # background photo itself has a bright/light object sitting directly
        # behind a limb at this anchor/crop -- despill can't fix that, only
        # repositioning can. adjust_params has no visibility into the image
        # content, so it pushes the matte-side knobs and lets verify_quality's
        # suggestion string flag the anchor/crop possibility for a human.
        new_params["edge_despill"] = min(tuning["edge_despill_max"], params["edge_despill"] + tuning["edge_despill_gain"])
        new_params["edge_feather"] = min(tuning["edge_feather_max"], params["edge_feather"] + tuning["edge_feather_gain"])

    sharp = checks.get("sharpness_match", {})
    if not sharp.get("pass", True):
        ratio = sharp.get("metric")
        low, high = tuning["sharpness_ratio_low"], tuning["sharpness_ratio_high"]
        if ratio is not None and ratio < low:
            # Subject reads softer than the background. Per README lesson #5,
            # prefer softening the (often already-soft-source) background to
            # match rather than oversharpening the subject's face -- pushing
            # fg_sharpen up aggressively produces ringing/halo artifacts
            # around hair, eyebrows, glasses rims that read as "pixelated".
            error = (low - ratio) / low  # 0..1+, how far below the boundary
            new_params["bg_blur"] = min(tuning["bg_blur_max"], params["bg_blur"] + tuning["bg_blur_gain"] * (1 + error))
            new_params["fg_sharpen"] = min(tuning["fg_sharpen_max"], params["fg_sharpen"] + tuning["fg_sharpen_gain"])
        elif ratio is not None and ratio > high:
            new_params["fg_sharpen"] = max(0.0, params["fg_sharpen"] - tuning["fg_sharpen_relief"])

    bright = checks.get("brightness", {})
    if not bright.get("pass", True):
        luma = bright.get("metric")
        if luma is not None and luma < tuning["brightness_low"]:
            new_params["relight_luma_weight"] = max(0.0, params["relight_luma_weight"] - tuning["relight_luma_weight_relief"])
            new_params["brightness"] = params["brightness"] + tuning["grade_brightness_gain"]
        elif luma is not None and luma > tuning["brightness_high"]:
            new_params["brightness"] = params["brightness"] - tuning["grade_brightness_gain"]

    return new_params


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--background", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--content-crop", default=None, help="w:h:x:y, see module docstring")
    p.add_argument("--max-iterations", type=int, default=3)
    p.add_argument("--param", action="append", default=[],
                    help="override a default/estimated param, e.g. --param bg_anchor_y=0.3")
    p.add_argument("--tune-param", action="append", default=[],
                    help="override an auto-tune gain/bound, e.g. --tune-param bg_blur_gain=1.0 (see TUNING_DEFAULTS)")
    p.add_argument("--verify-param", action="append", default=[],
                    help="forwarded to verify_quality.py, e.g. --verify-param sharpness-ratio-min=0.05")
    p.add_argument("--no-estimate", action="store_true",
                    help="skip estimate_params.py and use FALLBACK_SCENE_PARAMS instead of analyzing the footage")
    p.add_argument("--estimate-param", action="append", default=[],
                    help="forwarded to estimate_params.estimate_params(), e.g. --estimate-param num_samples=8")
    p.add_argument("--work-dir", default=None, help="where intermediate renders/reports go (default: <output>.agent_work/)")
    p.add_argument("--auto-upscale-background", action="store_true",
                    help="if the background image's native resolution is too low for the output canvas (would be "
                         "upscaled by cover_resize, magnifying the source JPEG/WebP's own compression artifacts -- "
                         "see README), AI-upscale it first via prepare_background.py/Real-ESRGAN. Off by default "
                         "since it requires the Real-ESRGAN ncnn-vulkan binary (see tools/realesrgan-ncnn/); "
                         "replace_background.py prints a warning either way when this would help.")
    p.add_argument("--upscale-margin", type=float, default=1.0,
                    help="with --auto-upscale-background: only upscale if native resolution would need more than "
                         "this much cover_resize scaling (1.0 = upscale whenever any upscale at all is needed)")
    args = p.parse_args()

    if args.auto_upscale_background:
        if args.content_crop:
            canvas_w, canvas_h, _, _ = (int(v) for v in args.content_crop.split(":"))
        else:
            import cv2
            cap = cv2.VideoCapture(args.input)
            canvas_w, canvas_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        sys.path.insert(0, str(SCRIPT_DIR))
        from prepare_background import prepare_background  # noqa: E402
        try:
            args.background = prepare_background(args.background, canvas_w, canvas_h, margin=args.upscale_margin)
        except subprocess.CalledProcessError as e:
            print(f"[warn] --auto-upscale-background failed ({e}); continuing with the original background image")

    params = dict(DEFAULT_PARAMS)
    if args.no_estimate:
        params.update(FALLBACK_SCENE_PARAMS)
    else:
        estimate_overrides = parse_kv_overrides(args.estimate_param, {})
        params.update(estimate_scene_params(args.input, args.background, args.content_crop, estimate_overrides))
    params = parse_kv_overrides(args.param, params)

    tuning = parse_kv_overrides(args.tune_param, TUNING_DEFAULTS)

    verify_params = {}
    for kv in args.verify_param:
        k, v = kv.split("=", 1)
        verify_params[k.replace("-", "_")] = v

    output = Path(args.output)
    work_dir = Path(args.work_dir) if args.work_dir else output.parent / f"{output.stem}.agent_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    history = []
    final_path, report = None, None
    for i in range(1, args.max_iterations + 1):
        final_path, report, diagnostics = run_iteration(
            args.input, args.background, args.content_crop, work_dir, i, params, verify_params
        )
        history.append({"iteration": i, "params": dict(params), "report": report, "diagnostics": diagnostics})
        (work_dir / "iterations.json").write_text(json.dumps(history, indent=2))

        if report["overall_pass"]:
            print(f"[agent] iteration {i}: PASSED all checks")
            break
        print(f"[agent] iteration {i}: FAILED checks -> {[k for k, v in report['checks'].items() if not v['pass']]}")
        if i < args.max_iterations:
            params = adjust_params(params, report, tuning)
        else:
            print("[agent] max iterations reached, using best-effort result")

    shutil.copy(final_path, output)
    print(f"[agent] final output: {output}")
    print(f"[agent] iteration history: {work_dir / 'iterations.json'}")
    sys.exit(0 if report["overall_pass"] else 1)


if __name__ == "__main__":
    main()
