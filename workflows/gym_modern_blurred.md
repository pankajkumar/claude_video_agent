---
name: gym_modern_blurred
description: >
  Same gym background as gym_dark_moody, but with a deliberate uniform
  background blur so focus reads on the person ("focus on the person" look).
  Uniform blur, not a depth gradient -- see README lesson #9 for why a
  2D-distance-based depth ramp was tried and is NOT used here by default.
stages:
  # See gym_dark_moody.md's comment: enhance_grade.py params live INSIDE the
  # bg_replace stage's params (pipeline_agent.py grades every iteration as
  # part of its own verify/tune loop) -- not a separate trailing stage.
  - type: bg_replace
    params:
      background: assets/backgrounds/gym_modern_minimal.jpg
      content_crop: "608:1080:656:0"
      estimate: false
      variant: resnet50
      device: auto
      bg_blur: 3.5
      fg_sharpen: 0.3
      relight_strength: 0.114
      relight_luma_weight: 0.122
      edge_feather: 2
      edge_despill: 0.6
      bg_anchor_x: 0.5
      bg_anchor_y: 0.3
      bg_crop_top: 0.0
      bg_crop_bottom: 0.0
      denoise: 0.5
      sharpen: 0.4
      grain: 1.2
      contrast: 1.06
      saturation: 1.10
      warmth: 0.03
      vignette: 1.0
      verify:
        sharpness-ratio-max: 150  # deliberate strong blur -- don't let the
                                  # auto-tuner try to "fix" this on its own
---

# gym_modern_blurred

**Confirmed by user 2026-06-29** as one of two preferred looks ("option C")
out of a 4-candidate review. `bg_blur=3.5` is a deliberate stylistic choice,
well past what `estimate_params.py` would suggest just to match camera
softness (~2.0) -- that's why `estimate: false` and every value is pinned
here rather than re-derived.

A gentler depth-gradient variant (`--bg-blur-near`/`--bg-blur-ramp-px` in
replace_background.py, small near/far gap + a wide ramp) was also tried and
looked plausible, but still carries a faint physically-incorrect softness
gradient along rigid straight background lines (no real depth data backing
it) -- this uniform-blur version is the one actually confirmed.

Apply to the full clip:
    python3 scripts/run_workflow.py --workflow workflows/gym_modern_blurred.md \
      --input /Users/falguneesharma/pankaj/video/glycogen/glycogen3.mp4 \
      --output /path/to/glycogen3_final_blurred.mp4
