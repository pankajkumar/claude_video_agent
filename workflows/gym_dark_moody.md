---
name: gym_dark_moody
description: >
  Dark moody gym (linear ceiling lights), sharp throughout -- no deliberate
  background blur. For a camera-facing person standing/talking, head-to-waist
  framing, vertical-video pillarboxed source.
stages:
  # A single bg_replace stage carries BOTH replace_background.py params AND
  # enhance_grade.py params together -- pipeline_agent.py's own auto-tune
  # loop runs matte+composite -> enhance/grade -> verify every iteration (the
  # verifier checks the GRADED output), so grading can't be split into a
  # separate trailing stage without either double-grading or breaking the
  # verify/tune loop's own checks. Use a separate `enhance` stage only for an
  # enhance-only workflow with no background work at all.
  - type: bg_replace
    params:
      background: assets/backgrounds/gym_modern_minimal.jpg
      content_crop: "608:1080:656:0"
      estimate: false  # every scene-dependent param below is already pinned
      variant: resnet50
      device: auto
      bg_blur: 2.0
      fg_sharpen: 0.3
      relight_strength: 0.114
      relight_luma_weight: 0.122
      relight_smoothing: 0.959
      edge_feather: 1
      edge_despill: 0.2
      bg_anchor_x: 0.7
      bg_anchor_y: 0.0
      bg_crop_top: 0.3
      bg_crop_bottom: 0.0
      denoise: 0.06
      sharpen: 0.38
      grain: 1.42
      contrast: 1.06
      saturation: 1.10
      brightness: 0.0393
      warmth: 0.03
      vignette: 1.0
---

# gym_dark_moody

**Confirmed by user 2026-06-29** as one of two preferred looks ("option A")
out of a 4-candidate review (A dark/sharp, B bright/sharp, C blurred-bg,
D blurred-bg with depth gradient).

All params here came from `estimate_params.py`'s analysis of the actual test
clip (`glycogen3.mp4`, 5–8s) against `assets/backgrounds/gym_modern_minimal.jpg`,
followed by `pipeline_agent.py`'s auto-tune loop converging to a passing
result in 1 iteration. `bg_anchor_x=0.7, bg_anchor_y=0.0, bg_crop_top=0.3`
came from the automated background-framing clutter search (see README) --
re-run `estimate_params.py` (or set `estimate: true` above) if you point this
at different footage or a different background image, since these values are
specific to this exact pair.

Apply to the full clip:
    python3 scripts/run_workflow.py --workflow workflows/gym_dark_moody.md \
      --input /Users/falguneesharma/pankaj/video/glycogen/glycogen3.mp4 \
      --output /path/to/glycogen3_final.mp4
