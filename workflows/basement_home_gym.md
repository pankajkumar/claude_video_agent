---
name: basement_home_gym
description: >
  Basement gym (wood floor, yellow/grey accent wall), sharp throughout.
  Uses the ORIGINAL (non-upscaled) background image -- AI-upscaling this
  background was tried and deliberately NOT used here, see notes below.
  grain disabled (was the dominant visible "quality loss" -- see notes).
stages:
  - type: bg_replace
    params:
      background: "/Users/falguneesharma/pankaj/gym images/Basement-Home-Gym.jpg"
      content_crop: "608:1080:656:0"
      estimate: false
      variant: resnet50
      device: auto
      bg_blur: 1.8
      fg_sharpen: 0.3
      relight_strength: 0.064
      relight_luma_weight: 0.073
      relight_smoothing: 0.959
      edge_feather: 1
      edge_despill: 0.2
      bg_anchor_x: 0.5
      bg_anchor_y: 0.0
      bg_crop_top: 0.3
      bg_crop_bottom: 0.0
      denoise: 0.06
      sharpen: 0.38
      grain: 0.0
      contrast: 1.06
      saturation: 1.10
      brightness: 0.0393
      warmth: 0.03
      vignette: 1.0
---

# basement_home_gym

From the 14-background batch review (2026-06-29), user picked this one. Params
came from `estimate_params.py`'s per-image analysis, `pipeline_agent.py`
passed all checks on iteration 1.

**On AI-upscaling this background (decided against, 2026-06-29):**
`Basement-Home-Gym.jpg` is 1200x895, below the ~1.21x cover_resize would need
for a 608x1080 canvas -- a real quality consideration (see README lesson
#11). An `upscale_background` workflow stage was built for this
(`workflows/basement_home_gym_upscaled.md` demonstrates it) and a real bug
in `upscale_realesrgan.py` was found and fixed along the way (lesson #12:
fixed-4x models corrupted by a non-native `-s` value). But once actually
A/B'd in this workflow, the upscale's detail gain turned out to be
*invisible* here specifically because `bg_blur: 1.8` (intentional, matches
the source camera's own softness -- README lesson #5) smooths away exactly
the fine detail Real-ESRGAN adds. Confirmed by re-testing both versions with
blur removed: the difference is real and visible without blur, but with the
blur this workflow actually uses, upscaling added complexity for no visible
benefit. Decision: keep the original image, skip the upscale stage.

**On `grain` (disabled here, 2026-06-29):** the synthetic film-grain filter
(`noise=alls=...` in enhance_grade.py) was the actual dominant source of
visible "quality loss" reported on this background -- confirmed by an A/B
with grain on (1.42) vs off: clearly visible speckling on flat surfaces
(wall, carpet) with grain on, none with it off. Worth revisiting per-clip if
a deliberate filmic/grainy look is wanted, but defaults to off.
