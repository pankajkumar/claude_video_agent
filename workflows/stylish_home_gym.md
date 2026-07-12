---
name: stylish_home_gym
description: >
  Bright, airy stylish home gym, sharp throughout. AI-upscales the
  background first (see notes -- supersedes the 2026-06-29 decision against
  this, which was computed on a bug that understated the real upscale need).
  grain disabled (was the dominant visible "quality loss" -- see notes).
stages:
  # No upscale_background stage needed: the background already exists as a
  # pre-computed 2x Real-ESRGAN upscale (2000x1996). For the 1216x2160
  # canvas, cover_resize applies a 1.08x scale from that asset (barely over
  # 1.0, same as the original situation at the smaller 608x1080 canvas) --
  # visually clean, no blocky artifacts. If a cleaner result is ever needed,
  # run: python3 scripts/prepare_background.py \
  #   --image "Stylish-.webp" --canvas-width 1216 --canvas-height 2160
  # and update the bg_replace background path to the _upscaled_3x.png result.

  # Stage 1: crop the person video to the content area ONLY (strip black bars)
  # so the next stage (upscale) processes only the subject pixels, not the
  # full 1920x1080 frame including bars. output: 608x1080.
  - type: crop_video
    params:
      content_crop: "608:1080:656:0"

  # Stage 2: AI-upscale the cropped person video 2x (608x1080 -> 1216x2160).
  # The matting model then sees 2x more pixels -- finer edges, better hair
  # detail, sharper matte. All subsequent processing runs at this resolution.
  # WARNING: ~11s/frame on Apple M5 -- ~33 min for a 3s/180-frame clip,
  # ~8.5h for the full 46s/2775-frame clip. Run the 3s clip to verify first.
  - type: upscale
    params:
      scale: 2
      model: realesrgan-x4plus

  # Stage 3: background replace. Input is now 1216x2160 (already cropped+
  # upscaled) so no content_crop needed here. Background is resolved from
  # the $bg upscale_background result above.
  - type: bg_replace
    params:
      background: "/Users/falguneesharma/pankaj/gym images/Stylish-home-gym-ideas-by-Decorilla-interior-designer-Lori-D_upscaled_2x.png"
      estimate: false
      variant: resnet50
      device: auto
      bg_blur: 0.0
      fg_sharpen: 0.15
      relight_strength: 0.078
      relight_luma_weight: 0.082
      relight_smoothing: 0.959
      edge_feather: 3
      edge_despill: 0.7
      bg_anchor_x: 0.4
      bg_anchor_y: 0.0
      bg_crop_top: 0.0
      bg_crop_bottom: 0.0
      denoise: 0.06
      sharpen: 0.15
      grain: 0.0
      contrast: 1.06
      saturation: 1.10
      brightness: 0.0393
      warmth: 0.03
      vignette: 1.0
---

# stylish_home_gym

From the 14-background batch review (2026-06-29), user picked this one. Params
came from `estimate_params.py`'s per-image analysis, `pipeline_agent.py`
passed all checks on iteration 1.

**On AI-upscaling this background (reversed, 2026-06-30 -- was decided
against on 2026-06-29):** the original decision computed the upscale need
as ~1.08x and judged the detail gain not worth it once `bg_blur: 1.8`
smooths it away. Now upscaled regardless -- see README lesson #14.

**`bg_crop_top: 0.0` (changed 2026-06-30 -- was `0.3`):** the original
`0.3` top-crop was meant to discard a cluttered ceiling/lights band per
README lesson #2's general guidance, but was never actually verified
against *this specific* photo's ceiling, which is plain white with only a
small AC vent -- no clutter to remove. Worse, combining that *separate*
top-crop with `cover_resize`'s own unavoidable side-crop (this photo is
much wider than the portrait canvas) cropped the image on three sides
total (top + both sides) instead of one, which is what was actually wrong
(caught by user review, not by any automated check -- see README lesson
#16). With `bg_crop_top: 0.0`, the math works out so the FULL vertical
extent of the (upscaled) image maps exactly to the canvas height with zero
vertical cropping and a pure *downscale* (0.54x, never an enlargement) --
only the sides are cropped, satisfying single-axis cropping. If a
background photo's ceiling genuinely needs decluttering, lesson #2 still
applies -- just verify it's actually needed for that specific photo first
(render a background-only canvas preview, lesson #16) rather than copying
a `bg_crop_top` value from a different background's workflow.

**On `grain` (disabled here, 2026-06-29):** see `basement_home_gym.md`'s
identical note -- the synthetic film-grain filter was the dominant source of
visible "quality loss," confirmed by A/B (clear speckling on flat surfaces
with grain on, none with it off). Defaults to off; revisit per-clip if a
deliberate filmic/grainy look is wanted.

**`edge_despill: 0.7` / `edge_feather: 3` / `bg_anchor_x: 0.4`
(updated 2026-06-30, confirmed by user):** the original values above
(`edge_despill: 0.2`, `edge_feather: 1`, `bg_anchor_x: 0.7`) produced a
visible bright rim along the silhouette boundary, most noticeable on an
elbow against the background's treadmill. Root-caused to two bugs in
`replace_background.py` (see README lesson #13: relight's interior-weight
ramp too narrow, and edge despill only fixing chroma not brightness
contamination) — both fixed at the source, and `edge_despill`/`edge_feather`
bumped here to fully clear it. Separately, `bg_anchor_x: 0.7` framed the
background's treadmill rail directly behind the hanging arm regardless of
matte quality — `0.4` frames a plain wall there instead. Re-verified with
the new `edge_brightness_halo` check (see README): 0.0% worst-frame fraction
with these params, vs. a real defect at the old ones.

Apply to the full clip:
    python3 scripts/run_workflow.py --workflow workflows/stylish_home_gym.md \
      --input /Users/falguneesharma/pankaj/video/glycogen/glycogen3.mp4 \
      --output /path/to/glycogen3_final_sharp.mp4
