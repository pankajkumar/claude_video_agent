---
name: workflow_template
description: >
  Template/reference for authoring a workflow. Copy this file, rename it,
  edit the `stages` list. Run with:
    python3 scripts/run_workflow.py --workflow workflows/<name>.md \
      --input video.mp4 --output final.mp4

# `stages` is an ordered list. Each stage's output feeds the next stage's
# input automatically. Stage types (see scripts/run_workflow.py's
# STAGE_RUNNERS for the full/current list):
#
#   bg_replace  -- full background replacement against a photo. Wraps
#                  pipeline_agent.py (estimate -> render -> grade -> verify ->
#                  auto-tune -> retry). `params.background` is required.
#                  IMPORTANT: this stage's params carry BOTH
#                  replace_background.py params AND enhance_grade.py params
#                  (denoise/sharpen/grain/contrast/saturation/brightness/
#                  warmth/vignette) together in one flat dict -- pipeline_
#                  agent.py grades every iteration as part of its own
#                  verify/tune loop (the verifier checks the GRADED output),
#                  so do NOT also add a separate `enhance` stage after this
#                  one, that would grade twice. If params already pins
#                  bg_blur/fg_sharpen/relight_strength, estimate_params.py's
#                  analysis pass is skipped automatically (estimate: false)
#                  since there's nothing left to estimate -- set
#                  `estimate: true` explicitly to force re-analysis anyway.
#                  `verify` / `tune` / `estimate_overrides` sub-maps forward to
#                  --verify-param / --tune-param / --estimate-param.
#
#   bg_remove   -- matte + composite onto a flat color (no scene to match,
#                  so no estimate/verify/auto-tune loop, and no grading --
#                  add a separate `enhance` stage after this one if you want
#                  denoise/sharpen/color-grade too). `params.color` is a name
#                  (green/black/white/gray/blue) or hex "#RRGGBB".
#
#   enhance     -- denoise/sharpen + cinematic color grade only, no matting
#                  at all. Same params as scripts/enhance_grade.py. Use this
#                  as the ONLY stage for an enhance-only/no-bg-work job, or
#                  after a `bg_remove` stage -- not after `bg_replace` (see
#                  above).
#
#   audio       -- audio-only processing, video stream untouched. Same
#                  params as scripts/process_audio.py (normalize, denoise,
#                  volume_db, replace_audio).
#
#   upscale     -- Real-ESRGAN upscale of the VIDEO. Same params as
#                  scripts/upscale_realesrgan.py. Slow -- see README's
#                  "Why RVM" section before adding this to a long clip.
#
#   upscale_background -- OPTIONAL. AI-upscale (Real-ESRGAN) a background
#                  IMAGE before a later bg_replace/bg_remove stage uses it,
#                  if its native resolution is too low for the output canvas
#                  (see README lesson #11: cover_resize would otherwise
#                  upscale it with plain Lanczos, magnifying the source
#                  JPEG/WebP's own compression artifacts). Does not touch
#                  the video at all -- it resolves an image path and stores
#                  it under `params.name` (default "background") for a
#                  LATER stage to reference via `background: "$<name>"`.
#                  Entirely opt-in: omit this stage and use a literal path
#                  in `background` instead, and nothing else changes.
#                  params: image (required), content_crop (or it reads the
#                  video's own dimensions), margin (default 1.0 -- only
#                  upscales if truly needed), model, name, force.
#
# Examples of different job shapes, all expressible as a stages list:
#   - background replacement only:        [bg_replace]
#   - background replacement, blurred bg: [bg_replace]  (set bg_blur higher)
#   - background replacement, low-res bg image: [upscale_background, bg_replace]
#   - background removal only:            [bg_remove]
#   - enhance/grade only, no bg work:      [enhance]
#   - audio only, no video work:           [audio]
#   - everything (bg + grade, then audio pass): [bg_replace, audio]
#
# Any param can be overridden at run time without editing this file:
#   --stage-param bg_replace.bg_blur=4.0   (targets by stage type)
#   --stage-param 0.bg_blur=4.0            (targets by 0-based position)

stages:
  - type: bg_replace
    params:
      background: assets/backgrounds/gym_modern_minimal.jpg
      content_crop: "608:1080:656:0"
      # bg_blur / fg_sharpen / relight_strength intentionally omitted here --
      # leaving them out means estimate_params.py computes them from the
      # actual (clip, background) pair at run time instead of reusing a
      # fixed value. Pin them (see the confirmed workflows in this folder
      # for real examples) once you've confirmed a result you like.
      # enhance_grade.py params (contrast/saturation/warmth/vignette/denoise/
      # sharpen/grain/brightness) belong HERE too, not in a separate stage --
      # see the note on `bg_replace` above.
      contrast: 1.06
      saturation: 1.10
      warmth: 0.03
      vignette: 1.0
---

# Workflow template

This is reference documentation, not a workflow meant to be run as-is on
real footage (though it will run -- it just always re-estimates params).
See `gym_dark_moody.md` and `gym_modern_blurred.md` in this folder for
fully-pinned, confirmed examples.
