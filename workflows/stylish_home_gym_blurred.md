---
name: stylish_home_gym_blurred
description: >
  Same background/look as stylish_home_gym, but with a deliberate uniform
  background blur so focus reads on the person ("focus on the person" look).
  Uniform blur, not a depth gradient -- see README lesson #9 for why a
  2D-distance-based depth ramp was tried and is NOT used here by default.
stages:
  - type: upscale_background
    params:
      image: "/Users/falguneesharma/pankaj/gym images/Stylish-home-gym-ideas-by-Decorilla-interior-designer-Lori-D.webp"
      content_crop: "608:1080:656:0"
      bg_crop_top: 0.0
      bg_crop_bottom: 0.0
      name: bg
  # See gym_dark_moody.md's comment: enhance_grade.py params live INSIDE the
  # bg_replace stage's params (pipeline_agent.py grades every iteration as
  # part of its own verify/tune loop) -- not a separate trailing stage.
  - type: bg_replace
    params:
      background: "$bg"
      content_crop: "608:1080:656:0"
      estimate: false
      variant: resnet50
      device: auto
      bg_blur: 4.0
      fg_sharpen: 0.3
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
      sharpen: 0.38
      grain: 0.0
      contrast: 1.06
      saturation: 1.10
      brightness: 0.0393
      warmth: 0.03
      vignette: 1.0
      verify:
        sharpness-ratio-max: 150  # deliberate strong blur -- don't let the
                                  # auto-tuner try to "fix" this on its own
---

# stylish_home_gym_blurred

**Confirmed by user 2026-06-30**, alongside the sharp `stylish_home_gym`
variant, after a 4-candidate review (stylish/basement x sharp/blurred) where
basement was dropped (busy "Fitness" signage behind the subject's head at
every `bg_anchor_x` tried -- no clean crop exists for it; see
`stylish_home_gym.md`'s note for why the rest of these params are what they
are).

`bg_blur: 4.0` is a deliberate stylistic choice, well past what
`estimate_params.py` would suggest just to match camera softness (~1.8) --
that's why `estimate: false` and every value is pinned here rather than
re-derived. `edge_despill: 0.7` / `edge_feather: 3` / `bg_anchor_x: 0.4`
carry over the same boundary-halo fix as the sharp variant (README
lesson #13) -- the bug and the fix are in `replace_background.py` itself, so
they apply regardless of `bg_blur`.

`upscale_background` (added 2026-06-30, see README lesson #14 and
`stylish_home_gym.md`'s matching note): without it, `cover_resize` would
enlarge this 1000x998 photo by ~1.08x -- a heavy blur on top of a blocky
enlargement still reads worse than a heavy blur on top of AI-recovered
detail, so this applies regardless of how strong `bg_blur` is.

`bg_crop_top: 0.0` (changed 2026-06-30 -- was `0.3`, see `stylish_home_gym.md`'s
matching note for the full explanation): this photo's ceiling doesn't
actually need decluttering, and the `0.3` top-crop combined with
`cover_resize`'s own side-crop was cropping three sides instead of one.

Apply to the full clip:
    python3 scripts/run_workflow.py --workflow workflows/stylish_home_gym_blurred.md \
      --input /Users/falguneesharma/pankaj/video/glycogen/glycogen3.mp4 \
      --output /path/to/glycogen3_final_blurred.mp4
