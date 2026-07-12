---
name: basement_home_gym_upscaled
description: >
  Same as basement_home_gym.md, but demonstrates the optional
  upscale_background stage instead of pointing straight at a pre-upscaled
  file in assets/backgrounds/. Native source is 1200x895, below what a
  608x1080 canvas needs (would be upscaled ~1.21x by cover_resize otherwise
  -- see README lesson #11).
stages:
  - type: upscale_background
    params:
      image: "/Users/falguneesharma/pankaj/gym images/Basement-Home-Gym.jpg"
      content_crop: "608:1080:656:0"
      name: bg
  - type: bg_replace
    params:
      background: "$bg"
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
      grain: 1.42
      contrast: 1.06
      saturation: 1.10
      brightness: 0.0393
      warmth: 0.03
      vignette: 1.0
---

# basement_home_gym_upscaled

Upscaling is optional and explicit: this stage can be deleted entirely (and
`background: "$bg"` changed to the literal original path) with zero effect
on anything else in the workflow. It was added here specifically to validate
the upscale_realesrgan.py fix (see README lesson #11/#12) -- `-s 2` on the
fixed-4x `realesrgan-x4plus` model used to silently produce a corrupted
mirrored-tile mosaic; confirmed fixed by checking the resolved `$bg` image
dimensions and content before trusting a render against it.
