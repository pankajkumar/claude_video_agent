---
name: bg_remove_green
description: >
  Background removal only -- composite the subject onto a flat green
  (chroma-key-style) backdrop, no replacement photo, no color grading.
stages:
  - type: bg_remove
    params:
      color: green
      content_crop: "608:1080:656:0"
      edge_despill: 0.4
---

# bg_remove_green

For when downstream tooling (e.g. a real chroma-key step elsewhere) needs a
flat backdrop rather than a finished composite. Add a trailing `enhance`
stage if you also want denoise/sharpen/grade on top of the flat-color result.
