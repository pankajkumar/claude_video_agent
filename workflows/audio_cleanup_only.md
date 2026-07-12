---
name: audio_cleanup_only
description: >
  Audio-only pass -- loudness normalize + light denoise. Video stream is
  stream-copied untouched, so this is safe and fast to run on an already-
  finished video.
stages:
  - type: audio
    params:
      normalize: true
      target_lufs: -16.0
      denoise: 8
---

# audio_cleanup_only

No video re-encoding happens at all in this workflow -- `process_audio.py`
always uses `-c:v copy`.
