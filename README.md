# claude_video_agent

Agentic toolkit for replacing a video's background with a still image so it
looks like it was shot there — built and tuned iteratively against a real
talking-head gym video. Designed to be invoked by an AI agent end-to-end.

## DaVinci Resolve MCP connector (`tools/davinci-resolve-mcp/`)

Installed via [samuelgursky/davinci-resolve-mcp](https://github.com/samuelgursky/davinci-resolve-mcp)
(no official Blackmagic-built MCP server exists; this is the most complete
community one, built on Resolve's official Scripting API) and registered in
`.mcp.json` at the project root, using a `uv`-managed Python 3.12 venv at
`tools/davinci-resolve-mcp/venv` (separate from the pipeline's own 3.9 `.venv`
since the `mcp` SDK requires 3.10+).

**It will not connect right now** — DaVinci Resolve's scripting API is a
**Resolve Studio-only feature**. Blackmagic's own scripting README says so
explicitly ("introduction to the Scripting API for DaVinci Resolve Studio"),
and the free edition's `fusionscript` library returns `None` from
`scriptapp("Resolve")` unconditionally — there's no preference toggle to
enable in the free edition's Preferences (confirmed: it's simply not there),
and no other workaround. This was verified directly against this machine's
install (DaVinci Resolve 21, free).

To activate it: install **DaVinci Resolve Studio** (one-time purchase, no
subscription), open Preferences > General > "External scripting using" >
set to **Local**, restart Resolve. The MCP server config is already correct
and needs no changes — `.mcp.json` will just start working.

Without Studio, file-based interop (export an FCPXML/EDL timeline that you
import into free Resolve by hand) is the available alternative if scripted
Resolve control is needed before upgrading — ask to have that added if useful.

## Quick start

**If you already have a saved workflow** (see `workflows/*.md` — a confirmed
look, ready to reapply), this is the only command you need:
```
source .venv/bin/activate
python3 scripts/run_workflow.py --workflow workflows/gym_dark_moody.md \
  --input /path/to/video.mp4 --output /path/to/final.mp4
```
A workflow is an ordered list of stages (background replace/remove, enhance,
audio, upscale) — see "Workflows" below for the full picture, including
jobs that don't touch background at all (audio-only, enhance-only, etc).

**If you're starting from scratch** (no saved workflow yet, picking a
background/look for the first time), drop straight to the underlying engine:
```
python3 scripts/pipeline_agent.py \
  --input /path/to/video.mp4 \
  --background assets/backgrounds/gym_modern_minimal.jpg \
  --output /path/to/final.mp4 \
  --content-crop 608:1080:656:0   # optional, see below
```
This runs the full loop: estimate scene-dependent starting params from the
actual clip+background (see `estimate_params.py` below) → matte+composite →
denoise/sharpen → cinematic grade → automated quality check → if it fails,
nudge parameters proportionally and re-render (up to `--max-iterations`,
default 3). Everything is logged to `<output>.agent_work/iterations.json` so
you can see what was tried and why. Once a result looks good, save it as a
workflow (see below) so future runs (or the full-length video) don't repeat
the trial-and-error.

Nothing about the starting params or the tuning behavior is a fixed preset
baked into one specific video:
- `bg_blur` / `fg_sharpen` / `relight_strength` are computed per (clip,
  background) pair by `estimate_params.py` before the first render (pass
  `--no-estimate` to skip this and use the static `FALLBACK_SCENE_PARAMS` in
  `pipeline_agent.py` instead, or `--param key=value` to override any single
  one directly).
- Every gain/bound the auto-tune loop uses (how hard to push `bg_blur` when
  the subject reads too soft, the cap on `fg_sharpen`, etc.) lives in
  `pipeline_agent.py`'s `TUNING_DEFAULTS` dict and is overridable via
  `--tune-param key=value` — none of it is an inline magic number.
- `verify_quality.py`'s pass/fail thresholds are themselves CLI flags
  (`--sharpness-ratio-min`, `--brightness-max`, etc.), forwardable via
  `--verify-param key=value`.

If the source video has black pillarbox/letterbox bars, find the real content
box first:
```
python3 scripts/detect_content_crop.py --input video.mp4
# prints e.g. "--content-crop 608:1080:656:0", paste straight into the command above
```

## Folder layout

```
scripts/
  replace_background.py   core matting + compositing (RobustVideoMatting / RVM)
  enhance_grade.py         denoise/sharpen + cinematic color grade (ffmpeg)
  process_audio.py         audio-only processing (normalize/denoise/volume/replace-track); video stream untouched
  estimate_params.py       analyzes a (clip, background) pair -> data-driven starting params (see below)
  prepare_background.py    AI-upscales a background image if its native resolution is too low for the output canvas (see lesson #11)
  verify_quality.py        post-hoc quality checks, generic (no alpha needed)
  pipeline_agent.py        single-job orchestrator: estimate -> render -> grade -> verify -> auto-tune -> retry
  run_workflow.py          MULTI-STAGE orchestrator: runs a workflows/*.md file's stage list end to end
  detect_content_crop.py   find --content-crop W:H:X:Y for footage with pillarbox bars
  upscale_realesrgan.py    optional AI upscale wrapper, not in the default flow (see below)
  extract_clip.py          cut a short, always-re-encoded test clip out of a longer source video
  contact_sheet.py         labeled thumbnail grid of candidate background images, for picking one
  inspect_edge.py          zoom into a frame's silhouette boundary to check for seams/halos/fringe
  compare_renders.py       side-by-side frame grid across multiple candidate output videos (whole-frame or zoomed region)
models/
  rvm/                     RobustVideoMatting repo + checkpoints (resnet50, mobilenetv3)
tools/
  realesrgan-ncnn/         Real-ESRGAN ncnn-vulkan binary + x4plus models (used by upscale_realesrgan.py)
assets/backgrounds/        background images evaluated/selected during development (see below)
workflows/                 saved workflows: *.md files with YAML frontmatter, run via run_workflow.py (see below)
.venv/                     python deps: torch, torchvision, opencv-python-headless, numpy, PyYAML
```

## Background images

`assets/backgrounds/gym_modern_minimal.jpg` is the winner from evaluating ~13
candidate gym photos against the test footage: a real photo (not a 3D
render) with a plain dark ceiling + simple linear lights, which stays clean
behind the subject's head regardless of framing/zoom — stylized renders,
brick/cork-tile ceilings, and photos with text/decor/people on them all lost.
`gym_brick_runnerup.jpg` is the next-best alternative (brick walls, cork-tile
ceiling — usable but shows more clutter behind the head in close-ups).

When picking a new background for different footage, look for: real photo
(not CG render) for natural noise/lighting match, a plain/simple ceiling or
wall area roughly where the subject's head will land, no text/logos/people,
and lighting temperature not wildly different from the source footage (less
work for `--relight-strength` to do).

## Why RVM, not [other matting tool]

RobustVideoMatting was picked over per-frame tools (rembg/U2Net) because it's
*temporally consistent* — alpha doesn't flicker frame to frame — and over
generic segmentation (Mediapipe selfie-seg) because it produces a proper
alpha+decontaminated-foreground matte, which is what you need for clean
edges around hair. `resnet50` variant = best accuracy; `mobilenetv3` = faster,
use it for quick iteration/preview before committing to a full resnet50 render.

**Real-ESRGAN is present (`tools/realesrgan-ncnn/`, wrapped by
`scripts/upscale_realesrgan.py`) but intentionally not in the default
pipeline.** On this machine (Apple M5, ncnn-vulkan) it runs ~8-10s/frame,
which is hours for anything beyond a few seconds of footage. The
`enhance_grade.py` ffmpeg-based denoise+unsharp pass gives most of the
visible quality benefit in real time instead. Use `upscale_realesrgan.py`
directly for short hero clips, thumbnails, or a handful of still frames where
quality matters more than turnaround — or re-benchmark it on a machine with a
strong discrete GPU, where it may be fast enough to fold into the default
flow.

```
python3 scripts/upscale_realesrgan.py --input clip.mp4 --output clip_2x.mp4 --scale 2
```

## Lessons baked into the defaults (read before changing them)

These came from real failures observed against a real video — don't undo them
without a reason:

1. **Relight the subject's color, not the original scene's lighting** — but
   *only the warm/cool color cast, barely any brightness* (`relight-luma-weight`
   stays low, ~0.25). Shifting brightness to match the new background made the
   face look dark/muddy.
2. **Crop the busy part out of the background image, don't just crop the
   canvas.** If a background photo's ceiling/lights occupy the top 30% of the
   *source* image, no amount of `bg-anchor-y` cropping of the final canvas
   removes enough of it — you have to discard that band from the *source*
   first (`--bg-crop-top`).
3. **Relight/sharpen must skip the alpha boundary, not just the interior.**
   Pixels near the edge of the matte have unreliable "foreground" color
   (RVM extrapolates them); pushing color or contrast there creates a colored
   halo once composited. Both effects are masked by an eroded-alpha
   "interior weight" in `replace_background.py`.
4. **Edge despill is a separate problem from the above** — even with relight/
   sharpen correctly scoped to the interior, *semi-transparent hair-edge
   pixels themselves* still carry a tint bled through from the ORIGINAL
   background (not the new one). `--edge-despill` neutralizes chroma
   specifically in that semi-transparent band. This is what fixes the
   "yellow fringe around hair" symptom.
5. **Match background sharpness to the source footage, not the other way
   around.** If the source camera footage has any softness/motion blur, a
   crisp stock photo background will look obviously pasted in by contrast.
   `--bg-blur` (light Gaussian, ~1.0-1.5) usually reads better than trying to
   sharpen the (already-soft) subject to match a sharp background.
6. **Always re-encode test clips you cut for review** (`-c:v libx264`, not
   `-c copy`). Stream-copying a cut at a non-keyframe boundary can scramble
   frame order on decode (B-frame reordering issues), which looks like a
   pipeline bug but isn't — wasted real debugging time on this once.
7. **Verification must compare apples to apples.** `verify_quality.py`'s
   sharpness check compares the subject against the *local* background patch
   behind them (same coordinates), not the whole background image — a face is
   inherently lower-frequency than e.g. gym equipment, so a whole-image
   comparison flags every face as "too soft" by default. Likewise the fringe
   check only flags edges that are warmer than the background *already is* at
   that spot — some backgrounds (warm gym lighting) are legitimately
   yellow/orange, and that's not fringe.
8. **A visible "stair-step" border around the subject is usually a double
   lossy re-encode, not a bad matte.** The matting alpha itself is normally
   a clean gradient (check with `inspect_edge.py` on a raw-alpha dump if in
   doubt) — but writing composited frames through OpenCV's low-quality
   default `mp4v` codec and then re-encoding that file again, or running
   `enhance_grade.py`'s denoise and color-grade as two separate ffmpeg passes
   instead of one filtergraph, compounds macroblock artifacts hardest right
   at high-contrast silhouette edges. `replace_background.py` now pipes
   frames directly into a single ffmpeg encode; `enhance_grade.py` now chains
   both stages into one pass. Don't reintroduce an intermediate re-encode.
9. **A 2D pixel-distance-based blur ramp looks worse than a uniform blur, not
   better — this was tried and reverted.** The idea: keep the background
   near-sharp right at the subject's silhouette, ramping to full `--bg-blur`
   further away, mimicking depth-of-field falloff. It doesn't work because
   real depth-of-field depends on actual scene *depth*, not 2D screen
   distance from a 2D silhouette — a straight background line (e.g. a gym
   equipment bar) that happens to pass through the "near" zone at one point
   and the "far" zone at another stays sharp on one side and blurry on the
   other, which reads as an obvious halo/ring artifact around the subject,
   worse than the single soft seam a uniform blur produces. Without real
   depth data (a depth-estimation model, stereo, or LiDAR metadata), a
   *uniform* `--bg-blur` is the more honest choice. For a deliberate strong
   blur / "focus on the person" look, keep `--bg-blur` to what still looks
   plausible against that specific background's geometry (start around
   3–4 and check the full frame, not just a zoomed edge crop, for line
   discontinuities) and relax the verifier so it doesn't "fix" your creative
   choice: `--verify-param sharpness-ratio-max=150` (the default `6.0`
   assumes natural camera-matched sharpness, not stylized bokeh).
10. **The verifier must replicate every render param that changes what it's
   measuring against, or it can't see the effect of that param.** Originally
   `verify_quality.py` always sampled the *original unblurred* background
   image for its sharpness check, so increasing `--bg-blur` in the render
   had zero effect on the check — the auto-tuner pushed `fg_sharpen` up
   instead (chasing a check it could actually move), which produces
   ringing/halo artifacts around hair and glasses that read as "pixelated"
   skin. Fixed by giving `verify_quality.py` matching `--bg-blur`/
   `--bg-anchor-x/y`/`--bg-crop-top/bottom` flags, and having
   `pipeline_agent.py` forward the render's actual values automatically. If
   you add a new scene-dependent render param, check whether the verifier
   needs to replicate it too.
11. **"The original photo looks fine, but the background in my video looks
   pixelated" usually means the background's native resolution is lower
   than the output canvas needs.** `cover_resize` scales a background to
   *fill* the canvas (`scale = max(canvas_w/img_w, canvas_h/img_h)`); if the
   source image is smaller than that requires, it gets upscaled with plain
   Lanczos, which faithfully magnifies the source JPEG/WebP's own
   compression artifacts (blocky edges, soft detail) that were invisible at
   the photo's original, smaller display size. A 1536px-square source
   covering a 608x1080 canvas is fine (downscale, 0.7x); a 1200x895 or
   1000x998 stock photo covering the same canvas needs ~1.08-1.21x upscale,
   and that's enough to show. Fix: `scripts/prepare_background.py` checks
   the required scale and AI-upscales (Real-ESRGAN) only if needed, caching
   the result next to the original; `replace_background.py` prints a
   `[warn]` whenever a background would need upscaling so this isn't a
   silent quality loss. Two opt-in ways to actually trigger it: the
   `pipeline_agent.py --auto-upscale-background` flag for one-off CLI use,
   or an `upscale_background` workflow stage (see "Workflows" below) feeding
   `background: "$name"` into a later `bg_replace`/`bg_remove` stage when
   you want it explicit and saved.
12. **`realesrgan-x4plus`/`realesrgan-x4plus-anime` are FIXED 4x networks —
   asking the ncnn-vulkan binary for `-s 2` or `-s 3` with one of them
   corrupted the image into a mirrored/tiled mosaic on this machine, NOT a
   clean 2x/3x upscale.** Found while validating lesson #11's fix: the
   "AI-upscaled" backgrounds looked "distorted, cut off, weird" — turned out
   the upscale step itself was broken, identically in both single-file and
   batch-folder invocation, and even with an explicit `-t` tile-size
   override (ruling out a tiling-stitch bug specifically). Confirmed by
   running the same source at `-s 4` (this model's true native scale, the
   only scale it has actual trained weights for) — clean output, no
   artifact. `scripts/upscale_realesrgan.py` now always invokes fixed-scale
   models at their native scale and does any further downsampling itself
   afterward with a normal `cv2.resize`/Lanczos pass, for both the
   single-image and video-frame-batch code paths. The `realesr-animevideov3`
   family is NOT affected — it ships separately-trained `-x2`/`-x3`/`-x4`
   weight files, so requesting a different `-s` there correctly loads
   different (properly trained) weights instead of forcing a broken
   internal resize. If you add another fixed-scale model later, list it in
   `FIXED_SCALE_MODELS` at the top of that script.
13. **A whole-boundary *average* fringe/halo metric cannot see a defect
   confined to one limb.** `verify_quality.py`'s original `edge_fringe` check
   pools every boundary pixel from every sampled frame into one mean — a
   bright rim covering, say, 10% of one elbow's local boundary is a rounding
   error against the *whole* silhouette's boundary pixel count, so it passed
   clean while a visible halo sat on an elbow the whole time. Two separate
   bugs were hiding behind that blind spot, found only by zooming into a
   specific limb at 6x and comparing against the raw, pre-composite footage
   at the same coordinates (the original footage had no line there — proof
   it was introduced by compositing, not a real skin highlight):
   - `replace_background.py`'s relight step masks itself away from the alpha
     boundary using a separately eroded+blurred "interior weight" so it
     doesn't push color onto unreliable edge pixels (see lesson #3) — but
     that weight's blur kernel (9x9) was narrow enough that its ramp didn't
     line up with the real alpha matte's ramp, leaving a thin band where
     relit and un-relit color met as a visible seam. Widened to 31x31.
   - `--edge-despill` only pulled semi-transparent boundary pixels' *chroma*
     toward neutral (desaturating them), but RVM's raw `fgr` near the
     boundary is also *brightness*-contaminated (extrapolated toward
     whatever's behind it, often brighter/grayer than true skin) — fixing
     hue without fixing brightness still leaves a pale/bright rim once
     alpha-blended. Now inpaints (`cv2.INPAINT_TELEA`) the unreliable band
     from the nearest *reliable* interior color (alpha > 0.7) instead of
     just desaturating in place, then blends that in proportional to
     `--edge-despill` same as before.

   Both are fixed at the source now, but a third, distinct cause can produce
   the exact same symptom with a clean matte: the background *photo itself*
   may have a bright/light object (a treadmill rail, equipment trim) sitting
   directly behind a limb at the chosen crop, which reads as a rim tracing
   the silhouette purely by coincidence of framing. No despill/feather tuning
   fixes that — only `--bg-anchor-x/y`/`--bg-crop-top/bottom` repositioning,
   or a different background photo, does. **`edge_brightness_halo`** (next
   section) tells the two apart: near-zero after the matte-side fix but still
   failing means it's the second cause.
14. **`--bg-crop-top`/`--bg-crop-bottom` shrink the EFFECTIVE source image
   for the upscale-need check, but the check used to run against the
   pre-crop native size.** Found while tracing a "the background still looks
   soft/blocky" report: `replace_background.py`'s own `[warn] ... needs N.NNx
   upscale` line, and `prepare_background.py`'s margin check, both computed
   `N` from the *full* image dimensions, before `--bg-crop-top`/`--bg-crop-bottom`
   discard rows. Discarding rows shrinks the effective source for whichever
   dimension is being cropped, so the REAL required upscale is higher than
   reported — sometimes a lot higher: `stylish_home_gym.md`'s background is
   1000x998 (≈1.08x needed, looked borderline-fine) but after its
   `bg_crop_top: 0.3` the effective source is ~1000x699, which actually needs
   **1.55x**. That gap is exactly why an earlier decision ("upscaling this
   background isn't worth it, the detail gain is invisible under `bg_blur`")
   was wrong — it was evaluated against the wrong, smaller number. Both
   `replace_background.py`'s warning and `prepare_background.py` (now takes
   `--bg-crop-top`/`--bg-crop-bottom`, and `run_workflow.py`'s
   `upscale_background` stage forwards them) compute against the *post-crop*
   size now. If you add another stage/script that pre-processes a background
   image before `cover_resize` sees it, make sure any upscale-need
   calculation accounts for that pre-processing too — this is the same class
   of bug as lesson #10 (verifier must replicate every render param that
   changes what it's measuring).
15. **A video-matting network's `fgr` output is a *reconstruction*, not a
   copy of the input — even in fully-opaque interior pixels where the
   correct answer is unambiguously just the original pixel.** Checked by
   comparing Laplacian-variance sharpness of the original frame against
   RVM's `fgr` in the same solid-foreground region: consistently ~25-30%
   softer, on a clip where >97% of "solid" (alpha > 0.85) pixels were
   already at alpha > 0.999 — i.e. nowhere near the boundary, with no
   decontamination work left to do, yet still measurably smoothed. Composite
   math (`fgr * pha + bg * (1 - pha)`) only strictly *needs* the network's
   reconstructed color where alpha is partial (genuine fg/bg mixing) — in
   fully-opaque regions the original camera pixel IS the correct foreground
   color. `replace_background.py` now substitutes the original pixel back in
   wherever alpha ramps past 0.90→0.999 (smooth blend, not a hard cutoff, so
   it never touches genuinely semi-transparent edge pixels where the network
   reconstruction is actually needed). Recovers real detail the network's
   refinement otherwise softens away, on top of whatever `--fg-sharpen`/grade
   `sharpen` are already doing.
16. **`--bg-crop-top`/`--bg-crop-bottom` and `cover_resize`'s own crop are
   TWO INDEPENDENT crops on potentially different axes — stacking them
   crops the background on 3 sides instead of 1, and nothing catches this
   automatically.** `cover_resize` by construction only ever overflows (and
   therefore crops) ONE axis: it scales by `max(canvas_w/img_w,
   canvas_h/img_h)`, which makes the OTHER axis land exactly on the canvas
   size with nothing to crop. But `--bg-crop-top`/`--bg-crop-bottom` is a
   *separate*, manual crop on the vertical axis, applied *before*
   `cover_resize` ever runs (intentionally — lesson #2 — to discard a
   cluttered ceiling/floor band from the source rather than just cropping
   the final canvas). If the photo is landscape-oriented behind a portrait
   canvas (the common case here), `cover_resize` crops the *horizontal*
   axis — so the net result is cropped top AND both sides, not one axis,
   even though each step is individually "correct." This shipped unnoticed
   for an entire workflow (`stylish_home_gym.md` had `bg_crop_top: 0.3`)
   because every automated check in this pipeline (`verify_quality.py`,
   `edge_brightness_halo`, the upscale-need warning) checks pixel-level
   properties of the *already-cropped* result — none of them check "how
   many sides got cropped relative to the original," because that requires
   comparing against the *uncropped source*, a comparison none of them make.
   Caught only by a human looking at the actual rendered frame next to the
   original photo. Two fixes: (a) `replace_background.py` now warns
   specifically when this combination occurs (compares which axis the
   manual crop touches against which axis `cover_resize`'s own math will
   crop); (b) before trusting any `--bg-crop-top`/`--bg-crop-bottom` value,
   render a background-only canvas preview at the actual target resolution
   (crop+resize the image alone, no video/matting involved — see
   `try_anchors.py`-style snippets used during this session, or just call
   `cover_resize` directly) and *look at it* — don't assume a crop value
   that was right for one background photo is right for another, and don't
   assume "the math is internally consistent" means "the result looks
   right." This is also why the AI-upscale decision in lesson #14 was wrong
   for a full day: the wrong `bg_crop_top` made the upscale math (correctly)
   compute a worse number for the wrong premise.

## `replace_background.py` — key flags

| flag | meaning |
|---|---|
| `--content-crop W:H:X:Y` | crop input video before processing (strip pillarbox) |
| `--bg-crop-top/bottom F` | discard a fraction of the *source* background image (remove clutter) |
| `--bg-anchor-x/y F` | which part of the cover-cropped canvas to keep (0=top/left .. 1=bottom/right) |
| `--bg-blur F` | Gaussian sigma on background, applied uniformly (see lesson #9 on why not a depth ramp) |
| `--relight-strength F` | 0-1, how much to color-match subject to new background's tone |
| `--relight-luma-weight F` | keep low (~0.25); fraction of relight applied to brightness vs color |
| `--fg-sharpen F` | unsharp strength on subject, applied only in matte interior |
| `--edge-despill F` | neutralize tint in semi-transparent boundary pixels (fixes hair fringe) |
| `--edge-feather N` | px of extra blur on the alpha matte itself |
| `--diagnostics-out PATH` | write JSON with sharpness/brightness/edge stats for the verifier |

Run `python3 scripts/replace_background.py --help` for the full list.

## `verify_quality.py` — standalone use

Can be run on *any* background-replacement output, not just this pipeline's,
since it works by diffing frames against the known background image (doesn't
need the alpha matte):

```
python3 scripts/verify_quality.py --video final.mp4 --background bg.jpg \
  --source-video original.mp4 --report report.json
```

Exit code 0 = passed, 1 = at least one check failed. Checks: `edge_fringe`,
`edge_brightness_halo`, `sharpness_match`, `brightness`, `duration_match`.
Each failing check comes with a `suggestion` string naming which flag to
adjust.

**`edge_brightness_halo`** (added after lesson #13) specifically catches a
bright/pale rim confined to one part of the silhouette boundary (e.g. one
limb), which `edge_fringe`'s whole-boundary average dilutes into nothing.
Unlike `edge_fringe`, it takes the **worst sampled frame**, not the mean
across frames, and only flags a boundary pixel as haloed if it's brighter
than *both* the subject's local interior tone AND the new background's local
brightness at that spot — i.e. genuine overshoot past the legitimate alpha-
blend range, not just "dark hair against a bright wall" (which is correct,
expected blending, not a defect). Tunable via `--halo-brightness-delta`
(default 18, on a 0-255 V scale) and `--halo-fraction-max` (default 0.12).

This check is necessarily approximate here, since `verify_quality.py` only
has the flat output video (no real alpha) and has to re-derive an approximate
subject mask by diffing against the background image — noisy enough that it
failed to distinguish a known-buggy render from its fix in testing. **Prefer
`replace_background.py --diagnostics-out`'s `edge_brightness_halo_worst_frame_fraction`**
(computed from the *real* alpha matte during rendering, far more accurate)
when available — `pipeline_agent.py` already does this automatically,
overriding `verify_quality.py`'s estimate with the diagnostics one whenever
both exist.

## Test-clip / candidate-review tooling

These are the supporting scripts for the "try several options, look at them,
pick one" loop, so that loop doesn't depend on one-off shell commands:

```
# 1. cut a short test clip (always re-encoded, see lesson #6 above)
python3 scripts/extract_clip.py --input full_video.mp4 --start 5 --duration 3 \
  --output clip3s.mp4

# 2. eyeball candidate background images before committing to one
python3 scripts/contact_sheet.py --images-dir "/path/to/background photos" \
  --output contact_sheet.jpg

# 3. render the clip against a few candidates (loop pipeline_agent.py per background)

# 4. compare renders side by side at the same frame/timestamp
python3 scripts/compare_renders.py \
  --video A:out_candidate_a.mp4 --video B:out_candidate_b.mp4 \
  --frame 90 --output compare.jpg

# 5. zoom into the silhouette boundary on whichever looks best, to check for
#    a visible seam/halo/fringe that's easy to miss at full-frame size
python3 scripts/inspect_edge.py --video out_candidate_a.mp4 --frame 90 \
  --output edge_zoom.jpg
# or target a specific spot (x,y,w,h in source pixels), repeatable:
python3 scripts/inspect_edge.py --video out_candidate_a.mp4 --frame 90 \
  --region 190,0,180,220 --output edge_zoom_head.jpg

# 6. once a candidate is approved, copy its exact param set (read out of
#    pipeline_agent.py's own iter*.agent_work/iterations.json, last entry)
#    into a new workflows/<name>.md -- see "Workflows" below.
```

## Workflows

A **workflow** is a saved, named recipe — one or more **stages** chained
together, stored as a `workflows/*.md` file (YAML frontmatter + free-text
notes below it), run end to end by `run_workflow.py`. This is what makes a
confirmed look reusable without re-typing or re-discovering a long flag list,
and it's the same mechanism regardless of what kind of job it is:

```
python3 scripts/run_workflow.py --workflow workflows/gym_dark_moody.md \
  --input full_video.mp4 --output full_video_final.mp4
```

Stage types (see `workflows/workflow_template.md` for the full reference and
`scripts/run_workflow.py`'s `STAGE_RUNNERS` for the current list):

| stage | wraps | for |
|---|---|---|
| `bg_replace` | `pipeline_agent.py` (estimate→render→grade→verify→tune) | full background replacement, sharp or stylized-blur |
| `bg_remove` | `replace_background.py` directly, flat-color background | background removal only, no replacement photo |
| `enhance` | `enhance_grade.py` directly | grading only, no matting at all |
| `audio` | `process_audio.py` (video stream untouched) | audio-only changes, no video re-encode |
| `upscale` | `upscale_realesrgan.py` | AI upscale the VIDEO (slow, see warning below) |
| `upscale_background` | `prepare_background.py` (image only, no video touched) | AI-upscale a low-res background IMAGE before a later `bg_replace`/`bg_remove` uses it — see lesson #11 |

A workflow's `stages` list is exactly as long as the job needs — this is
what "scalable to every shape of job" means in practice:
```yaml
stages: [bg_replace]                  # background replacement only
stages: [bg_remove]                   # background removal only (flat color)
stages: [bg_remove, enhance]          # removal + grade, no replacement photo
stages: [enhance]                     # grade/denoise only, no bg work at all
stages: [audio]                       # audio only, no video work at all
stages: [bg_replace, audio]           # bg + grade, then an audio pass
stages: [upscale_background, bg_replace]  # low-res background image, AI-upscaled first
```

`upscale_background` is the optional-component case: it resolves an image
path and stores it under `params.name` (default `"background"`) instead of
touching the video at all; a later stage references it with
`background: "$<name>"`. Delete the stage and use a literal path instead and
nothing else in the workflow changes — see `workflows/basement_home_gym_upscaled.md`.

**Important:** a `bg_replace` stage's `params` carry BOTH
`replace_background.py` params AND `enhance_grade.py` params together in one
flat map — `pipeline_agent.py` grades every auto-tune iteration itself (the
verifier checks the *graded* output), so a separate trailing `enhance` stage
after a `bg_replace` stage would grade the video twice. Use a standalone
`enhance` stage only when there's no `bg_replace` stage in the workflow.

Override any single param at run time without editing the file:
```
--stage-param bg_replace.bg_blur=4.0   # targets the first stage of that type
--stage-param 0.bg_blur=4.0            # targets by 0-based position instead
```

Confirmed examples are checked in: `workflows/gym_dark_moody.md` (sharp
throughout) and `workflows/gym_modern_blurred.md` (deliberate background
blur), and `workflows/stylish_home_gym.md` / `workflows/stylish_home_gym_blurred.md`
(same sharp/blurred pair, different background — see README lesson #13 for
why their `edge_despill`/`edge_feather`/`bg_anchor_x` values look different
from a fresh `estimate_params.py` run) — all came from the same test-clip
review loop above, with `estimate: false` and every param pinned to what was
actually confirmed, rather than left to re-estimate on every run.
`workflows/bg_remove_green.md` and `workflows/audio_cleanup_only.md` show the
other job shapes.

## `pipeline_agent.py` — auto-tune loop

`adjust_params()` maps each failed check to a concrete parameter nudge
(e.g. edge fringe fail -> `edge_despill += 0.2`). It's a small, explicit
rule table, not a search — if you hit a failure mode it doesn't know how to
fix, add a rule there rather than expanding `verify_quality.py`'s scope.

Known-good starting params (already the defaults) came from tuning against a
1080x1920-content, 60fps gym talking-head video with an `IMG_0342`-style
real-photo gym background. For a different kind of subject/background,
expect iteration 1 to fail at least the sharpness check and self-correct —
that's normal, not a bug.
