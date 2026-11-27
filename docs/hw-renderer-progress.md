# Hardware renderer: progress notes

Design: `docs/hw-renderer-design.md` (section 6 lists where the code deviates). Code: `runtime/src/gs/gpu/hwr.cpp`, `hwr.hpp`,
`hwr.wgsl` (drawing program), `hwr_sync.wgsl` (target <-> memory compute programs, appended to gs.wgsl); hooks in `gpu.cpp`
(each behind `hwr::g_on` / `hwr::requested()`), sources and shader embedding in `CMakeLists.txt`.

Switch: `PS2_RENDERER=hw` (default exact). Debug: `PS2_HWR_LOG=1` (one line per frame + summary at exit), `PS2_HWR_CACHE_MB=<n>`,
`PS2_HWR_SKIP_TBP=<hex>` (shapes with that texture base not drawn: bisecting), `PS2_FRAME_GPU_MS=1` (both renderers: card time per frame,
mean every 100 frames), `PS2_VRAM_DUMP=<frame>,<file>` (both renderers: 4 MB memory at the end of that frame; hw writes targets back first).

Test material: `C:\Users\Ihor\CFR2Recomp\hwr\m1` (`run.sh <exe> <outdir> [last] [every]` = race drive 3425-4400 every 25;
`runst.sh <exe> <state> <outdir> <last> <every>`; `cmpall.sh <ref> <test>` framecmp of every frame; `crop.py` zoomed ref|test|diff crops;
`vram32.py` / `adiff.py` / `pngcrop.py` look at CT32 pictures in memory dumps; `cover.py` / `runs.py` read `PS2_TRACE_DRAWS` logs).
Baseline exe of the tree before hwr: `hwr\exact_before.exe`; its frames: `m1\ref` (race), `m1\cup_ref` (cup race 5020-5300 every 20).

## M1 status (2026-11-18): done on PC

* Race drive (40 frames 3425-4400): 40/40 framecmp PASS, PSNR 46.9-50.5 dB, >48 diff <= 0.01 %, bad blocks 0 (`m1\cmp_hw4.txt`).
  Cup race (15 frames 5020-5300): 15/15 PASS, PSNR >= 48.7 dB (`m1\cmp_cup.txt`). Residual differences: isolated edge pixels of thin
  geometry (rasterisation rules), +-1 rounding of As blends.
* Fallback shapes: 0 in race and cup race. Texture cache hit rate 99.7 % (race), 99.8 % (cup); ~1.7 decodes per frame.
* Exact renderer: `PS2_RENDERER` unset and `=exact` frames byte-identical to `exact_before.exe` (race 40/40, cup 15/15).
* Android lib (`build/android-gpu`) builds.

Per race frame (PS2_HWR_LOG summary): ~11 000 shapes, 626 draws, 327 render passes, 569 pipeline switches, 412 bind-group switches,
29 self copies (glow ladder), 20 pipelines in all, processor time in hwr::shape ~1.3 ms. Cup race: 21 000 shapes, 1112 draws,
591 passes, 1014 pipeline switches.

Card time per frame (PS2_FRAME_GPU_MS, RTX 4050 laptop, frames 4000-4400, wall time submit -> done): Direct3D 12: exact 6.3-6.8 ms,
hw 8.2-9.0 ms; Vulkan: exact 5.1-5.9 ms, hw 11-12.6 ms. The hw path is dominated by the ~330 render passes per frame (load/store of whole
512x512 attachments per pass, barriers), which is what M2 removes.

## M2 status (2026-11-18): done on PC (phone not measured yet)

What M2 does (design 3.3; deviations in design section 6, "M2"):

* **Surfaces and renaming.** A target's picture lives in a surface (pooled RGBA8 / Depth32Float textures). The game clears with rows of
  16 untextured 32-pixel strips; a run of such strips that covers every page the target holds is taken back and becomes a *clear*: the
  target gets a new surface whose first render pass loads with `loadOp clear`. A run of opaque textured strips over the whole target (the
  F800 -> F0 display copy) gives a new surface too (no load). F800, Z1400 (twice: main and HUD), the shadow buffers and every car's 0x3FA0
  blur get a new surface per frame, so nothing waits for readers of the old picture.
* **Frame graph.** Every op of the submit (render-pass nodes, decoded-texture copies, self copies, memory loads/stores) gets read/write
  edges from the resources it touches. Loads/stores of graphics memory stay at their place among gpu.cpp's compute commands; everything else
  goes as late as it can (normally after all compute work of the frame = the "prologue"), ordered by a list schedule that keeps drawing into
  one colour surface while it can and, when it must switch, first takes a surface whose remaining nodes can all be drawn now (shadow, blur)
  before one that would break again (the main picture). Consecutive nodes on one surface are one render pass; their draws are re-packed into
  one vertex buffer in pass order and neighbours with equal state become one draw call.
* **Depth.** A node whose depth surface nothing uses afterwards and whose draws do not test depth drops its depth attachment (the glow
  composite then shares the HUD pass; the display copy's dead depth writes cost nothing). `storeOp discard` for a surface no target holds
  after its last use in the submit (the main pass's depth); depth that no draw reads and that is discarded is not loaded.
* **Self copies by region.** A draw that reads its own target reads a copy made before it, and the copy is made again only when something
  was drawn into the rectangle the draw reads since the last copy (texel rectangle from the FST coordinates, filter and wrap mode). The glow
  ladder needs 3 copies instead of 29, exactly as before (pictures bit-identical to M1).
* **Uploads in hw mode (gpu.cpp `emit_pre_hwr`).** The game uploads each texture with its palette and loads the palette right after; with
  page-wide ranges that made ~124 upload + ~42 palette dispatches per race frame, each a tiny dispatch with a barrier (about 2 ms of the
  PC frame). In hw mode the pending work is grouped with the exact blocks each op touches (`hwr::pre_blocks`): non-conflicting uploads go
  before the palette loads queued ahead of them. Now 5.5 upload and 9.8 palette commands per race frame. Palette commands are cut to 16
  loads (one workgroup): with more, gs.wgsl `load_clut`'s chunks in other workgroups gave different palettes in two runs of the same build
  (a race in that path; the exact renderer's runs of consecutive palette loads are short, but other games might reach it).
* **Positions.** hwr asks gpu.cpp for a stream position only for memory loads/stores (asking ends gpu.cpp's open run of uploads).
* **Pipelines.** The 20 pipelines of the race and the cup race are built in (`kBuiltinKeys`) and made at start (171 ms on the PC); new keys
  are appended to `PS2_HWR_PIPELINES=<file>` or `games/<name>/hw/hwr_pipelines.txt` (beside PS2_VU_RECORD) and made at the next start.
  Without dual-source blending the dual keys are converted to the source-alpha form.

New switches: `PS2_HWR_ORDER=stream` (M1's order, no renaming: for comparing), `PS2_HWR_DUMP=<frame>` (that frame's clear candidates and
schedule), `PS2_HWR_TIME=1` (card time of each render pass by timestamp queries, mean per 100 frames), `PS2_HWR_NOPASS=1|draws` (measuring
aid: no render passes / passes without draws), `PS2_HWR_PIPELINES=<file>`.

Checks (exe `hwr\m2\ps2run_m2.exe`; frames and logs in `C:\Users\Ihor\CFR2Recomp\hwr\m2`):

* Race drive 40 frames 3425-4400: 40/40 PASS vs exact (`m2\cmp_hw7.txt`, PSNR >= 46.9 dB), all 40 bit-identical to M1's frames.
  Cup race 15 frames 5020-5300: 15/15 PASS (`m2\cmp_cup7.txt`, PSNR >= 48.7 dB), bit-identical to M1. Fallback shapes 0. Texture cache hit
  rate 99.7 % / 99.8 %. `PS2_HWR_ORDER=stream` gives M1's frames too.
* Exact renderer (PS2_RENDERER unset): race 40/40 and cup 15/15 frames byte-identical to `hwr\exact_before.exe`'s.
* `build/native-check` and `build/android-gpu` build.

Per frame, steady race driving (frame 3700) / mean of the 1000-frame race run / cup race steady (mean of 300 frames):

| | M1 race | M2 race steady | M2 race mean | M2 cup steady (mean) |
|---|---|---|---|---|
| render passes | 327 | 9 | 10.4 | 17 (19.4) |
| draw calls | 626 | 240-300 | 316 | (569) |
| pipeline switches | 569 | 190-250 | 255 | (456) |
| self copies | 29 | 3 (1.1 MB) | 3 | 3 |
| attachment loads / stores | ~330 / ~330 whole targets | 5 (3.0 MB) / 10 (6.25 MB) | 5.8 / 10.8 | (8.5 / 22: 6.2 / 10.5 MB) |
| upload + palette commands | 124 + 42 | 5.5 + 9.8 | | (11.5 + 14.6) |
| hwr processor time | 1.3 ms | 1.27 ms (schedule 0.2 ms) | | 2.3 ms (0.4) |

Race frame schedule (all after the frame's compute work): F0 display copy (no load, 1 MB store) | shadow 64x512 (clear, 128 KB) | blur
0x3FA0 (clear, 128 KB) | F800 + Z: clears, all 3D, shadow, blended 3D (clear both, store colour 1 MB, depth discarded) | glow 256x512 x4
(load + store 512 KB each, 3 copies between) | F800: glow composite + HUD with its new depth (load colour, clear depth, store 1 MB + 1 MB).
The cup race adds 3 shadow buffers and 3 blur versions (8 small passes instead of 2); the main picture is one pass.

Card time per frame (PS2_FRAME_GPU_MS=1, RTX 4050 laptop, frames 4000-4400, means of 100 frames; logs `m2\time2_*.log`):

| | exact | M1 hw | M2 hw |
|---|---|---|---|
| Vulkan | 5.8-6.9 ms | 11-12.6 ms | **4.3-4.8 ms** |
| Direct3D 12 | 6.3-7.1 ms | 8.2-9.0 ms | **5.0-5.3 ms** |

PS2_HWR_TIME=1 (Vulkan): all render passes 1.8 ms per frame (main pass 0.6-0.7, glow composite + HUD 0.45-0.5, display copy 0.23, glow
0.3, shadow and blur 0.07). The rest of the hw frame is the shared compute work (uploads, palette loads, ~7 decodes).

Tile-based GPU (Mali): each render pass loads its attachments unless cleared and stores them unless discarded. Race frame: 9 passes,
3.0 MB loaded + 6.25 MB stored + 2.25 MB of self copies + 1.2 MB vertices + 0.5 MB display write-back ~ 13 MB per frame (~0.65 GB/s at
50 fps); M1's ~327 passes loaded and stored whole 512x512 targets each time (several hundred MB per frame).

## Known gaps / next steps

1. M2 left: shadow atlas (cup race: 8 small 64x512 passes; an atlas needs texel offsets in the shader's target reads, or copies back, which
   cost more bytes than they save), sorting draws by pipeline inside a pass (not done: the main pass mixes GEQUAL depth, blending and
   two-draw alpha tests, and pipeline switches cost little next to passes), depth discard at the end of the frame (the HUD depth, 1 MB, is
   stored because the next frame is not known yet), present straight from the target (kept: the write-back is one 512x256 dispatch and the
   display kernel's field/deflicker path stays exact).
2. First phone run with M2: `android-spike/CrazyFrogRacer2.apk` is built (2026-11-18) with PS2_RENDERER=hw as the app default (SpikeActivity's
   Os.setenv block; `PS2_RENDERER=exact` in files/env.txt switches back) but not installed (no phone on adb then):
   `MSYS_NO_PATHCONV=1 adb install -r C:/Users/Ihor/CFR2Recomp/android-spike/CrazyFrogRacer2.apk`. hw runs at PS2_SCALE 1 only (the app's scale
   setting above 1 falls back to exact). Useful there: PS2_HWR_LOG=1, PS2_HWR_TIME=1, PS2_SPEED_LOG=1, PS2_PROF=1.
3. Fallback to the painter for unsupported states is not implemented: such shapes are counted ("FALLBACK <reason>" in the log) and skipped.
   Needs: store involved targets, let gpu::draw's batch path draw, reload (colour: hwr_load; depth: a load draw).
4. Depth targets are not loaded from memory when created (start at 0). Save state in hw mode: `sync_all` writes colour and depth back
   (untested); state load drops all targets. Untested: save/load while in hw mode, menus, minigames, FMV, PS2_SCALE>1 (hw is refused
   at scale > 1, exact draws).
5. Dual-source blending is required for As blends whose alpha is written (no two-draw fallback yet, reason "no dual-source"); M3.
6. A texture inside a target at another base or width (reason "texture in a target (other layout)") is not supported; not seen in races.
7. Region clamp + mipmaps samples level 0; As > 0x80 saturates at 1.0; colour wrap (COLCLAMP=0) ignored. All counted as approximations.
8. Processor cost: per-shape work (triangle expansion, page ownership, SAT test for two-draw runs) ~1.3 ms per race frame on PC.

## Scale > 1 (2026-11-18)

hw now runs at PS2_SCALE 2-4 (before, scale > 1 fell back to the exact renderer: ~20x slower on the phone). Surfaces are made at
scale x size; hwr's bookkeeping (targets, rectangles, clears, self-copy regions) stays in GS pixels and only the card sizes are scaled
(scissor, self copies, sync dispatches). hwr_sync.wgsl runs one thread per sample (graphics memory keeps SCALE^2 samples per word,
k = sub-row * SCALE + sub-column; the sync module gets the SCALE override). A target read as a texture (TEX=2) reads its own samples:
draw word c.w = scale for target sources, `wrap_scaled` in hwr.wgsl wraps the GS texel and keeps the sample inside it.
Checks: 1x race 40/40 PASS vs exact (Vulkan). 2x race 13 frames vs exact 2x: PSNR 28.6-29.4 dB, >48 diff ~1.2 % (FAIL by framecmp's
thresholds): the same picture, differences soft along edges (the exact renderer's tent-filtered copies and sprite snapping at scale are
not reproduced; moving the card samples to x + j/scale made it worse). Card time 2x (Vulkan, RTX 4050): hw 6.3-7.1 ms, exact 6.0-6.4 ms;
the gain is on tile GPUs (memory traffic ~4x of 1x: ~50 MB per frame instead of hundreds of MB per pass set).

## Processor work, threaded run like the phone (2026-11-18)

New tool: `PS2_PROF_PC=<file>` with PS2_PROF=1 (Windows): prof.cpp samples each profiled thread's instruction pointer every 1 ms
(SuspendThread / GetThreadContext, only outside wait labels) and the word at its stack pointer (the caller of a leaf such as memcpy), and
writes "thread module offset count" / "thread stack offset count" every 5 s. Symbolize with `llvm-nm --numeric-sort` of the exe (image base
0x140000000) and the export table of system DLLs (`llvm-objdump -p`); helper `hwr\s2\sym.py`. Use PS2_THREADS=1 so that the headless run
has the phone's threads (emulation, vu1, drawing).

Found and fixed (race, 2x, hw, Vulkan):
* gif.cpp `release`: the PATH3 data held under MSKPATH3 was erased from the front after every released packet (the game lets streamed
  textures out one packet at a time): memmove of the whole remaining buffer each time, ~57 % of the vu1 thread's working samples. Now a
  read offset (`Held::head`), compacted when the released part is the larger one. vu1 thread busy 67-75 % -> ~41 %. Frames bit-identical.
* hwr `shape`: two steady_clock reads per shape for PS2_HWR_LOG's cpu time, also when the log is off (~8 % of the drawing thread). Now only
  with the log.
* hwr `texture_key`: every lookup hashed all blocks of the texture (and mip levels). Now memoized per setup words + palette id, valid while
  no page the texture covers changed (page generations bumped wherever a block identity or stale flag changes; whole-memory changes bump
  g_mem_all). resolve_texture 14.6 % -> 4.9 % of the drawing thread; hwr processor time per race frame 1.27 -> 0.96 ms. Race 1x frames
  bit-identical; cup 15/15 PASS.
Remaining drawing-thread costs: hwr::shape ~26 % (vertex building), gs write_packed_run ~14 %, uploads (hash + staging copy) ~16 %.
Emulation thread: EE code + IOP. vu1: compiled VU1 programs ~70 % of its work.

60 Hz: the disc is PAL (50 Hz; 50 fps is full speed). `--ntsc` runs the machine at 60 Hz, but the game keeps its 512-line PAL picture
and the NTSC display shows ~425 lines (the bottom HUD is cut), and whether the game's logic is per frame (then 20 % too fast) is not
settled. Not offered.

## Phone at high resolution: the picture shown straight from the target (2026-11-18)

First phone measurements of the hw renderer (Xiaomi 2311DRK48G, Mali-G615; PS2_PROF / PS2_SPEED_LOG / PS2_FRAME_GPU_MS, logs in
`C:\Users\Ihor\CFR2Recomp\perf\s1`, `s2`): at 40 % (targets 0.94x, memory 1x) a steady 50 fps with the processor threads idle in the
limiter ~75 % of the time; at 80 % (targets 1.69x, memory 2x) 36-44 fps everywhere, menus included, the window thread waiting in present
70-90 % and the card busy 25-30 ms per frame even for menu frames of 2 draws. Not the drawing: fixed per-frame work. Mali's render-pass
timestamps read ~0 (not usable there).

Changes (all PS2_RENDERER=hw unless noted):
* **Present pass reads a texture** (blit.wgsl, all renderers): `prep_mem` / `prep_tex` write the field (or the blend of the two latest
  fields) once per presented frame into an RGBA8 texture; the present pass reads it with textureLoad and the card's bilinear filter. Before,
  every screen pixel (~2.7 M) read the fields from a storage buffer, both fields, up to ~70 reads for the default sharp_edges look.
* **Display straight from the target** (`hwr::display_target` / `hwr::show`, op kOpShow): the displayed rectangle of the target is copied
  (card texels) into a field texture; no store to memory and no display kernel. Overlay circuit, live mode or no matching target: the
  memory path as before. `PS2_HWR_PRESENT=memory` turns it off. At 1x the prepared picture is identical to the memory path's (row k = memory
  row 2k: 0 of 391 680 values differ, race frame 3650).
* **Surfaces without storage usage**: loads (memory -> new target) go through a staging texture and a copy, so tile GPUs can compress the
  render targets (Mali turns AFBC off for storage textures). Race 3400-4400 headless: frames and end state byte-identical.
* App: PS2_SCALE (graphics memory) is 1 always; the resolution setting only sets PS2_HWR_SCALE.
* Debug: `PS2_PRESENT_DUMP=<frame>,<file>` writes that frame's prepared picture (PPM).

## Widescreen at the screen's shape and the screen's own resolution (2026-11-18)

* **Aspect knobs** (eerecomp `--aspect-knobs games/<name>/aspect.txt`, `pnach.py knobs <elf> <pnach>`, port.sh `--aspect-pnach FILE`): a
  widescreen patch whose lines all replace `lui` float constants becomes knobs: the translation loads `aspect_word(constant, power)`
  (generated.hpp), the constant times r^power with r = PS2_ASPECT / (4:3), the constant itself without PS2_ASPECT. CFR2: Arapapa's 16:9
  patch from PCSX2's repository (SLES-54549_31E4639D.pnach): zoom 0047F2C8 r^-1, vertical field of view 0046C7AC r^1, HUD scale 0046CCB0
  r^-1; full float precision (the patch keeps 16 bits). Without PS2_ASPECT race frames and end state byte-identical. The HUD stays in the
  4:3 middle (the patch's HUD fix scales it, it does not move it to the edges). The display layout uses 4:3 x r as the picture's shape.
* **Across and down apart** (`PS2_HWR_SCALE` across, `PS2_HWR_SCALE_Y` down; hwr card_x / card_y, sync and target-read shaders take
  both): CFR2 draws 512 x 256 fields shown on all lines, so a screen of W x H pixels needs W / 512 across and H / 256 down. 1x: race frames
  and end state byte-identical. 2712:1220 at 5.297 x 4.766: the prepared picture is 2712 x 1215.
* App: resolution in percent of the screen's own pixels (100 % = one texel per screen pixel), "Game view: Widescreen / 4:3" (default
  widescreen; PS2_ASPECT = the real screen size).

### Harry Potter and the Chamber of Secrets (SLES-51192): knobs found without a patch (2026-11-19)

No community patch was used: the camera was traced in the game's own code.
* `0x4E8EB0` is a virtual "aspect" getter that returns 4/3 (lui 0x3FAA + ori 0xAAAB); its one caller (0x1A75F0) only derives a vertical
  value from it: scaling it zooms the picture in (the vertical view shrinks), so it is not the knob.
* The 3D screen matrix (0x541740: 0.4583 across, -0.5 down at 4:3) is built at boot and on loads by 0x1DC940, called from 0x1BE3D8 with
  768.0 for both axes (one lui, so not a knob either). Its horizontal scale is one `div.s f20 = across / 768` at **0x1DCA74**: knob power -1.
  The vertical scale is left alone and the game's side clipping is 1 / that scale, so the world is drawn out to the new edges (the car and
  the gate right of the heads scene, the barn wall left). `PS2_WATCH=541740,40` from boot finds the writer.
* 3D HUD pieces (the button ring) go through the same matrix and keep their shape on screen; 2D pieces (subtitles, menus) are drawn in screen
  coordinates and are widened with the picture.
* Without PS2_ASPECT: Flipendo spam frames pixel-identical to the build before the knob.

Translator changes for it: a knob on an `ori` that follows the knob's `lui` into the same register scales the whole 32-bit constant (the lui
alone scaled only the high half and the ori ORed the original low bits back); a knob on a `div.s` scales its quotient (a camera scale that is
computed, not loaded). The phone translates the game on the device: `PS2_ASPECT_KNOBS=<file>` gives the knobs to every live translation, they
are part of the live stamp (libraries made with other knobs are dropped), and the widening reaches the libraries through `EeLiveHost::aspect_widen`.
Live path frame at 20:9 identical to the static build's. App: Settings > Display > Game view (Widescreen default / Original 4:3, restarts),
the knob table per serial in `Ps2Activity.aspectKnobs`.

## The screen's own resolution: what it costs and two cuts (2026-11-18)

Phone at 2712 x 1215 (5.297 x 4.766): ~20-22 fps, the card the limit everywhere (menus 20.0-20.4 fps). PC at the same scale (RTX 4050,
race 3400-3800): attachment loads ~290 MB and stores ~360 MB per frame, render passes 2.5-3.2 ms, the display copy (F800 -> F0, 16 strips
read with the scaled bilinear path: four texel loads through the GS wrap each) 0.64-0.82 ms, the glow ladder 4 x ~0.2 ms.
* **Target reads through the card's filter** (hwr.wgsl sample_loads): a bilinear read of a target at another scale, view 0 (its own
  colours), clamp (or region clamp over the whole texture) or repeat over a texture of the target's size: one textureSampleLevel with a
  bilinear clamp / repeat sampler (tex group bindings 2, 3) instead of four loads. Display copy 0.64-0.82 -> 0.32 ms. Colours may differ by 1.
* **Effect buffers at a smaller scale** (hwr `sx_of` / `sy_of`; `PS2_HWR_SMALL_SCALE`, default 2, 0 = off): targets narrower than 512 GS
  pixels (glow 256, blur and shadow 64) at most 2x across. Glow passes ~0.2 -> 0.03-0.1 ms; attachment traffic 32 MB loaded + 109 MB stored
  per frame. Render passes 2.0 ms, card 2.7 ms per frame. 1x race byte-identical (the cut applies only above 2x).

Second round (phone at 100 %: 20 fps everywhere, 80 %: 30-34 fps; `dumpsys thermalservice` during play: CPU/GPU 87 C, status severe,
skin 51 C: the card runs throttled, so card work is heat as much as time):
* **Targets as tall as their first draw's scissor** (rounded up to 32 rows; `PS2_HWR_FULL_HEIGHT=1` = 512 as before): every texel of a
  render pass is loaded and stored by a tile GPU whether drawn or not. CFR2: display buffer 512x256, frame 512x384 (were 512x512). A draw
  below makes the target 512 rows (`grow`: colour copied into the new surface, depth not kept).
* **Frame rows capped at the screen's** (`PS2_HWR_ROWS`, the app gives screen rows x resolution): a frame-wide target whose first scissor
  ends at row r gets at most ROWS / r texels per GS row down. CFR2's 384-row frame (squeezed into the 256-row field) at 3.18 instead of 4.77.
* **Depth buffer shared by targets of two scales** (frame and display copy share Z 1400): a draw that only writes depth into a buffer held at
  another scale drops its depth writes (approx "depth write dropped"); one that tests depth gets its own buffer. Before, the two evicted
  each other every frame.
* **Dead loads**: a load into a surface that nothing uses and no target holds after it (made over current memory, then cleared) is not
  scheduled (approx "load not done").
* **Present without a prepare pass**: a target shown straight is not blended with the field before (`PS2_FIELD_BLEND=1` blends), the present
  pass reads the field copy itself; a picture of the screen's size is read once (no sharp / edge-smoothing passes at 1:1).
Race at 2712 x 1215 (PC): attachments 18 MB loaded + 56 MB stored per frame (290 + 360 before the first round), card 2.0-2.3 ms
(3.0). 1x race frames and end state byte-identical.

## Measured on the phone (2026-11-18): where the card time goes

`PS2_GPU_SWEEP=1` (gpu.cpp; with PS2_FRAME_GPU_MS's wait, now also after the present submit): every 8 s another part is left out, one
line per part. Race (race.state auto-loaded by the app), 80 % (4.24 x 3.81, 2170 x 976 on a 2712 x 1220 screen):
| | all | no draws | no render passes | no present picture |
|---|---|---|---|---|
| before | 32-33 ms | 19.3 | 19.0 | 18.4-18.9 |
| plain present (below) | 19-19.6 ms | 8.5 | 7.8 | 18.2-19.2 |
The present pass cost as much as the game's drawing (~13 ms): at a picture smaller than the screen the sharp_edges look read 20-40
pixels per screen pixel, plus the prepare pass for the field blend. Now a picture with 60 % to 106 % of the screen's pixels each way is
read once per field (bilinear) and the two latest fields are blended in the present pass itself (blend word 2): the game moves every
other field by half a line (measured: 2 rows at 4x), which jumped when shown unblended.
Phone fps (race, no measuring): 80 % 50.0 steady (emulation thread 63-69 % in the limiter); 100 % (2712 x 1215) 45-47 (was 20-22).
* **Draw parameters as flat varyings** (hwr.wgsl): the vertex stage reads the draw's 13 parameter words once per corner and passes them
  flat; the fragment stage no longer reads the storage buffer at every pixel (twice). 1x race byte-identical.
  Phone sweep at 100 % (2712 x 1215), race: all 22-23 ms, no draws 8.5, no render passes 8.0, no present picture 21.4-22.3, no compute
  (`E`, new) 20.2-21.1. Phone fps without measuring: **50.0 steady at 100 %** (emulation thread ~60 % in the limiter).

## Movies (2026-11-18)

Phone, intro movie at 100 %: 16-29 fps; sweep with 2 s parts (`PS2_GPU_SWEEP_S=2`): card only ~12 ms per frame, the window thread busy in
the driver (present 41-61 %, submit 18-38 %). PC count: 16 825 submits in 420 movie frames (40 a frame; a race: 6). Cause: gpu.cpp
queue_upload's content cache submits when its ring wraps while the pending submit reads the oldest entry; a movie's picture comes as
hundreds of small uploads a frame, many alike (seen twice -> cached), so the ring wrapped ~40 times a frame. Now at most one such submit a
frame (`g_ring_submits`, reset in flush); later uploads take the per-submit buffer. Movie run: 791 submits (2 a frame); movie frames
(power-on 0-1200 every 50) and the 1x race byte-identical to the build before.
Also: hwr texture cache entries unused for 50 frames are recycled by size (texture, view and groups kept) instead of making a texture per
movie picture. Debug: `[gpu] submits by site` at exit.
App version 1.1 (AndroidManifest versionName, versionCode 2), shown under the title in the settings.

## M6 (Harry Potter 2, weak tile GPU): DATE (2026-11-27)

* **Destination alpha test** (was "FALLBACK DATE": ~2 600 shapes a frame of the heads scene not drawn). A DATE draw reads a copy of its
  own target (the self-copy mechanism, alpha only) bound as group 2 (hwr.wgsl `dst_copy`, override `DATE`, flag F_DATM) and discards
  where the copy's alpha top bit differs from DATM. Every pipeline layout has group 2 (a 1x1 texture is bound when no draw needs it).
* HP2's DATE shapes: projected character shadows (texture: the 128x... shadow target at 0x3B80, CT32) drawn into the 704x576 CT16S buffer
  at 0x2F20, DATM 0, no blending, no depth write, alpha written: the first shape that reaches a pixel sets its A bit and blocks the
  shapes after it. With a copy per shape that is ~1 550 copies and render passes a frame. Instead such draws (DATE, alpha written, no
  blend, no depth write, one draw) are sent to the card in reverse order (`DrawRec::rev`, reversed in schedule()): the first shape is
  drawn last and wins its pixels, and all shapes of the run read one copy. Assumption: the alpha written fails DATM (exact for these
  shadows). Other DATE draws keep their written triangles (`Target::date_tris`); a DATE shape that overlaps none of them (separating-axis
  test) reads the same copy. A copy made for a DATE draw also takes the whole written rectangles it meets, so they are clean afterwards.
* Check: the 0x2F20 buffer after frame 5700 (PS2_VRAM_DUMP) is identical with the reversed draws and with a copy per shape
  (PS2_HWR_DATE_REVERSE=0 PS2_HWR_DATE_TRIS=0). The presented frames 5600/5700/5800 are bit-identical with and without DATE: in this
  scene the composite of 0x2F20 into the frame does not show the shadow pixels (5 840 pixels of 0x2F20 differ). The exact renderer's
  0x2F20 has none of these shadow pixels at frame 5700 (its 0x3B80 shadow target has 3 860 fewer pixels with alpha 0x80 than hw's): an
  upstream difference of the shadow target, not of DATE.
* Cost (heads scene, per drawn frame): render passes 43 -> 62.6, attachment loads 27.6 -> 59.7 MB, stores 26.6 -> 58.7 MB, self copies
  17 -> 31.6 (each character's shadow run is a 0x2F20 pass with the frame's depth, between the passes of its shadow target).
* Switches: `PS2_HWR_DATE=0` (DATE shapes not drawn, as before), `PS2_HWR_DATE_REVERSE=0` (no reversed draws), `PS2_HWR_DATE_TRIS=0`
  (any DATE alpha write under a shape's rectangle makes a new copy).

## M6: texture decodes and render passes (2026-11-27)

* **Texture decodes, measured** (heads scene, PS2_HWR_LOG=4, per drawn frame): 78 misses, 934 k texels, 228 mip levels. 58 of them
  (690 k texels, 174 levels) are FROM-TARGET: mip chains the game renders into the frame's corner (address 0, T8/T4) and reads back,
  written back to memory and decoded each frame; 10 more (118 k texels) read pages a target drew into (STALE). Only 10 textures (126 k
  texels, 27 levels) are uploaded data. Card cost on the PC (PS2_HWR_TIME): the decode compute pass ~0.36 ms (with the uploads and
  palette loads), ~105 level copies ~0.85 ms (timed with two empty compute passes each, so inflated). A decode on the processor is not done:
  hw mode keeps no processor copy of graphics memory or of the palettes (gs.cpp g_memory is only kept for the processor rasterizer), and
  88 % of the texels come from pictures that only the card holds, so it could serve 13 % of the work while adding a memory mirror and the
  decode to the drawing thread, which is the busy one. PS2_HWR_DIRECT_MIPMAP=1 (view 4 for mipmapped reads) changes nothing here (same
  counts, same frames): the corner's levels are not all held by the frame target.
* **Dirty spans** (`PS2_HWR_DIRTY_GRID`, default on): what was drawn into a target since its copy is kept per pair of rows as a column
  span (DirtyGrid). The rectangle lists merged HP2's 64x2 shift strips (the 704x576 buffer at 0x2F20 moved up 2 rows, reading the rows
  below) into one box per column once they were 32, so each 64-pixel column made a copy and a render pass of the buffer. A copy made for a
  stale read also takes everything written since the last copy.
* **Read snapshots** (`PS2_HWR_READ_SNAP`, default on): a draw into a target at least 4 times larger than the target it reads (DATE
  shadows into 0x2F20 reading each character's 128x128 shadow target at 0x3B80) reads a copy of the small target made at that point
  (64-128 KB), so the large target's draws need not come before the small target's later draws.
* **Schedule** (`PS2_HWR_SCHED_COPIES`, default on): when choosing the next render pass, copies whose inputs are reachable count as
  reachable, so a small target that reads copies of itself counts as complete and is drawn before a large target's node that came first in
  the stream. The 12 shadow targets now come before the frame's and 0x2F20's late passes; 0x2F20's DATE runs are 2 passes instead of 5.
* DATE overlap test: the triangle lists keep their bounding box (TriList), limit 4096 triangles (was 256, which made a copy at the start
  of every character's run).
* Numbers (heads scene, per drawn frame, PS2_HWR_LOG): DATE skipped (as before): 43 passes, 27.6 / 26.6 MB loaded / stored, 17 self
  copies (all switches above change nothing there). DATE drawn, old structure (PS2_HWR_DIRTY_GRID=0 PS2_HWR_READ_SNAP=0
  PS2_HWR_SCHED_COPIES=0): 60-63 passes, 54-60 / 53-59 MB, 29-32 copies. DATE drawn, new: 49 passes, 35.5 / 34.5 MB, 23 copies
  (4.2 MB). Frames 5600/5700/5800 bit-identical in every configuration.
* Left: the frame (0000, 704x576 with depth) is still cut into ~7 passes (~6 MB of attachment traffic each): three write-backs to memory
  for the corner's mip chains (anchored among the compute work) and the late 0x2F20 passes between; WebGPU's read-only depth is loaded
  and stored by wgpu anyway (wgpu-core maps it to Load + Store), so it saves nothing.
