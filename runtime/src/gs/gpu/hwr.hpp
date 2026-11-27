#pragma once
// The hardware renderer ("hwr", docs/hw-renderer-design.md, progress in docs/hw-renderer-progress.md): the GS's triangles and sprites drawn
// with the graphics card's rasterizer into render targets that stand for GS memory pages, textures decoded once into a content-keyed cache.
// PS2_RENDERER=hw turns it on (read once at start); the default (PS2_RENDERER unset or =exact) runs none of this code and compiles none of its shaders.
//
// gpu.cpp keeps the device, the queue, graphics memory and the compute path for uploads, palette loads and decodes; in hw mode gpu::draw hands
// every shape to hwr::shape instead of the painter, and hwr's work (render passes, copies of decoded textures into the cache, write-backs of
// targets into graphics memory) goes into gpu.cpp's command stream at the position where it happened (encode, called from submit()).

#include "../gs.hpp"

#include <webgpu/webgpu.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ps2::gs::gpu::hwr {

// What hwr needs from gpu.cpp: shared objects and a few calls into it.
struct Context {
    WGPUDevice device;
    WGPUQueue queue;
    WGPUBuffer decoded;                // gpu.cpp's decoded-texture buffer (decode in gs.wgsl writes here; needs CopySrc in hw mode)
    std::uint64_t decoded_words;       // its size in words
    WGPUBindGroupLayout layout;        // gpu.cpp's group 0 (9 entries, binding 2 with a dynamic offset of the parameter record)
    WGPUBindGroup *group;              // gpu.cpp's current group 0 (rebuilt when its buffers grow; read at encode time)
    std::uint32_t param_stride;        // bytes per parameter record (the dynamic offset step)
    const char *gs_shader;             // gs.wgsl text; hwr_sync.wgsl is appended to it for the load/store compute programs
    const std::uint32_t *swizzle;      // the processor's copy of the address tables (byte offsets inside a page, as gs.wgsl's `tables`)
    bool dual;                         // the device has dual-source blending
    std::uint32_t (*position)();       // queued uploads and palette loads become commands; returns the number of compute commands so far
    std::uint32_t (*decode)(const std::uint32_t *setup, std::uint32_t out_word, std::uint32_t words);  // queue gs.wgsl decode; returns the position after it
    std::uint32_t (*params)(const std::uint32_t *words, std::uint32_t count);  // a parameter record for a compute program of hwr; returns its index
    void (*fmt_words)(std::uint32_t psm, std::uint32_t bp, std::uint32_t bw, std::uint32_t *out);  // gpu.cpp's format words of a setup
    void (*submit_now)();              // ends the current submit (all queued work goes to the card)
    std::uint32_t scale = 1;           // PS2_SCALE: graphics memory keeps scale^2 samples per word
    float target_scale = 0.0f;         // PS2_HWR_SCALE: card texels per GS pixel of the targets (any size, e.g. 1.35); 0 = scale
    bool gl = false;                   // OpenGL (ES): no storage buffers in vertex shaders (Mali); the fragment stage reads the draw records (hwr.wgsl vs_gl)
    std::size_t swizzle_size = 0;      // words in `swizzle`
    WGPUBuffer clut = nullptr;         // gpu.cpp's palettes (2048 slots of 512 words): a CT32 target read as T8H is looked up here by the draw (hwr.wgsl view 3)
    WGPUBuffer vram = nullptr;         // gpu.cpp's graphics memory (PS2_SCALE 1 only): mip levels a view-4 draw reads from memory (hwr.wgsl view4_at)
};

bool requested();  // PS2_RENDERER=hw
// PS2_GPU_SWEEP=1 (gpu.cpp): what is left out now to measure it: 0 nothing, 1 the render passes' draws, 2 the render passes, 3 the present pass's
// picture (a measuring aid: the picture is wrong in 1 to 3)
extern int g_sweep;
extern bool g_on;  // set by init when the renderer runs
bool init(const Context &ctx);

// gpu::draw in hw mode: one primitive (3 corners for triangles, 2 for sprites) with gpu.cpp's 40 setup words and 8 mipmap words.
// `serial` changes whenever the setup words change (gpu.cpp's setup cache); `mip_top` is the highest mip level a pixel can read.
void shape(const State &st, const std::uint32_t *setup, const std::uint32_t *mip, std::uint32_t mip_top, std::uint64_t serial, const Vertex *v, int count);
// an upload chunk queued by gpu.cpp (rec: dbp, dbw, dpsm, dsax, dsay, rrw, done, pixels as queue_upload's record) and its bytes
void upload(const std::uint32_t *rec, const std::uint8_t *bytes, std::size_t size);
// a palette load (after gpu.cpp queued it): what load_clut in gs.wgsl reads
void clut(std::uint32_t cbp, std::uint32_t cpsm, std::uint32_t csm, std::uint32_t csa, bool eight, std::uint32_t cbw, std::uint32_t cou, std::uint32_t cov);
// the 256-byte blocks of graphics memory that gpu.cpp's queued upload writes (clut false; rec as queue_upload's record) or palette load reads
// (clut true; rec as load_clut's record): gpu.cpp groups its pending uploads and palette loads with them in hw mode (emit_pre_hwr)
void pre_blocks(bool clut, const std::uint32_t *rec, std::vector<std::uint32_t> &out);
// a local copy (TRXDIR 2) overwrote this rectangle of memory
void local_copy(std::uint32_t dbp, std::uint32_t dbw, std::uint32_t dpsm, int x0, int y0, int x1, int y1);
// the display reads this rectangle of memory at this point of the stream: the target holding it is written back first
void before_display(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::uint32_t x, std::uint32_t y, std::uint32_t w, std::uint32_t h);
// The display shown straight from a target (no write-back to memory, no display kernel): display_target finds the colour target that holds
// this rectangle of memory (the first pixel's page, same base and width) and gives the size of its picture in card texels; show then records
// the copy of that picture into `dst` (at 0, 0), after the draws that made it. PS2_HWR_PRESENT=memory turns it off (display_target false).
bool display_target(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::uint32_t x, std::uint32_t y, std::uint32_t w, std::uint32_t h, std::uint32_t &card_w, std::uint32_t &card_h);
void show(WGPUTexture dst);
// Every present asks first: true when that buffer missed the draws of a frame the governor skipped (ps2/governor.hpp, hwr.cpp FrameSkip):
// nothing new is shown, the screen keeps the last picture.
bool hold_picture(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::uint32_t x, std::uint32_t y);
// the game downloads these bytes of graphics memory: the targets that hold them are never skipped (FrameSkip)
void game_read(std::uint32_t offset, std::uint32_t bytes);
// all targets go back to graphics memory (save states, downloads)
void sync_all();
// graphics memory was replaced (state load): every target and the identities are dropped
void state_loaded();

// submit(): hwr's work goes into the same command encoder as gpu.cpp's compute commands, in stream order
bool pending();
bool before(std::uint32_t command);   // hwr work waits that goes before compute command `command`
void prepare(bool defer_writes = false);  // the submit's vertex and parameter data go to the card (one write each; defer_writes: handoff(true) takes them for the job)
WGPUComputePassEncoder encode(WGPUCommandEncoder enc, std::uint32_t upto, WGPUComputePassEncoder pass);
void submitted();                      // the submit is done: per-submit state is cleared, textures dropped in it are released
void frame_end();                      // vblank: counters (PS2_HWR_LOG=1 prints one line per frame)
// With the submit thread (gpu.cpp post): what this submit releases after the commands are submitted (the textures of dropped cache entries
// and their groups) goes with the job instead of being released at once, when the commands that use them have not been submitted yet.
struct Handoff;
Handoff *handoff(bool take_writes);    // the releases of this submit, taken out of hwr (call after encode, before the hand-over); take_writes: and the vertex
                                       // and parameter data of a prepare(true), to be written by the job
void handoff_write(Handoff *h);        // the submit thread, before the job's commands are finished: the taken vertex and parameter writes
void handoff_release(Handoff *h);      // the submit thread, after wgpuQueueSubmit
// PS2_GPU_PROFILE=1: gpu.cpp's compute passes of a renderer frame get timestamps too. prof_compute fills `tw` for the pass about to begin
// and returns its piece (-1: not profiling, or the frame's slots are used up); prof_name adds a program's name to that piece.
int prof_compute(WGPUPassTimestampWrites &tw);
void prof_name(int piece, const char *program);
void stats();                          // end of the run: totals

}
