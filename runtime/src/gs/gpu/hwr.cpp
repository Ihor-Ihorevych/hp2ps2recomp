// The hardware renderer (M1 and M2 of docs/hw-renderer-design.md; what is done and what is left: docs/hw-renderer-progress.md).
// PS2_RENDERER=hw: the GS's triangles and sprites are drawn by the graphics card's rasterizer into render targets, textures are decoded once
// (gs.wgsl decode, palette applied) into a cache keyed by their content, and the targets stand for the memory pages they were drawn into.
// M2: a target's picture lives in a surface that a full clear replaces (renaming); the submit's passes and copies form a graph that
// schedule() reorders into few render passes (about 9 per race frame instead of ~330), after all of the frame's compute work.
//
// Debug switches: PS2_HWR_LOG=1 (one line per frame: shapes, draws, passes, pipeline switches, texture cache, fallbacks by reason),
// PS2_HWR_CACHE_MB=<n> (texture cache budget, default 256), PS2_HWR_ORDER=stream (M1's order), PS2_HWR_DUMP=<frame> (that frame's
// schedule), PS2_HWR_TIME=1 (card time per render pass), PS2_HWR_NOPASS=1|draws (measuring aid), PS2_HWR_PIPELINES=<file> (pipeline list).
#include "hwr.hpp"
#include "../../prov.hpp"
#include "../drawtime.hpp"

#ifndef __EMSCRIPTEN__

#include "ps2/os.hpp"
#include "ps2/prof.hpp"

#include <webgpu/wgpu.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

extern const char kHwrShader[];
extern const char kHwrSyncShader[];

namespace ps2::gs::gpu::hwr {

bool g_on = false;
int g_sweep = 0;

bool requested() {
    static const bool on = os::env("PS2_RENDERER") == "hw";
    return on;
}

namespace {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

Context C;
u32 g_s = 1;  // PS2_SCALE (C.scale): samples per GS pixel each way in graphics memory
// PS2_HWR_SCALE (C.target_scale): card texels per GS pixel of the targets (fractional allowed); hwr's bookkeeping stays in GS pixels and only
// card sizes, scissors and copies are scaled. The sync programs resample between targets and memory (hwr_sync.wgsl).
// across (x) and down (y) apart: a game that draws half-height fields (512 x 256 shown on 512 lines) fills a screen with about twice the
// texels per GS pixel down that it needs across (PS2_HWR_SCALE_Y; unset: the same as across)
float g_fx = 1.0f, g_fy = 1.0f;
// Targets narrower than the frame (effect buffers: glow, blur, shadow; the frame and the display are 512 or more) are drawn at a smaller
// scale, at most PS2_HWR_SMALL_SCALE across (default 2; 0 = the full scale): their pictures are blurred or small on screen, and at a
// phone's own resolution they cost as much card time as the frame. A target's scale depends on its width only, so a colour target and the
// depth target found for it (same size) always match.
constexpr u32 kSmallWidth = 512;
constexpr u32 kMaxRows = 576;  // the tallest target (a PAL frame); draws below it are cut (approximation "outside target")
float g_small = 1.0f;  // the factor of the small targets' scale to the full one (<= 1)
// PS2_HWR_ROWS: the screen's rows the picture fills (the app gives them). A frame-wide target whose first draw's scissor ends at row r is
// drawn at most rows / r card texels per GS row down: the game's 384-row frame that it squeezes into a 256-row field then has the screen's
// rows, not 1.5 times them (the rows past r still exist, at that scale). 0: no cap.
float g_rows = 0.0f;
// a target's card texels per GS pixel across and down (fixed for its life; its surfaces have it)
struct Sc {
    float x = 1.0f, y = 1.0f;
    bool operator==(const Sc &o) const { return x == o.x && y == o.y; }
};
inline Sc base_scale(u32 tw) { return tw < kSmallWidth ? Sc{g_fx * g_small, g_fy * g_small} : Sc{g_fx, g_fy}; }
inline u32 fbits(float f) {
    u32 b;
    std::memcpy(&b, &f, 4);
    return b;
}
// a GS length / position in card texels at scale `c`
inline u32 card_x(u32 v, const Sc &c) { return std::max<u32>(1u, static_cast<u32>(std::lround(static_cast<double>(v) * c.x))); }
inline u32 card_y(u32 v, const Sc &c) { return std::max<u32>(1u, static_cast<u32>(std::lround(static_cast<double>(v) * c.y))); }
inline u32 card_at_x(u32 v, const Sc &c) { return static_cast<u32>(std::lround(static_cast<double>(v) * c.x)); }
inline u32 card_at_y(u32 v, const Sc &c) { return static_cast<u32>(std::lround(static_cast<double>(v) * c.y)); }
bool g_log = false;

WGPUStringView sv(const char *s) { return WGPUStringView{s, std::strlen(s)}; }

// ---------------------------------------------------------------- hashing
inline u64 mix(u64 h, u64 v) {
    h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
    h *= 0xBF58476D1CE4E5B9ull;
    return h ^ (h >> 31);
}

u64 hash_bytes(const u8 *p, std::size_t n) {
    u64 a = 0x243F6A8885A308D3ull ^ n, b = 0x13198A2E03707344ull, c = 0xA4093822299F31D0ull, d = 0x082EFA98EC4E6C89ull;
    std::size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        u64 w[4];
        std::memcpy(w, p + i, 32);
        a = (a ^ w[0]) * 0x9E3779B97F4A7C15ull;
        b = (b ^ w[1]) * 0xC2B2AE3D27D4EB4Full;
        c = (c ^ w[2]) * 0x165667B19E3779F9ull;
        d = (d ^ w[3]) * 0x27D4EB2F165667C5ull;
        a ^= a >> 29, b ^= b >> 31, c ^= c >> 27, d ^= d >> 33;
    }
    u64 h = mix(mix(mix(a, b), c), d);
    for (; i + 8 <= n; i += 8) {
        u64 w;
        std::memcpy(&w, p + i, 8);
        h = mix(h, w);
    }
    if (i < n) {
        u64 w = 0;
        std::memcpy(&w, p + i, n - i);
        h = mix(h, w ^ 0xFEEDull);
    }
    return h;
}

// ---------------------------------------------------------------- GS memory layout (block identities)
constexpr u32 kBlocks = 16384, kPages = 512;

bool is16(u32 psm) { return psm == 0x02 || psm == 0x0A || psm == 0x32 || psm == 0x3A; }

void page_dims(u32 psm, u32 &pw, u32 &ph) {
    if (is16(psm)) pw = 64, ph = 64;
    else if (psm == 0x13) pw = 128, ph = 64;
    else if (psm == 0x14) pw = 128, ph = 128;
    else pw = 64, ph = 32;
}

void block_dims(u32 psm, u32 &bw, u32 &bh) {
    if (is16(psm)) bw = 16, bh = 8;
    else if (psm == 0x13) bw = 16, bh = 16;
    else if (psm == 0x14) bw = 32, bh = 16;
    else bw = 8, bh = 8;
}

// the block (256 bytes) that holds pixel x, y of a buffer: gs.wgsl's addr() with the processor's copy of its tables
u32 block_at(const u32 *f, u32 x, u32 y) {
    const u32 pws = f[3] & 15u, phs = (f[3] >> 4) & 15u, nbit = (f[3] >> 10) & 1u;
    const u32 n = C.swizzle[f[2] + ((y & ((1u << phs) - 1u)) << pws) + (x & ((1u << pws) - 1u))];
    const u32 a = f[0] + ((x >> pws) + (y >> phs) * f[1]) * 8192u + (n >> nbit);
    return (a >> 8) & (kBlocks - 1);
}

template <typename Fn> void for_blocks(u32 psm, u32 bp, u32 bw, int x0, int y0, int x1, int y1, Fn &&fn) {
    u32 f[5];
    C.fmt_words(psm, bp, bw, f);
    u32 bwp, bhp;
    block_dims(psm, bwp, bhp);
    x0 = std::max(x0, 0), y0 = std::max(y0, 0);
    for (int by = y0 - y0 % static_cast<int>(bhp); by <= y1; by += static_cast<int>(bhp))
        for (int bx = x0 - x0 % static_cast<int>(bwp); bx <= x1; bx += static_cast<int>(bwp)) fn(block_at(f, static_cast<u32>(bx), static_cast<u32>(by)), bx, by, bwp, bhp);
}

template <typename Fn> void for_pages(u32 psm, u32 bp, u32 bw, int x0, int y0, int x1, int y1, Fn &&fn) {
    u32 pw, ph;
    page_dims(psm, pw, ph);
    const u32 per_row = std::max<u32>(1, bw * 64 / pw);
    x0 = std::max(x0, 0), y0 = std::max(y0, 0);
    if (x1 < x0 || y1 < y0) return;
    for (u32 py = static_cast<u32>(y0) / ph; py <= static_cast<u32>(y1) / ph; ++py)
        for (u32 px = static_cast<u32>(x0) / pw; px <= static_cast<u32>(x1) / pw; ++px) fn((bp / 32 + px + py * per_row) & (kPages - 1));
}

u64 g_block[kBlocks];     // content identity of each block (upload id and offset, mixed with the last one for partial writes)
u8 g_blk_stale[kBlocks];  // the block was drawn into by a target and memory does not hold that picture
int g_owner[kPages];      // the target that holds the page's picture, -1: memory holds it
bool g_page_stale[kPages];
u64 g_mem_read_submit[kPages];   // the submit in which a view-4 draw read the page of memory directly (hwr.wgsl from_mem); an upload or write-back into it later in that submit would change what the draw reads
u64 g_mem_read_hits = 0, g_mem_read_changes = 0;
u32 g_page_write[kPages], g_page_stored[kPages];   // draws into each page, and how many of them the last write-back to memory had (store_for_memory)
u64 g_fresh = 1;          // counter for identities that match nothing
u64 g_clut_id = 0;        // identity of the palette buffer's content (what the last palette loads put there)
// texture_key's memo (resolve_texture): a page's generation is the memory change count when one of its blocks' identity or stale flag
// last changed; a memo made at change count n is good while every page the texture covers is at <= n. g_mem_all moves on whole-memory changes.
u64 g_mem_gen = 1, g_mem_all = 0;
u64 g_page_gen[kPages];
inline void touch_page(u32 page) { g_page_gen[page & (kPages - 1)] = g_mem_gen; }
inline void touch_all() { g_mem_all = ++g_mem_gen; }
u64 g_epoch = 1;          // bumped whenever an upload, a palette load or a change of page owner may change what a setup reads

// ---------------------------------------------------------------- counters
enum Reason { kPrimKind, kDate, kFramePsm, kFbmsk, kBlend, kPabe, kNoDual, kTexTarget, kZeroWidth, kReasons };
const char *const kReasonName[kReasons] = {"lines/points", "DATE", "frame format", "partial frame mask", "blend form", "PABE", "no dual-source", "texture in a target (other layout)", "fbw 0"};
enum Approx { kOutside, kStaleTex, kColclamp, kRegionMip, kAfx, kDepthScale, kDeadLoad, kMipLevel0, kStaleSelf, kApproxes };
const char *const kApproxName[kApproxes] = {"outside target", "stale texture memory", "colour wrap", "mip with region clamp", "As > 0x80",
                                            "depth write dropped (other scale)", "load not done (cleared)", "mip read at level 0", "own target read as of earlier in the frame"};

struct Counters {
    u64 shapes = 0, drawn = 0, skipped = 0, clears = 0;
    u64 fallback[kReasons] = {}, approx[kApproxes] = {};
    u64 draws = 0, passes = 0, pipelines = 0, groups = 0, vertices = 0;
    u64 lookups = 0, hits = 0, misses = 0, decoded_texels = 0, target_reads = 0, self_copies = 0, loads = 0, stores = 0, targets_made = 0;
    u64 builds_epoch = 0; // of them only because the epoch moved on (same setup)
    u64 builds = 0;       // shape states worked out (build_state: a new setup, or the epoch or submit moved on)
    u64 memo_hits = 0;    // of them taken from the memo of recent states (memo_find)
    u64 view4_out = 0;    // view-4 shapes whose texels left the source target's pages (set up the old way)
    u64 mipneed[8] = {};  // PS2_HWR_LOG=1: mipmapped draws by the number of levels (1..7) their corners can sample at most (a conservative bound)
    u64 mipdraws = 0;     // mipmapped draws counted
    double build_ms = 0;  // their time (PS2_HWR_LOG only)
    u64 nodes = 0, renames = 0, surfaces_made = 0, copy_bytes = 0, overwrites = 0, dead_depth = 0, dead_loads = 0, grown = 0, recycled = 0;
    u64 att_loads = 0, att_stores = 0, att_load_bytes = 0, att_store_bytes = 0;  // render pass attachments that are loaded / stored (tile memory traffic)
    double cpu_ms = 0, sched_ms = 0;
};
Counters F, T;  // this frame, the whole run
u64 g_frames = 0;
bool g_stream_order = false;  // PS2_HWR_ORDER=stream: no renaming, no reordering (M1's order; for comparing)
// PS2_DRAW_TIME=ab / ab-memo: a measuring aid, the drawing thread's shortcuts (runs and state memo / only the memo) are switched off for
// every other 100 pictures (frame_end), so both are timed in one run under the same load
bool g_ab_off = false, g_ab_memo_only = false, g_ab_on = false;
bool g_no_view4 = false;      // set while a shape whose texels leave its view-4 source target is set up again the old way (shape())
bool g_dump_now = false;     // PS2_HWR_DUMP=<frame>: this frame's shapes and schedule are printed

// ---------------------------------------------------------------- surfaces (M2: the card textures that hold a target's picture)
// A target's picture lives in a surface. A full overwrite of the target (a clear, docs/hw-renderer-design.md 3.3 "renaming") gives it a new
// surface, so that draws that still read the old picture and draws of the new one do not depend on each other and can be reordered.
// Surfaces that no target holds any more go back to a pool after the submit that used them last.
struct Surf {
    WGPUTexture tex = nullptr;
    WGPUTextureView view = nullptr;
    WGPUBindGroup sample_group = nullptr, sync_group = nullptr;
    u32 w = 0, h = 0;
    Sc sc;
    bool depth = false;
    bool retired = false;  // no target holds it: its content is not needed after its last use in this submit
    u64 id = 0;            // resource identity for the dependency graph (new for every use as a new picture)
};
std::vector<std::unique_ptr<Surf>> g_surfs;       // every surface made (owned here)
std::vector<Surf *> g_pool;                        // free surfaces
std::vector<Surf *> g_retired;                     // given up in this submit: back to the pool after it
u64 g_res = 1;                                     // resource id counter (surfaces and cache textures)

// ---------------------------------------------------------------- targets
struct Rect {
    int x0, y0, x1, y1;
    bool hits(const Rect &o) const { return x0 <= o.x1 && o.x0 <= x1 && y0 <= o.y1 && o.y0 <= y1; }
};
struct Target {
    u32 bp = 0, bw = 0, psm = 0;  // psm: the class for lookups (CT32 for CT24); depth targets keep their Z format
    u32 last_psm = 0;             // the format last drawn (CT24 keeps the top byte when written back)
    bool depth = false, alive = true;
    u32 w = 0, h = 0, zmax = 0;
    Sc sc;                        // card texels per GS pixel (its surfaces')
    Surf *s = nullptr;            // the surface with the current picture
    u32 pages = 0;
    int valid_w = 0, valid_h = 0;
    Surf *snap = nullptr;         // a copy of the picture for draws that read the target they draw into
    Surf *snap_of = nullptr;      // the surface it was copied from
    std::vector<Rect> dirty;      // what was drawn into the surface since the copy
    std::vector<Rect> dirty_a;    // the part of it whose alpha was written (a draw that reads only the copy's alpha, view 3, needs no new copy for the rest)
    std::vector<u64> page_snap_frame;  // view-4 self reads: per page of the target, the frame its part of the copy was brought up to date in
                                       // (+1; read as of then for the rest of that frame, as store_for_memory once_per_frame writes a page back)
    u64 writes = 0, stored = ~0ull;  // draws into it so far, and how many of them the last write-back to memory (store_for_memory) had
    int sx0 = 0, sy0 = 0, sx1 = -1, sy1 = -1;  // the rectangle (target pixels) written back at that count
};
std::vector<std::unique_ptr<Target>> g_targets;
std::vector<WGPUTexture> g_release_tex;
std::vector<WGPUTextureView> g_release_view;
std::vector<WGPUBindGroup> g_release_group;
std::vector<int> g_dead_slots;  // slots of g_targets free again after the submit

// ---------------------------------------------------------------- texture cache
struct TexEntry {
    u64 key = 0;
    WGPUTexture tex = nullptr;
    WGPUTextureView view = nullptr;
    std::array<WGPUBindGroup, 32> groups{};
    u32 levels = 1, bytes = 0;
    u64 last_frame = 0;
    u64 id = 0;  // resource id (the copies that fill it and the draws that read it)
    u64 dims = 0;  // size and levels: which entries a recycled one can stand for
    bool transient = false;  // decoded from a target that was written back to memory (store_for_memory): the key never comes again, so it is recycled after a frame
};
// Entries not used for a second go here (by size) and are filled again for a new texture of that size: a movie's picture is a new texture
// every frame, and making a card texture, its view and groups each frame cost the phone's driver more than the decode (texture_recycle).
std::unordered_map<u64, std::vector<std::unique_ptr<TexEntry>>> g_tex_free;
constexpr std::size_t kFreePerSize = 16, kFreePerSizeBig = 256;
constexpr u64 kRecycleFrames = 50;
std::unordered_map<u64, std::unique_ptr<TexEntry>> g_cache;
u64 g_cache_bytes = 0, g_cache_budget = 256ull << 20;

// ---------------------------------------------------------------- the submit's recorded work
struct Vtx {
    float x, y, z;
    u32 rgba;
    float s, t, q, fog;
    u32 di;
    Vtx() {}  // not zeroed: g_verts grows by a whole run at once (prim_run) and every vertex is written right after
};
static_assert(sizeof(Vtx) == 36, "vertex layout");

// One draw call: pipelines as packed keys (the depth attachment's part is chosen per render pass when it is encoded)
struct DrawRec {
    u64 k1 = 0, k2 = 0;  // packed PipeKey | kKeyOn; 0: no such draw (k2: the second draw of a two-draw alpha test)
    WGPUBindGroup group = nullptr;
    u32 sc[4] = {};
    float blend = 0;
    u32 first = 0, count = 0;
    u64 res = 0;         // the resource (surface or cache texture) it samples, 0 none
    bool zuse = false;   // it needs the depth attachment (test or write)
    bool zread = false;  // it reads depth (a test other than ALWAYS)
};
constexpr u64 kKeyOn = 1ull << 63;

// The submit's work in stream order. Passes ("nodes": a run of draws into one colour surface and at most one depth surface) and copies are
// reordered at submit time (schedule()); loads and stores of graphics memory keep their place among gpu.cpp's compute commands.
enum OpKind : u8 { kOpPass, kOpCopyLevel, kOpLoad, kOpStore, kOpStoreZ, kOpSelfCopy, kOpShow };
struct Op {
    OpKind kind;
    u32 pos;
    int tgt = -1;  // the target (debug output)
    // pass
    Surf *cs = nullptr, *ds = nullptr;
    bool clear_c = false, clear_d = false;
    float cc[4] = {}, cz = 0;
    u32 first_draw = 0, draw_count = 0;
    // copies and transfers
    WGPUTexture tex = nullptr, tex2 = nullptr;
    WGPUBindGroup group = nullptr;
    u32 level = 0, w = 0, h = 0, offset = 0, pitch = 0, param = 0;
    Sc sc;               // the target's scale (self copies, loads)
    u64 rd = 0, wr = 0;  // resource read / written (copies, loads, stores)
};

std::vector<Op> g_ops;
std::vector<DrawRec> g_draws;
std::vector<Vtx> g_verts;
std::vector<std::array<u32, 16>> g_params;
int g_open_pass = -1;  // index of the open pass op (always the last op), -1 none
u64 g_submit = 1;
u64 g_dec_used = 0;

// What the schedule made of the ops: the encode order. A pass item covers several pass ops (merged into one render pass).
struct Item {
    u32 pos;               // encoded before gpu.cpp's compute command `pos` (~0u: after all of them)
    int op = -1;           // a single op (copy, load, store)
    u32 first = 0, n = 0;  // a render pass: g_pass_ops[first .. first + n)
    Surf *cs = nullptr, *ds = nullptr;
    bool clear_c = false, clear_d = false, store_c = true, store_d = true, zread = false;
    float cc[4] = {}, cz = 0;
    u32 draw_first = 0, draw_n = 0;  // into g_final (the merged draw calls)
};
std::vector<Item> g_items;
std::vector<int> g_pass_ops;
std::vector<DrawRec> g_final;
std::vector<Vtx> g_vorder;
// The vertices go to the card as they were recorded and a draw call merges only neighbours that are next to each other there (they nearly
// always are: the schedule moves whole nodes). Copying all of them into draw order cost ~2 ms a frame on the PC for Harry Potter 2's 210000
// vertices (7.5 MB), more on a phone. PS2_HWR_VORDER=1: the copy into draw order as before (more merged calls when nodes were moved).
const bool g_reorder_verts = os::env("PS2_HWR_VORDER") == "1";
inline const std::vector<Vtx> &vsource() { return g_reorder_verts ? g_vorder : g_verts; }
std::size_t g_next_item = 0;

WGPUBuffer g_vbuf = nullptr, g_pbuf = nullptr;
u64 g_vbuf_size = 0, g_pbuf_size = 0;
WGPUBindGroup g_param_group = nullptr;

// ---------------------------------------------------------------- pipelines and layouts
WGPUShaderModule g_module = nullptr, g_sync_module = nullptr;
// The draw shader's address tables (view 4: a target read through graphics memory addresses as another format): gs.wgsl's tables (byte
// offset inside a page of every pixel of each format, T4: nibble index) and their inverses for the target formats (CT32: word -> x | y << 8,
// CT16 and CT16S: halfword -> x | y << 8), made once from gpu.cpp's copy.
constexpr u32 kInv32 = 45056, kInv16 = kInv32 + 2048, kInv16S = kInv16 + 4096, kAddrWords = kInv16S + 4096;
WGPUBuffer g_addr = nullptr;
WGPUBindGroupLayout g_params_layout = nullptr, g_tex_layout = nullptr, g_store_layout = nullptr, g_load_layout = nullptr, g_depth_layout = nullptr;
WGPUPipelineLayout g_draw_layout = nullptr;
WGPUComputePipeline g_load_pipe = nullptr, g_store_pipe = nullptr, g_store_z_pipe = nullptr;
std::unordered_map<u64, WGPURenderPipeline> g_pipes;
WGPUSampler g_samplers[32] = {};
WGPUTexture g_dummy = nullptr;
WGPUTextureView g_dummy_view = nullptr;
WGPUBindGroup g_dummy_group = nullptr;

struct PipeKey {
    u32 tex = 0, atest = 0;
    bool dual = false, blend = false;
    u32 src = 0, dst = 0, op = 0;
    u32 mask = 0xF;
    bool depth = false, zwrite = false;
    u32 zcmp = WGPUCompareFunction_Always;
    u64 pack() const {
        return static_cast<u64>(tex) | static_cast<u64>(atest) << 2 | static_cast<u64>(dual) << 4 | static_cast<u64>(blend) << 5 | static_cast<u64>(src) << 6 |
               static_cast<u64>(dst) << 11 | static_cast<u64>(op) << 16 | static_cast<u64>(mask) << 19 | static_cast<u64>(depth) << 23 | static_cast<u64>(zwrite) << 24 |
               static_cast<u64>(zcmp) << 25;
    }
    static PipeKey unpack(u64 v) {
        PipeKey k;
        k.tex = v & 3u, k.atest = (v >> 2) & 3u, k.dual = (v >> 4) & 1u, k.blend = (v >> 5) & 1u, k.src = (v >> 6) & 31u, k.dst = (v >> 11) & 31u;
        k.op = (v >> 16) & 7u, k.mask = (v >> 19) & 15u, k.depth = (v >> 23) & 1u, k.zwrite = (v >> 24) & 1u, k.zcmp = (v >> 25) & 15u;
        return k;
    }
};

// The pipelines made are listed in a file (one hex key per line) and made at the next start, so that a scene's first frame does not wait for
// the driver (slow on phones): PS2_HWR_PIPELINES=<file>, else games/<name>/hw/hwr_pipelines.txt beside the recorded vector programs.
// Keys of the race and the cup race are built in as well (kBuiltinKeys).
// the 20 pipelines of the race and the cup race (recorded with PS2_HWR_PIPELINES on a device with dual-source blending; init converts the
// dual-source ones when the device has none)
const u64 kBuiltinKeys[] = {0x103b6320ull, 0x10780000ull, 0x10780002ull, 0x10780006ull, 0x107908a2ull, 0x107910a2ull, 0x10796b22ull, 0x10f80001ull, 0x10f910a2ull, 0x10f96b21ull,
                            0x11f80001ull, 0x11f80002ull, 0x11f96b22ull, 0xeb913b1ull,  0xeb97bb9ull,  0xebb10a2ull,  0xfb80001ull,  0xfb97bb5ull,  0xff80001ull,  0xff97bb5ull};
std::string g_pipe_path;
bool g_pipe_loading = false;
void record_key(u64 key) {
    if (g_pipe_path.empty() || g_pipe_loading) return;
    if (std::FILE *f = os::open(g_pipe_path.c_str(), "ab")) {
        std::fprintf(f, "%llx\n", static_cast<unsigned long long>(key));
        std::fclose(f);
    }
}

WGPURenderPipeline pipeline(const PipeKey &k) {
    const u64 key = k.pack();
    if (const auto it = g_pipes.find(key); it != g_pipes.end()) return it->second;
    record_key(key);
    WGPUVertexAttribute at[6];
    const WGPUVertexFormat formats[6] = {WGPUVertexFormat_Float32x2, WGPUVertexFormat_Float32, WGPUVertexFormat_Unorm8x4, WGPUVertexFormat_Float32x3, WGPUVertexFormat_Float32, WGPUVertexFormat_Uint32};
    const u32 offsets[6] = {0, 8, 12, 16, 28, 32};
    for (int i = 0; i < 6; ++i) {
        at[i] = WGPU_VERTEX_ATTRIBUTE_INIT;
        at[i].format = formats[i];
        at[i].offset = offsets[i];
        at[i].shaderLocation = static_cast<u32>(i);
    }
    WGPUVertexBufferLayout vl = WGPU_VERTEX_BUFFER_LAYOUT_INIT;
    vl.arrayStride = sizeof(Vtx);
    vl.stepMode = WGPUVertexStepMode_Vertex;
    vl.attributeCount = 6;
    vl.attributes = at;
    WGPUConstantEntry consts[2];
    consts[0] = WGPU_CONSTANT_ENTRY_INIT;
    consts[0].key = sv("TEX");
    consts[0].value = k.tex;
    consts[1] = WGPU_CONSTANT_ENTRY_INIT;
    consts[1].key = sv("ATEST");
    consts[1].value = k.atest;
    WGPUBlendState bs = WGPU_BLEND_STATE_INIT;
    bs.color.srcFactor = static_cast<WGPUBlendFactor>(k.src);
    bs.color.dstFactor = static_cast<WGPUBlendFactor>(k.dst);
    bs.color.operation = static_cast<WGPUBlendOperation>(k.op);
    bs.alpha.srcFactor = WGPUBlendFactor_One;
    bs.alpha.dstFactor = WGPUBlendFactor_Zero;
    bs.alpha.operation = WGPUBlendOperation_Add;
    WGPUColorTargetState ct = WGPU_COLOR_TARGET_STATE_INIT;
    ct.format = WGPUTextureFormat_RGBA8Unorm;
    ct.blend = k.blend ? &bs : nullptr;
    ct.writeMask = k.mask;
    WGPUFragmentState fs = WGPU_FRAGMENT_STATE_INIT;
    fs.module = g_module;
    fs.entryPoint = sv(C.gl ? (k.dual ? "fs_dual_gl" : "fs_gl") : (k.dual ? "fs_dual" : "fs"));
    fs.constantCount = 2;
    fs.constants = consts;
    fs.targetCount = 1;
    fs.targets = &ct;
    WGPUDepthStencilState ds = WGPU_DEPTH_STENCIL_STATE_INIT;
    ds.format = WGPUTextureFormat_Depth32Float;
    ds.depthWriteEnabled = k.zwrite ? WGPUOptionalBool_True : WGPUOptionalBool_False;
    ds.depthCompare = static_cast<WGPUCompareFunction>(k.zcmp);
    WGPURenderPipelineDescriptor d = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    d.layout = g_draw_layout;
    d.vertex.module = g_module;
    d.vertex.entryPoint = sv(C.gl ? "vs_gl" : "vs");
    d.vertex.bufferCount = 1;
    d.vertex.buffers = &vl;
    d.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    d.primitive.cullMode = WGPUCullMode_None;
    d.depthStencil = k.depth ? &ds : nullptr;
    d.multisample.count = 1;
    d.multisample.mask = 0xFFFFFFFFu;
    d.fragment = &fs;
    WGPURenderPipeline p = wgpuDeviceCreateRenderPipeline(C.device, &d);
    g_pipes.emplace(key, p);
    return p;
}

WGPUSampler sampler(u32 idx) {
    if (g_samplers[idx]) return g_samplers[idx];
    WGPUSamplerDescriptor d = WGPU_SAMPLER_DESCRIPTOR_INIT;
    d.magFilter = (idx & 1) ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest;
    d.minFilter = (idx & 2) ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest;
    d.mipmapFilter = (idx & 4) ? WGPUMipmapFilterMode_Linear : WGPUMipmapFilterMode_Nearest;
    d.addressModeU = (idx & 8) ? WGPUAddressMode_ClampToEdge : WGPUAddressMode_Repeat;
    d.addressModeV = (idx & 16) ? WGPUAddressMode_ClampToEdge : WGPUAddressMode_Repeat;
    d.addressModeW = WGPUAddressMode_ClampToEdge;
    d.lodMinClamp = 0.0f;
    d.lodMaxClamp = 32.0f;
    d.maxAnisotropy = 1;
    g_samplers[idx] = wgpuDeviceCreateSampler(C.device, &d);
    return g_samplers[idx];
}

// bindings 2 and 3: bilinear with clamp and with repeat, for a target read as a texture at another scale (hwr.wgsl sample_loads' fast path)
WGPUBindGroup tex_group(WGPUTextureView view, u32 sampler_idx) {
    WGPUBindGroupEntry e[6] = {WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT,
                               WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT};
    e[0].binding = 0;
    e[0].textureView = view;
    e[1].binding = 1;
    e[1].sampler = sampler(sampler_idx);
    e[2].binding = 2;
    e[2].sampler = sampler(1 | 2 | 8 | 16);
    e[3].binding = 3;
    e[3].sampler = sampler(1 | 2);
    e[4].binding = 4;  // the view again for bindings 2 and 3 (hwr.wgsl tex_clamp, tex_repeat: one sampler per texture for OpenGL)
    e[4].textureView = view;
    e[5].binding = 5;
    e[5].textureView = view;
    WGPUBindGroupDescriptor d = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    d.layout = g_tex_layout;
    d.entryCount = 6;
    d.entries = e;
    return wgpuDeviceCreateBindGroup(C.device, &d);
}

WGPUBindGroup sync_group(WGPUBindGroupLayout layout, WGPUTextureView view) {
    WGPUBindGroupEntry e = WGPU_BIND_GROUP_ENTRY_INIT;
    e.binding = 0;
    e.textureView = view;
    WGPUBindGroupDescriptor d = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    d.layout = layout;
    d.entryCount = 1;
    d.entries = &e;
    return wgpuDeviceCreateBindGroup(C.device, &d);
}

WGPUBindGroupLayout make_layout(const WGPUBindGroupLayoutEntry *e, std::size_t n) {
    WGPUBindGroupLayoutDescriptor d = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    d.entryCount = n;
    d.entries = e;
    return wgpuDeviceCreateBindGroupLayout(C.device, &d);
}

WGPUComputePipeline compute(const char *entry, WGPUBindGroupLayout second) {
    WGPUBindGroupLayout layouts[2] = {C.layout, second};
    WGPUPipelineLayoutDescriptor pd = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pd.bindGroupLayoutCount = 2;
    pd.bindGroupLayouts = layouts;
    WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(C.device, &pd);
    WGPUComputePipelineDescriptor d = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    d.layout = pl;
    d.compute.module = g_sync_module;
    d.compute.entryPoint = sv(entry);
    WGPUConstantEntry ce = WGPU_CONSTANT_ENTRY_INIT;
    ce.key = sv("SCALE");
    ce.value = static_cast<double>(g_s);
    d.compute.constantCount = 1;
    d.compute.constants = &ce;
    WGPUComputePipeline p = wgpuDeviceCreateComputePipeline(C.device, &d);
    wgpuPipelineLayoutRelease(pl);
    return p;
}

// ---------------------------------------------------------------- recording
u32 position() { return C.position(); }

void close_pass() { g_open_pass = -1; }

Op &add_op(OpKind kind) {
    close_pass();
    Op op{};
    op.kind = kind;
    // only loads and stores of graphics memory keep a place among gpu.cpp's compute commands (asking for the place ends gpu.cpp's open run
    // of uploads, so passes and copies do not ask: schedule() puts them after the compute commands)
    op.pos = g_stream_order || kind == kOpLoad || kind == kOpStore || kind == kOpStoreZ ? position() : 0;
    g_ops.push_back(op);
    return g_ops.back();
}

// a surface from the pool (or a new one); it gets a new resource id
Surf *get_surf(u32 w, u32 h, bool depth, const Sc &sc) {
    Surf *s = nullptr;
    for (std::size_t k = 0; k < g_pool.size(); ++k)
        if (g_pool[k]->w == w && g_pool[k]->h == h && g_pool[k]->depth == depth && g_pool[k]->sc == sc) {
            s = g_pool[k];
            g_pool.erase(g_pool.begin() + static_cast<std::ptrdiff_t>(k));
            break;
        }
    if (!s) {
        g_surfs.push_back(std::make_unique<Surf>());
        s = g_surfs.back().get();
        s->w = w, s->h = h, s->depth = depth, s->sc = sc;
        WGPUTextureDescriptor d = WGPU_TEXTURE_DESCRIPTOR_INIT;
        d.size = {card_x(w, sc), card_y(h, sc), 1};
        d.format = depth ? WGPUTextureFormat_Depth32Float : WGPUTextureFormat_RGBA8Unorm;
        d.usage = depth ? (WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding)
                        : (WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst);
        d.mipLevelCount = 1;
        d.sampleCount = 1;
        s->tex = wgpuDeviceCreateTexture(C.device, &d);
        s->view = wgpuTextureCreateView(s->tex, nullptr);
        if (depth) {
            s->sync_group = sync_group(g_depth_layout, s->view);
        } else {
            s->sample_group = tex_group(s->view, 0);
            s->sync_group = sync_group(g_store_layout, s->view);
        }
        ++F.surfaces_made;
    }
    s->retired = false;
    s->id = ++g_res;
    return s;
}

// Loads (memory -> a new target) are written by a compute program into a staging texture of the surface's size and then copied into the
// surface: the surfaces themselves have no storage usage (tile GPUs such as Mali do not compress a texture that has it).
struct Staging {
    WGPUTexture tex = nullptr;
    WGPUTextureView view = nullptr;
    WGPUBindGroup group = nullptr;
};
std::unordered_map<u64, Staging> g_staging;
Staging &staging(u32 w, u32 h, const Sc &sc) {
    Staging &st = g_staging[mix(mix((static_cast<u64>(w) << 32) | h, fbits(sc.x)), fbits(sc.y))];
    if (!st.tex) {
        WGPUTextureDescriptor d = WGPU_TEXTURE_DESCRIPTOR_INIT;
        d.size = {card_x(w, sc), card_y(h, sc), 1};
        d.format = WGPUTextureFormat_RGBA8Unorm;
        d.usage = WGPUTextureUsage_StorageBinding | WGPUTextureUsage_CopySrc;
        d.mipLevelCount = 1;
        d.sampleCount = 1;
        st.tex = wgpuDeviceCreateTexture(C.device, &d);
        st.view = wgpuTextureCreateView(st.tex, nullptr);
        st.group = sync_group(g_load_layout, st.view);
    }
    return st;
}

// the surface's picture is not needed after this submit's uses: back to the pool after the submit
void retire(Surf *s) {
    if (!s || s->retired) return;
    s->retired = true;
    g_retired.push_back(s);
}

void drop_snap(Target &t) {
    retire(t.snap);
    t.snap = t.snap_of = nullptr;
    t.dirty.clear();
    t.dirty_a.clear();
}

void release_target(int i) {
    Target &t = *g_targets[static_cast<std::size_t>(i)];
    if (!t.alive) return;
    t.alive = false;
    retire(t.s);
    drop_snap(t);
    for (int k = 0; k < static_cast<int>(kPages); ++k)
        if (g_owner[k] == i) g_owner[k] = -1;
    g_dead_slots.push_back(i);
    ++g_epoch;
}

// a target drawn below the rows it was made with becomes `h` rows tall: a new surface, the colour picture copied into its top (depth is not
// kept: a depth format is copied only whole, and the games clear it per frame)
void grow(int ti, u32 h) {
    Target &t = *g_targets[static_cast<std::size_t>(ti)];
    if (t.h >= h) return;
    Surf *old = t.s;
    const u32 oh = t.h;
    t.h = h;
    drop_snap(t);
    t.s = get_surf(t.w, t.h, t.depth, t.sc);
    if (!t.depth) {
        Op &op = add_op(kOpSelfCopy);
        op.tex = old->tex;
        op.tex2 = t.s->tex;
        op.w = t.w;
        op.h = oh;
        op.sc = t.sc;
        op.rd = old->id;
        op.wr = t.s->id;
        op.tgt = ti;
    }
    retire(old);
    ++F.grown;
    ++g_epoch;
}

// a full overwrite of the target: its new picture goes into a new surface (the old one stays for the draws that still read it)
void rename(Target &t) {
    retire(t.s);
    drop_snap(t);
    t.s = get_surf(t.w, t.h, t.depth, t.sc);
    ++F.renames;
    ++g_epoch;  // the setups that read this target take the new surface
}

int page_log();
void lose_page(u32 page) {
    const int o = g_owner[page];
    if (o < 0) return;
    if (static_cast<int>(page) == page_log()) std::fprintf(stderr, "[hwr-page] frame %llu page %u back to memory from target %d\n", static_cast<unsigned long long>(g_frames), page, o);
    g_owner[page] = -1;
    ++g_epoch;
    Target &t = *g_targets[static_cast<std::size_t>(o)];
    if (t.pages) --t.pages;
    if (!t.pages) release_target(o);
}

// PS2_HWR_PAGE_LOG=<page>: who takes that page of graphics memory (a target's draw) and when it goes back to memory (an upload)
int page_log() {
    static const int p = [] {
        const std::string v = os::env("PS2_HWR_PAGE_LOG");
        return v.empty() ? -1 : std::atoi(v.c_str());
    }();
    return p;
}
void take_page(int ti, u32 page) {
    if (g_owner[page] == ti) return;
    if (static_cast<int>(page) == page_log())
        std::fprintf(stderr, "[hwr-page] frame %llu page %u taken by target %d (bp %04X psm %02X) from %d\n", static_cast<unsigned long long>(g_frames), page, ti,
                     g_targets[static_cast<std::size_t>(ti)]->bp, g_targets[static_cast<std::size_t>(ti)]->psm, g_owner[page]);
    lose_page(page);
    g_owner[page] = ti;
    ++g_targets[static_cast<std::size_t>(ti)]->pages;
    g_page_stale[page] = true;
    for (u32 b = page * 32; b < page * 32 + 32; ++b) {
        g_block[b] = mix(0xD7A3ull, ++g_fresh);
        g_blk_stale[b] = 1;
        touch_page(b / 32);
    }
    ++g_epoch;
}

int new_slot() {
    // a slot freed in an earlier submit can be used again (its texture was released after that submit)
    if (!g_dead_slots.empty()) {
        for (std::size_t k = 0; k < g_dead_slots.size(); ++k) {
            const int s = g_dead_slots[k];
            if (!g_targets[static_cast<std::size_t>(s)]) {
                g_dead_slots.erase(g_dead_slots.begin() + static_cast<std::ptrdiff_t>(k));
                return s;
            }
        }
    }
    g_targets.emplace_back();
    return static_cast<int>(g_targets.size() - 1);
}

// `sc`: the scale of a new target (a colour target found keeps its own; a depth target is found only at its colour target's scale, so the
// render pass's attachments are the same size)
int find_target(u32 bp, u32 bw, u32 psm, bool depth, u32 w, u32 h, const Sc &sc) {
    for (std::size_t i = 0; i < g_targets.size(); ++i) {
        const Target *t = g_targets[i].get();
        if (t && t->alive && t->depth == depth && t->bp == bp && t->bw == bw && t->psm == psm && (!depth || (t->w == w && t->h == h && t->sc == sc))) return static_cast<int>(i);
    }
    const int slot = new_slot();
    auto t = std::make_unique<Target>();
    t->bp = bp, t->bw = bw, t->psm = psm, t->last_psm = psm, t->depth = depth, t->w = w, t->h = h, t->sc = sc;
    t->s = get_surf(w, h, depth, sc);
    if (g_log) std::fprintf(stderr, "[hwr] new %s target %04x/%u %ux%u at %.3fx, %.3fx\n", depth ? "depth" : "colour", bp, bw, w, h, sc.x, sc.y);
    g_targets[static_cast<std::size_t>(slot)] = std::move(t);
    ++F.targets_made;
    ++g_epoch;
    // a colour target starts with what memory holds where memory is current (the picture of a state just loaded, an uploaded picture)
    if (!depth && !g_page_stale[(bp / 32) & (kPages - 1)]) {
        Target &tt = *g_targets[static_cast<std::size_t>(slot)];
        Op &op = add_op(kOpLoad);
        const u32 words[10] = {bp, bw, psm == 0x00 ? tt.last_psm : psm, 0, 0, w, h, 0, fbits(sc.x), fbits(sc.y)};
        op.param = C.params(words, 10);
        Staging &st = staging(w, h, sc);
        op.sc = sc;
        op.group = st.group;
        op.tex = st.tex;
        op.tex2 = tt.s->tex;
        op.w = w, op.h = h;
        op.wr = tt.s->id;
        op.tgt = slot;
        ++F.loads;
    }
    return slot;
}

u32 alloc_decoded(u32 words) {
    const u64 at = (g_dec_used + 63) & ~63ull;
    if (at + words > C.decoded_words) {
        C.submit_now();
        g_dec_used = words;
        return 0;
    }
    g_dec_used = at + words;
    return static_cast<u32>(at);
}

void drop_entry(TexEntry &e) {
    g_release_tex.push_back(e.tex);
    g_release_view.push_back(e.view);
    for (WGPUBindGroup g : e.groups)
        if (g) g_release_group.push_back(g);
}

// an entry out of the cache: kept for a new texture of its size, or released
void recycle(std::unordered_map<u64, std::unique_ptr<TexEntry>>::iterator it) {
    std::unique_ptr<TexEntry> e = std::move(it->second);
    g_cache_bytes -= e->bytes;
    g_cache.erase(it);
    auto &free = g_tex_free[e->dims];
    if (free.size() < (e->transient ? kFreePerSizeBig : kFreePerSize)) free.push_back(std::move(e));
    else drop_entry(*e);
}

void cache_evict() {
    // entries not used for a second (a movie's pictures): recycled
    for (auto it = g_cache.begin(); it != g_cache.end();) {
        auto next = std::next(it);
        if (it->second->last_frame + (it->second->transient ? 1 : kRecycleFrames) < g_frames) recycle(it);
        it = next;
    }
    if (g_cache_bytes <= g_cache_budget) return;
    std::vector<std::pair<u64, u64>> order;  // (last frame, key)
    for (auto &[k, e] : g_cache)
        if (e->last_frame + 2 < g_frames) order.emplace_back(e->last_frame, k);
    std::sort(order.begin(), order.end());
    for (auto &[f, k] : order) {
        if (g_cache_bytes <= g_cache_budget * 3 / 4) break;
        recycle(g_cache.find(k));
    }
}

// ---------------------------------------------------------------- what a setup reads
struct Source {
    int kind = 0;  // 0 none, 1 cache texture (sampler), 2 texel loads from a texture or target
    WGPUBindGroup group = nullptr;
    u32 view = 0;  // texel loads: 0 RGBA as stored, 1 CT24 view, 2 CT16 view
    u32 levels = 1;
    int target = -1;  // read from this target (self reads go through a scratch copy)
    u64 res = 0;      // the resource read
    bool fail = false;
};

struct KeyMemo {
    u64 key = 0, made = 0;
    bool stale = false;
};
std::unordered_map<u64, KeyMemo> g_key_memo;

u64 texture_key_full(const u32 *w, const u32 *mip, u32 top, bool &stale);

// texture_key without walking every block when no page the texture covers changed since the last time this setup asked
u64 texture_key(const u32 *w, const u32 *mip, u32 top, bool &stale) {
    const u32 tbp = w[12], tbw = w[13], tpsm = w[14], tw = w[15] & 0xFFu, th = w[15] >> 8;
    u64 sk = mix(mix(0x3E30ull, tbp | tbw << 14 | tpsm << 20 | top << 26), tw | th << 8 | w[16] << 16);
    sk = mix(mix(mix(sk, w[17]), w[20] | (w[0] & 32768u)), g_clut_id);
    for (u32 n = 1; n <= top; ++n) sk = mix(sk, mip[n]);
    const auto it = g_key_memo.find(sk);
    if (it != g_key_memo.end() && it->second.made >= g_mem_all) {
        bool ok = true;
        const u64 made = it->second.made;
        for (u32 n = 0; n <= top && ok; ++n) {
            const u32 bp = n ? (mip[n] & 0x3FFFu) : tbp, bw = n ? ((mip[n] >> 14) & 0x3Fu) : tbw;
            const u32 lw = tw > n ? tw - n : 0, lh = th > n ? th - n : 0;
            for_pages(tpsm, bp, bw, 0, 0, (1 << lw) - 1, (1 << lh) - 1, [&](u32 page) { ok = ok && g_page_gen[page & (kPages - 1)] <= made; });
        }
        if (ok) {
            stale = stale || it->second.stale;
            return it->second.key;
        }
    }
    if (g_key_memo.size() > 4096) g_key_memo.clear();
    bool st = false;
    const u64 key = texture_key_full(w, mip, top, st);
    g_key_memo[sk] = KeyMemo{key, g_mem_gen, st};
    ++g_mem_gen;  // later changes get a higher generation than this memo
    stale = stale || st;
    return key;
}

u64 texture_key_full(const u32 *w, const u32 *mip, u32 top, bool &stale) {
    const u32 tbp = w[12], tbw = w[13], tpsm = w[14], tw = w[15] & 0xFFu, th = w[15] >> 8, cpsm = w[16], csa = w[17];
    u64 h = mix(0x7E8ull, tpsm | tbw << 8 | tw << 16 | th << 20 | top << 24);
    const bool pal = tpsm == 0x13 || tpsm == 0x14 || tpsm == 0x1B || tpsm == 0x24 || tpsm == 0x2C;
    const bool aem = (w[0] & 32768u) != 0;
    if (pal) {
        h = mix(h, g_clut_id);
        h = mix(h, cpsm | ((tpsm == 0x14 || tpsm == 0x24 || tpsm == 0x2C) ? csa << 8 : 0u));
        if (cpsm != 0x00 && cpsm != 0x01) h = mix(h, w[20] | (aem ? 1u << 16 : 0u));
    } else if (tpsm == 0x01 || is16(tpsm)) {
        h = mix(h, w[20] | (aem ? 1u << 16 : 0u));
    }
    for (u32 n = 0; n <= top; ++n) {
        const u32 bp = n ? (mip[n] & 0x3FFFu) : tbp, bw = n ? ((mip[n] >> 14) & 0x3Fu) : tbw;
        const u32 lw = tw > n ? tw - n : 0, lh = th > n ? th - n : 0;
        h = mix(h, bw);
        for_blocks(tpsm, bp, bw, 0, 0, (1 << lw) - 1, (1 << lh) - 1, [&](u32 b, int, int, u32, u32) {
            h = mix(h, g_block[b]);
            stale = stale || g_blk_stale[b];
        });
    }
    return h;
}

TexEntry *decode_texture(u64 key, const u32 *w, const u32 *mip, u32 top) {
    if (prov::g_on) prov::count(prov::kDecodes);
    const u32 tw = w[15] & 0xFFu, th = w[15] >> 8;
    u32 levels = top + 1;
    levels = std::min(levels, std::max(tw, th) + 1);
    // PS2_HWR_LEVELS=<n>: at most n mip levels are decoded per texture (default: all). Each level is a decode dispatch and a copy on the card; a phone
    // is better off with 1 or 2 (the picture of far surfaces is a little sharper and shimmers more).
    static const u32 max_levels = [] {
        const std::string v = os::env("PS2_HWR_LEVELS");
        return v.empty() ? 16u : static_cast<u32>(std::clamp(std::atoi(v.c_str()), 1, 16));
    }();
    levels = std::min(levels, max_levels);
    const u64 dims = (static_cast<u64>(tw) << 16) | (static_cast<u64>(th) << 8) | levels;
    std::unique_ptr<TexEntry> e;
    if (auto f = g_tex_free.find(dims); f != g_tex_free.end() && !f->second.empty()) {  // a recycled one: its texture, view and groups stay
        e = std::move(f->second.back());
        f->second.pop_back();
        e->last_frame = 0;
        e->transient = false;
        ++F.recycled;
    } else {
        e = std::make_unique<TexEntry>();
        WGPUTextureDescriptor d = WGPU_TEXTURE_DESCRIPTOR_INIT;
        d.size = {1u << tw, 1u << th, 1};
        d.format = WGPUTextureFormat_RGBA8Unorm;
        d.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
        d.mipLevelCount = levels;
        d.sampleCount = 1;
        e->tex = wgpuDeviceCreateTexture(C.device, &d);
        e->view = wgpuTextureCreateView(e->tex, nullptr);
    }
    e->key = key;
    e->levels = levels;
    e->dims = dims;
    e->id = ++g_res;
    u32 bytes = 0;
    for (u32 n = 0; n < levels; ++n) {
        u32 s[40];
        std::memcpy(s, w, sizeof s);
        const u32 bp = n ? (mip[n] & 0x3FFFu) : w[12], bw = n ? ((mip[n] >> 14) & 0x3Fu) : w[13];
        const u32 lw = tw > n ? tw - n : 0, lh = th > n ? th - n : 0, pitch_log = std::max<u32>(lw, 6);
        s[12] = bp;
        s[13] = bw;
        s[15] = pitch_log | (lh << 8);
        C.fmt_words(w[14], bp, bw, s + 25);
        const u32 words = 1u << (pitch_log + lh);
        const u32 at = alloc_decoded(words);
        C.decode(s, at, words);
        Op &op = add_op(kOpCopyLevel);
        op.tex = e->tex;
        op.wr = e->id;
        op.level = n;
        op.w = 1u << lw;
        op.h = 1u << lh;
        op.offset = at;
        op.pitch = 1u << pitch_log;
        bytes += (1u << lw) * (1u << lh) * 4;
        F.decoded_texels += (1u << lw) * (1u << lh);
    }
    e->bytes = bytes;
    g_cache_bytes += bytes;
    TexEntry *raw = e.get();
    g_cache.emplace(key, std::move(e));
    ++F.misses;
    return raw;
}

// ---------------------------------------------------------------- the state of a run of shapes
struct DrawState {
    u64 serial = ~0ull, epoch = 0, submit = 0;
    int skip = 0;  // 1: draws nothing, 2: fallback (reason counted)
    Reason reason = kPrimKind;
    int color = -1, depth = -1;
    u64 kp1 = 0, kp2 = 0;  // the draws' packed pipeline keys (DrawRec::k1, k2)
    PipeKey k1, k2;
    Source src;
    u32 param = 0;
    std::array<u32, 16> pw{};
    float blend = 0;
    int sc[4] = {};
    double zscale = 1.0;
    u32 zmax = 0;
    bool needs_depth = false, zwrite_any = false, cwrite_any = false, awrite_any = false, color_full = false;
    bool self_read = false;
    bool zread = false;
    bool no_memo = false;  // read a target's picture through a write-back to memory: later writes to it are not seen in the epoch (memo_find)
};
DrawState g_ds;


// premul: 0 the card multiplies, 1 the shader gives floor(Cs * FIX / 128) and the card adds it (the GS's truncation, exact), 2 ceil (for -Cs * FIX)
bool blend_map(u32 A, u32 B, u32 Cc, u32 Dd, bool dual, bool alpha_written, PipeKey &k, bool &uses_as, bool &ok, u32 &premul) {
    ok = true;
    uses_as = false;
    premul = 0;
    k.blend = true;
    static const bool dsta_approx = os::env("PS2_HWR_DSTA_APPROX") != "0";   // approximation: the card's destination alpha (0..255) as the factor, where the GS's is 0..128 (exact version: a shader that reads the destination); PS2_HWR_DSTA_APPROX=0 drops these shapes
    if (Cc == 1 && dsta_approx) {
        const WGPUBlendFactor one = WGPUBlendFactor_One, zero = WGPUBlendFactor_Zero, da = WGPUBlendFactor_DstAlpha, ida = WGPUBlendFactor_OneMinusDstAlpha;
        if (A == 2 && B == 0 && Dd == 0) {  // Cs * (1 - Ad)
            k.src = ida, k.dst = zero, k.op = WGPUBlendOperation_Add;
            return true;
        }
        if (A == 0 && B == 2 && Dd == 1) {  // Cs * Ad + Cd
            k.src = da, k.dst = one, k.op = WGPUBlendOperation_Add;
            return true;
        }
    }
    if (Cc == 1) {  // destination alpha as the factor: not done yet (see PS2_HWR_DSTA_APPROX)
        ok = false;
        return false;
    }
    WGPUBlendFactor cf = WGPUBlendFactor_Constant, icf = WGPUBlendFactor_OneMinusConstant;
    if (Cc == 0) {
        uses_as = true;
        // Without dual-source blending the factor is the fragment's alpha, which is then what the card writes as alpha too (As/128 of full scale
        // instead of the raw As). That only matters for a later read of the destination alpha, and a shape that is not drawn at all (the menu's
        // text and buttons) matters more: drawn with that alpha, unless PS2_HWR_AS_STRICT=1 (the shapes are dropped as before).
        static const bool strict = os::env("PS2_HWR_AS_STRICT") == "1";
        if (dual) cf = WGPUBlendFactor_Src1, icf = WGPUBlendFactor_OneMinusSrc1;
        else if (!alpha_written || !strict) cf = WGPUBlendFactor_SrcAlpha, icf = WGPUBlendFactor_OneMinusSrcAlpha;
        else {
            ok = false;
            return false;
        }
    }
    const auto set = [&](WGPUBlendFactor s, WGPUBlendFactor d, WGPUBlendOperation o) {
        k.src = s, k.dst = d, k.op = o;
        return true;
    };
    const WGPUBlendOperation add = WGPUBlendOperation_Add, sub = WGPUBlendOperation_Subtract, rsub = WGPUBlendOperation_ReverseSubtract;
    const WGPUBlendFactor one = WGPUBlendFactor_One, zero = WGPUBlendFactor_Zero;
    if (A == B) {
        if (Dd == 0) {
            k.blend = false;
            uses_as = false;
            return true;
        }
        uses_as = false;
        return Dd == 1 ? set(zero, one, add) : set(zero, zero, add);
    }
    if (A == 0 && B == 2 && Cc == 2 && Dd != 0) {
        premul = 1;
        return Dd == 1 ? set(one, one, add) : set(one, zero, add);
    }
    if (A == 0 && B == 2) return Dd == 1 ? set(cf, one, add) : Dd == 2 ? set(cf, zero, add) : (ok = false);
    if (A == 0 && B == 1) return Dd == 1 ? set(cf, icf, add) : Dd == 2 ? set(cf, cf, sub) : (ok = false);
    if (A == 1 && B == 0) return Dd == 2 ? set(cf, cf, rsub) : Dd == 0 ? set(icf, cf, add) : (ok = false);
    if (A == 1 && B == 2) return Dd == 0 ? set(one, cf, add) : Dd == 2 ? set(zero, cf, add) : (ok = false);
    if (A == 2 && B == 0 && Cc == 2 && Dd == 1) {
        premul = 2;
        return set(one, one, rsub);
    }
    if (A == 2 && B == 0) return Dd == 1 ? set(cf, one, rsub) : Dd == 0 ? set(icf, zero, add) : set(zero, zero, add);
    if (A == 2 && B == 1) return Dd == 0 ? set(one, cf, sub) : Dd == 1 ? set(zero, icf, add) : set(zero, zero, add);
    ok = false;
    return false;
}

// colour write mask of the card from FBMSK; false: a mask that is not whole channels
bool color_mask(u32 fpsm, u32 fbmsk, u32 &mask) {
    mask = 0;
    if (is16(fpsm)) {
        for (int c = 0; c < 3; ++c) {
            const u32 m = (fbmsk >> (8 * c)) & 0xF8u;
            if (m == 0) mask |= 1u << c;
            else if (m != 0xF8u) return false;
        }
        if (!(fbmsk & 0x80000000u)) mask |= 8u;
        return true;
    }
    for (int c = 0; c < 4; ++c) {
        const u32 m = (fbmsk >> (8 * c)) & 0xFFu;
        if (m == 0) mask |= 1u << c;
        else if (m != 0xFFu) return false;
    }
    if (fpsm == 0x01) mask &= 7u;
    return true;
}

void fail(DrawState &s, Reason r) {
    s.skip = 2;
    s.reason = r;
}

// A draw reads a texture out of memory that a target holds in another format (the frame read as an 8-bit palette texture, a 128x160 target
// read as 8H or 16 bit, a 16 bit target read as 8 bit: Harry Potter 2's ground, characters and walls do this to every frame). The card cannot
// sample a target as another format, so the target is written back to graphics memory (the same store as a display or a save state needs)
// and the texture is decoded from there like any texture. Done once per target until something is drawn into it again; the blocks it covers
// get new versions then, so a texture cached from the older picture is not taken for this one. PS2_HWR_NOSTORE=1 turns it off (the old
// behaviour: such shapes are not drawn).
// once_per_frame: a page written back already in this frame is not written back again (its later draws are not seen by this read). For a
// draw that reads the target it draws into through memory: Harry Potter 2's spell sparkles are drawn near the top-left corner of the frame and
// textured from address 0, the frame's own corner (64x32 pixels read as T4/T8), so every batch of them read what the batch before drew and
// broke the frame's render pass for a write-back, ~34 times a frame while a spell is cast. They now read the corner as it was at the
// first such read of the frame. PS2_HWR_STALE_SELF=0 turns it off.
u64 g_page_store_frame[kPages];
void store_for_memory(int owner, u32 tbp, u32 tbw, u32 tpsm, u32 tw_log, u32 th_log, bool once_per_frame) {
    Target &t = *g_targets[static_cast<std::size_t>(owner)];
    // the part of the target the texture reads: the pages it covers, as a rectangle of the target's own page grid
    const int pw = 64, ph = is16(t.psm) ? 64 : 32;
    const int first_page = static_cast<int>(t.bp / 32), per_row = static_cast<int>(std::max<u32>(1, t.bw));
    int rx0 = 1 << 30, ry0 = 1 << 30, rx1 = -1, ry1 = -1;
    bool any = false;
    // the pages the texture covers that something was drawn into since they were last written back: only those are stored (a texture whose
    // pages were not drawn into again keeps its decoded copy, the block versions stay)
    for_pages(tpsm, tbp, tbw, 0, 0, (1 << tw_log) - 1, (1 << th_log) - 1, [&](u32 page) {
        const u32 pg = page & (kPages - 1);
        any = true;
        if (g_page_write[pg] == g_page_stored[pg]) return;
        if (once_per_frame && g_page_store_frame[pg] == g_frames + 1) {
            ++F.approx[kStaleSelf];
            return;
        }
        const int rel = static_cast<int>(pg) - first_page;
        if (rel < 0) return;
        const int col = rel % per_row, row = rel / per_row;
        rx0 = std::min(rx0, col * pw), ry0 = std::min(ry0, row * ph);
        rx1 = std::max(rx1, col * pw + pw - 1), ry1 = std::max(ry1, row * ph + ph - 1);
    });
    if (!any || rx1 < 0) return;   // nothing new in the pages the texture reads
    const int vw = std::clamp<int>(t.valid_w, 1, static_cast<int>(t.w)), vh = std::clamp<int>(t.valid_h, 1, static_cast<int>(t.h));
    rx0 = std::max(rx0, 0), ry0 = std::max(ry0, 0), rx1 = std::min(rx1, vw - 1), ry1 = std::min(ry1, vh - 1);
    if (rx1 < rx0 || ry1 < ry0) return;   // outside what was drawn: memory holds what it holds
    for (int py = ry0 / ph; py <= ry1 / ph; ++py)
        for (int px = rx0 / pw; px <= rx1 / pw; ++px) {
            const u32 pg = static_cast<u32>(first_page + py * per_row + px) & (kPages - 1);
            g_page_stored[pg] = g_page_write[pg];
            g_page_store_frame[pg] = g_frames + 1;
        }
    const u32 x = static_cast<u32>(rx0), y = static_cast<u32>(ry0), w = static_cast<u32>(rx1 - rx0 + 1), h = static_cast<u32>(ry1 - ry0 + 1);
    Op &op = add_op(t.depth ? kOpStoreZ : kOpStore);
    const float scale = static_cast<float>(static_cast<double>(t.depth ? (t.psm == 0x30 ? 0xFFFFFFFFu : t.psm == 0x31 ? 0xFFFFFFu : 0xFFFFu) : 0u) + 1.0);
    u32 sbits;
    std::memcpy(&sbits, &scale, 4);
    const u32 words[10] = {t.bp, t.bw, t.depth ? t.psm : t.last_psm, x, y, w, h, sbits, fbits(t.sc.x), fbits(t.sc.y)};
    op.param = C.params(words, 10);
    op.group = t.s->sync_group;
    op.rd = t.s->id;
    op.tgt = owner;
    op.w = w, op.h = h;
    ++F.stores;
    if (prov::g_on) prov::count(prov::kStores), prov::count(prov::kStorePixels, static_cast<std::uint64_t>(w) * h);
    for_blocks(t.psm, t.bp, t.bw, static_cast<int>(x), static_cast<int>(y), static_cast<int>(x + w) - 1, static_cast<int>(y + h) - 1, [&](u32 blk, int, int, u32, u32) {
        g_block[blk] = mix(0x57C0ull, ++g_fresh);
        g_blk_stale[blk] = 0;
        g_page_stale[(blk / 32) & (kPages - 1)] = false;
        touch_page(blk / 32);
    });
    ++g_epoch;
}

bool v4_mipmaps();

void resolve_texture(DrawState &s, const u32 *w, const u32 *mip, u32 mip_top) {
    Source &src = s.src;
    const u32 flags = w[0], modes = w[7];
    const u32 tbp = w[12], tbw = w[13], tpsm = w[14];
    const u32 wms = (modes >> 24) & 3u, wmt = (modes >> 26) & 3u;
    const int owner = g_owner[(tbp / 32) & (kPages - 1)];
    bool from_stored = false;
    ++F.lookups;
    if (owner >= 0) {
        const Target &t = *g_targets[static_cast<std::size_t>(owner)];
        u32 view = 99;
        if (!t.depth && t.bp == tbp && t.bw == tbw) {
            if (t.psm == 0x00 && tpsm == 0x00) view = 0;
            else if (t.psm == 0x00 && tpsm == 0x01) view = 1;
            else if (is16(t.psm) && t.psm == tpsm) view = 2;
            // T8H is laid out as CT32 (the index is the top byte of each word): the draw reads the target's alpha and looks it up in the
            // palette slot of its setup (w[22]) itself. Harry Potter 2 does this ~100 times a frame (a 128x160 target, a shadow or glow
            // mask), each of them a write-back to memory, a decode and a copy before. CT32/CT24 palettes only; PS2_HWR_T8H=0 turns it off.
            static const bool t8h = os::env("PS2_HWR_T8H") != "0";
            if (t8h && C.clut && t.psm == 0x00 && tpsm == 0x1B && (w[16] == 0x00 || w[16] == 0x01)) view = 3;
        }
        // View 4: the draw reads the target through graphics memory addresses as another format (the frame read as T8 or T4, a 16-bit target
        // read as T8, a CT32 target read as CT16): the shader finds the target pixel that holds each texel's bits (hwr.wgsl view4), so the
        // target is not written back to memory, decoded and copied for it. Not for mipmapped textures, a target read by a draw into itself,
        // depth, or palettes in 16-bit formats. PS2_HWR_DIRECT=0 turns it off.
        static const bool direct = os::env("PS2_HWR_DIRECT") != "0";
        // PS2_HWR_DIRECT_MIP=1: a mipmapped texture is taken when its LOD can hardly leave level 0 (a fixed LOD of 0 or less, or a bias K of -4 or
        // less), read at level 0 (counted as "mip read at level 0"). Measured in Harry Potter 2: the textures at address 0 that the game reads this
        // way (K = -7.3, 16-28 a frame, more while a spell is cast) are mip chains it renders into the frame's corner; level 0 alone changed
        // ~17000 pixels of far geometry, and as reads of the target being drawn they cost more as copies than as write-backs. Off.
        static const bool direct_mip = os::env("PS2_HWR_DIRECT_MIP") == "1";  // off: those textures are mip chains the game renders itself, level 0 alone is visibly wrong
        const u32 m0 = mip[0];
        const int lod_k = static_cast<int>(static_cast<int32_t>(((m0 >> 16) & 0xFFFu) << 20) >> 20);  // 1/16 units
        // a mipmapped texture: every level read through view 4 (hwr.wgsl sample_v4_mip; PS2_HWR_DIRECT_MIPMAP=0 turns it off)
        const bool level0 = mip_top == 0 || v4_mipmaps() || (direct_mip && ((m0 & 1u) ? lod_k <= 0 : lod_k <= -64));
        // a draw into the target it reads (PS2_HWR_DIRECT_SELF=1): a copy of the whole target per read is dearer than the old write-back of the pages read
        // opt-in (1, nomip or mip): no gain measured (same passes and decodes as the write-back path)
        static const std::string self_mode = os::env("PS2_HWR_DIRECT_SELF");  // 0 off, nomip: only textures without levels, mip: only mipmapped ones (test switch)
        const bool direct_self = (self_mode == "1" || self_mode == "nomip" || self_mode == "mip") && (self_mode != "nomip" || mip_top == 0) && (self_mode != "mip" || mip_top != 0);
        // Only for targets at the GS's own size: above it (PS2_HWR_SCALE > 1) Harry Potter 2's sun haze (a 16-bit effect buffer read as T8 at
        // 2F20/3020) came out much weaker than through the write-back, which takes the texel under each GS pixel the same way; not understood
        // yet, so a scaled target takes the old way (PS2_HWR_DIRECT_SCALED=1 keeps view 4 there).
        // PS2_HWR_DIRECT_SCALED: 1 = every scaled target too, psm = scaled targets but the 16-bit (CT16S) ones, unset = none (test switch)
        static const std::string scaled_mode = os::env("PS2_HWR_DIRECT_SCALED");
        const bool direct_scaled = scaled_mode == "1" || (scaled_mode == "psm" && t.psm != 0x0A);
        const bool unscaled = t.sc.x == 1.0f && t.sc.y == 1.0f;
        if (view == 99 && direct && !g_no_view4 && (unscaled || direct_scaled) && C.clut && g_addr && !t.depth && level0 && (owner != s.color || direct_self) && (t.psm == 0x00 || t.psm == 0x02 || t.psm == 0x0A)) {
            if (mip_top && !v4_mipmaps()) ++F.approx[kMipLevel0];
            const bool indexed = tpsm == 0x13 || tpsm == 0x14;
            bool ok = (indexed && (w[16] == 0x00 || w[16] == 0x01)) || tpsm == 0x02 || tpsm == 0x0A || ((tpsm == 0x00 || tpsm == 0x01) && t.psm == 0x00);
            // Every page the texture can read must be this target's: the shader reads all its texels from this one target. Harry Potter 2's
            // sun haze reads 2F20 as T8, whose pages are shared by the 704-wide 16-bit effect buffer and a 64-wide blur target (and memory):
            // taken from the first page's owner alone, the haze came out at about 80 % of its brightness (garden, frame 9603).
            static const bool all_pages = os::env("PS2_HWR_DIRECT_PAGES") == "full";
            if (ok && all_pages) {
                const int tw = 1 << (w[15] & 0xFFu), th = 1 << (w[15] >> 8);
                std::string owners;
                static const bool log_owners = os::env("PS2_HWR_LOG") == "6";
                for_pages(tpsm, tbp, tbw, 0, 0, tw - 1, th - 1, [&](u32 page) {
                    const int o = g_owner[page & (kPages - 1)];
                    if (o != owner) ok = false;
                    if (log_owners) owners += std::to_string(page) + ":" + std::to_string(o) + " ";
                });
                if (log_owners && !ok) {
                    static int n = 0;
                    if (n++ < 6) std::fprintf(stderr, "[hwr-owners] tex %04X/%u psm %02X %dx%d owner t%d (bp %04X bw %u psm %02X %ux%u): %s\n", tbp, tbw, tpsm, tw, th, owner, t.bp, t.bw, t.psm, t.w, t.h, owners.c_str());
                }
            }
            if (ok) view = 4;
        }
        if (view == 99) {
            // PS2_HWR_LOG=2: which target/texture format pairs end here (target bp/bw/psm and the texture's tbp/tbw/tpsm, with how many shapes)
            static const bool log_pairs = os::env("PS2_HWR_LOG") == "2";
            if (log_pairs) {
                static std::unordered_map<u64, u32> seen;
                const u64 key = (static_cast<u64>(t.bp) << 40) ^ (static_cast<u64>(t.bw) << 32) ^ (static_cast<u64>(t.psm) << 24) ^ (static_cast<u64>(t.depth) << 23) ^
                                (static_cast<u64>(tbp) << 12) ^ (static_cast<u64>(tbw) << 6) ^ tpsm;
                const u32 n = ++seen[key];
                if (n == 1 || n == 1000 || n == 100000)
                    std::fprintf(stderr, "[hwr-pair] target bp %04X bw %u psm %02X%s (%ux%u) read as texture tbp %04X tbw %u tpsm %02X (shape count so far %u)\n", t.bp, t.bw, t.psm,
                                 t.depth ? " DEPTH" : "", t.w, t.h, tbp, tbw, tpsm, n);
            }
            static const bool no_store = os::env("PS2_HWR_NOSTORE") == "1";
            if (no_store) {
                fail(s, kTexTarget);
                return;
            }
            static const bool stale_self = os::env("PS2_HWR_STALE_SELF") != "0";
            store_for_memory(owner, tbp, tbw, tpsm, w[15] & 0xFFu, w[15] >> 8, stale_self && owner == s.color);
            from_stored = true;
            s.no_memo = true;
            goto from_memory;
        }
        ++F.target_reads;
        src.kind = 2;
        src.view = view;
        src.target = owner;
        src.group = t.s->sample_group;
        src.res = t.s->id;
        s.self_read = owner == s.color;
        return;
    }
from_memory:
    bool stale = false;
    const u64 key = texture_key(w, mip, mip_top, stale);
    if (stale) ++F.approx[kStaleTex];
    TexEntry *e;
    if (const auto it = g_cache.find(key); it != g_cache.end()) {
        e = it->second.get();
        ++F.hits;
    } else {
        e = decode_texture(key, w, mip, mip_top);
        // PS2_HWR_LOG=4: every texture decoded (a cache miss) with what it is, to see why a frame decodes what it decodes
        static const bool log_miss = os::env("PS2_HWR_LOG") == "4";
        if (log_miss)
            std::fprintf(stderr, "[hwr-miss] frame %llu tbp %04X tbw %u tpsm %02X %ux%u levels %u cbp %04X cpsm %02X%s%s\n", static_cast<unsigned long long>(g_frames), w[12], w[13], w[14],
                         1u << (w[15] & 0xFFu), 1u << (w[15] >> 8), e->levels, w[16] & 0x3FFFu, (w[16] >> 14) & 0x3Fu, from_stored ? " FROM-TARGET" : "", stale ? " STALE" : "");
        if (log_miss)
            std::fprintf(stderr, "[hwr-miss-mip] mip0 %08X flags %05X modes %08X top %u levels %04X/%u %04X/%u %04X/%u owner %d color %d page owners %d %d %d\n", mip[0], w[0], w[7], mip_top,
                         mip[1] & 0x3FFFu, (mip[1] >> 14) & 0x3Fu, mip[2] & 0x3FFFu, (mip[2] >> 14) & 0x3Fu, mip[3] & 0x3FFFu, (mip[3] >> 14) & 0x3Fu, owner, s.color,
                         g_owner[(mip[1] & 0x3FFFu) / 32 & (kPages - 1)], g_owner[(mip[2] & 0x3FFFu) / 32 & (kPages - 1)], g_owner[(mip[3] & 0x3FFFu) / 32 & (kPages - 1)]);
    }
    e->last_frame = g_frames;
    if (from_stored) e->transient = true;
    const bool manual = wms >= 2 || wmt >= 2;
    if (manual && e->levels > 1) ++F.approx[kRegionMip];
    u32 si = 0;
    if (mip[0]) {
        const u32 m = mip[0], mmag = (m >> 4) & 1u, mmin = (m >> 5) & 7u;
        si = mmag | ((mmin == 1 || mmin >= 4) ? 2u : 0u) | ((mmin == 3 || mmin == 5) ? 4u : 0u);
    } else if (flags & 16384u) {
        si = 3;
    }
    if (wms == 1) si |= 8;
    if (wmt == 1) si |= 16;
    if (!e->groups[si]) e->groups[si] = tex_group(e->view, si);
    src.kind = manual ? 2 : 1;
    src.view = 0;
    src.group = e->groups[si];
    src.res = e->id;
    src.levels = e->levels;
}

void build_state(DrawState &s, const State &st, const u32 *w, const u32 *mip, u32 mip_top, u32 prim) {
    s.skip = 0;
    s.src = Source{};
    s.self_read = false;
    s.no_memo = false;
    s.color = s.depth = -1;
    const u32 flags = w[0], fbp = w[1], fbw = w[2], fpsm = w[3], fbmsk = w[4], zbp = w[5], zpsm = w[6], modes = w[7];
    if (prim < 3) return fail(s, kPrimKind);
    if (flags & 128u) {
        static const bool log_date = os::env("PS2_HWR_LOG") == "3";
        if (log_date) {
            static std::unordered_map<u32, u32> seen;
            const u32 key = fpsm | (modes & 0xFF) << 8 | (flags & 0xFFFFu) << 16;
            const u32 n = ++seen[key];
            if (n == 1 || n == 1000 || n == 100000) std::fprintf(stderr, "[hwr-date] fpsm %02X DATM %u flags %05X modes %08X zte %d (count %u)\n", fpsm, (modes >> 20) & 1u, flags, modes, (flags & 512u) ? 1 : 0, n);
        }
        return fail(s, kDate);
    }
    if (fpsm != 0x00 && fpsm != 0x01 && fpsm != 0x02 && fpsm != 0x0A) return fail(s, kFramePsm);
    if (fbw == 0) return fail(s, kZeroWidth);
    const bool zte = flags & 512u;
    const u32 ztst = (modes >> 8) & 3u;
    if (zte && ztst == 0) {
        s.skip = 1;
        return;
    }
    u32 cmask;
    if (!color_mask(fpsm, fbmsk, cmask)) return fail(s, kFbmsk);
    const bool zwrite = zte && !(flags & 32u);
    // alpha test: what passing and failing pixels write (docs/hw-renderer-design.md 3.5: NEVER is a pure write mask, a partial failure two draws)
    u32 pass_mask = cmask, fail_mask = 0;
    bool pass_z = zwrite, fail_z = false;
    int split = 0;  // 0 one draw without test, 1 one draw dropping failures, 2 two draws
    const bool ate = flags & 64u;
    const u32 atst = modes & 7u, afail = (modes >> 4) & 3u;
    if (ate && atst != 1) {
        switch (afail) {
        case 0: fail_mask = 0, fail_z = false; break;
        case 1: fail_mask = cmask, fail_z = false; break;
        case 2: fail_mask = 0, fail_z = zwrite; break;
        default: fail_mask = cmask & 7u, fail_z = false; break;
        }
        if (atst == 0) pass_mask = fail_mask, pass_z = fail_z;
        else if (fail_mask == pass_mask && fail_z == pass_z) split = 0;
        else if (fail_mask == 0 && !fail_z) split = 1;
        else split = 2;
    }
    if (!pass_mask && !pass_z && !(split == 2 && (fail_mask || fail_z))) {
        s.skip = 1;
        return;
    }
    // blending
    PipeKey base;
    const bool abe = flags & 8u;
    bool uses_as = false;
    u32 premul = 0;
    if (abe) {
        if (flags & 1024u) return fail(s, kPabe);
        const u32 A = (modes >> 12) & 3u, B = (modes >> 14) & 3u, Cc = (modes >> 16) & 3u, Dd = (modes >> 18) & 3u;
        bool ok;
        const bool alpha_written = (pass_mask | fail_mask) & 8u;
        blend_map(A, B, Cc, Dd, C.dual, alpha_written, base, uses_as, ok, premul);
        if (!ok) {
            static const bool log_blend = os::env("PS2_HWR_LOG") == "3";   // PS2_HWR_LOG=3: the blend forms (A B C D) and formats that are not drawn, with counts
            if (log_blend) {
                static std::unordered_map<u32, u32> seen;
                const u32 key = A | B << 2 | Cc << 4 | Dd << 6 | fpsm << 8 | (alpha_written ? 1u << 16 : 0u) | (w[9] & 0xFFu) << 17;
                const u32 n = ++seen[key];
                if (n == 1 || n == 1000 || n == 100000)
                    std::fprintf(stderr, "[hwr-blend] A %u B %u C %u D %u fix %02X fpsm %02X alpha written %d (count %u)\n", A, B, Cc, Dd, w[9] & 0xFFu, fpsm, alpha_written ? 1 : 0, n);
            }
            return fail(s, Cc == 0 && !C.dual ? kNoDual : kBlend);
        }
        if (!(flags & 2048u)) ++F.approx[kColclamp];
        base.dual = uses_as && C.dual;
    }
    s.blend = std::min<float>(static_cast<float>(w[9]) / 128.0f, 1.0f);
    // targets
    // A new colour target is as tall as the scissor of its first draw (rounded up to 32 rows; 512 with PS2_HWR_FULL_HEIGHT=1): the display
    // buffer of a field game holds 256 rows, its frame 384, and every card texel of a render pass is loaded and stored by a tile GPU
    // whether it is drawn or not. A later draw below it makes it 512 rows (grow()).
    static const bool full_height = os::env("PS2_HWR_FULL_HEIGHT") == "1";
    // At most kMaxRows (576): a PAL game's whole frame (Harry Potter 2 draws 704 x 576 and shrinks it into its 640 x 512 display buffer; at 512
    // rows its bottom 64 were missing and the shrink repeated the last row over the display's rows 456-511)
    const u32 used = std::clamp<u32>((w[11] >> 16) + 1u, 64u, kMaxRows);
    const u32 W = std::max<u32>(64, fbw * 64), H = full_height ? std::max(512u, (used + 31u) & ~31u) : std::min<u32>(kMaxRows, (used + 31u) & ~31u);
    const u32 cclass = is16(fpsm) ? fpsm : 0x00;
    Sc csc = base_scale(W);
    if (g_rows > 0.0f && W >= kSmallWidth) csc.y = std::min(csc.y, g_rows / static_cast<float>(used));
    // 16-bit colour targets stay at the GS's own size (PS2_HWR_NATIVE16=0: scaled like the others). Games read them back as other formats
    // (Harry Potter 2's sun haze: a 704-wide CT16S buffer read as T8, each pixel's bytes palette indices), where an upscaled picture's
    // slightly different values pick other palette entries: at 2x the haze came out at 52 against 62.6 in the reference renderer. They are
    // effect buffers (blurred on screen anyway), so 1x is also less card work.
    static const bool native16 = os::env("PS2_HWR_NATIVE16") != "0";
    if (native16 && cclass != 0x00) csc = Sc{1.0f, 1.0f};
    s.color = find_target(fbp, fbw, cclass, false, W, H, csc);
    g_targets[static_cast<std::size_t>(s.color)]->last_psm = fpsm;
    s.zmax = w[23];
    s.zscale = static_cast<double>(s.zmax) + 1.0;
    s.needs_depth = zte && (ztst != 1 || pass_z || fail_z);
    if (s.needs_depth) {
        // A depth buffer drawn with colour targets of two scales (the frame and the display copy share Z 1400 here): a draw that only writes
        // depth (no test) into one held at another scale drops its depth writes, instead of making a second buffer that would evict the
        // first one every frame. A draw that tests depth gets its own buffer at its scale.
        const Target &ctg = *g_targets[static_cast<std::size_t>(s.color)];
        const Sc &csc2 = ctg.sc;
        const u32 CH = ctg.h;
        bool other = false;
        for (const auto &t : g_targets)
            if (t && t->alive && t->depth && t->bp == zbp && t->bw == fbw && t->psm == zpsm && t->w == W && !(t->sc == csc2 && t->h == CH)) other = true;
        if (other && ztst == 1) {
            s.needs_depth = false;
            ++F.approx[kDepthScale];
        } else {
            s.depth = find_target(zbp, fbw, zpsm, true, W, CH, csc2);
        }
    }
    s.cwrite_any = (pass_mask | fail_mask) != 0;
    s.awrite_any = ((pass_mask | fail_mask) & 8u) != 0;
    s.zwrite_any = pass_z || fail_z;
    s.color_full = pass_mask == (fpsm == 0x01 ? 7u : 15u) && split == 0;
    // texture
    if (flags & 2u) {
        resolve_texture(s, w, mip, mip_top);
        if (s.skip) return;
    }
    // per-draw parameters for the fragment program
    std::array<u32, 16> &p = s.pw;
    p.fill(0);
    u32 f = 0;
    if ((flags & 2u) && s.src.kind) f |= 1u;
    if (flags & 4u) f |= 2u;
    if (flags & 16u) f |= 4u;
    if (flags & 8192u) f |= 8u;
    if (flags & 16384u) f |= 16u;
    if (flags & 32768u) f |= 32u;
    if (flags & 4096u) f |= 64u;
    if (is16(fpsm)) f |= 128u;
    if ((s.src.kind == 1 || (s.src.view == 4 && mip_top)) && mip[0]) f |= 256u;  // F_MIP: the decoded texture's levels, or view 4's (sample_v4_mip)
    if (uses_as && !C.dual) f |= 512u;
    if (premul) f |= premul == 1 ? 1024u : 2048u;
    if (is16(fpsm) && (!base.blend || premul)) f |= premul == 2 ? 8192u : 4096u;
    p[0] = f;
    p[1] = ((modes >> 20) & 3u) | (atst << 2) | (((modes >> 24) & 3u) << 8) | (((modes >> 26) & 3u) << 10) | (s.src.view << 12);
    p[2] = w[8];
    p[3] = w[21];
    p[4] = w[15];
    p[5] = w[18];
    p[6] = w[19];
    p[7] = w[20];
    p[8] = mip[0];
    p[9] = s.src.levels ? s.src.levels - 1 : 0;
    p[10] = w[9];  // FIX
    const Sc src_sc = s.src.kind == 2 && s.src.target >= 0 ? g_targets[static_cast<std::size_t>(s.src.target)]->sc : Sc{};
    const bool scaled = (src_sc.x != 1.0f || src_sc.y != 1.0f) && s.src.view != 4;  // view 4 reads GS texels and scales them itself
    p[11] = scaled ? fbits(src_sc.x) : 0u;  // a target source: its card texels per GS texel across and down (f32 bits; 0 = not scaled)
    p[12] = scaled ? fbits(src_sc.y) : 0u;
    // T8H from a target: the palette slot (PS2_HWR_T8H=mark: a test aid, those texels are drawn magenta to see where they are)
    static const bool t8h_mark = os::env("PS2_HWR_T8H") == "mark";
    // PS2_HWR_LOG=5: where the draws that read a target as T8H (view 3) or through memory addresses (view 4) write, once per kind
    static const bool log_view = os::env("PS2_HWR_LOG") == "5";
    if (log_view && s.src.view >= 3) {
        static std::unordered_map<u64, u32> seen;
        const Target &dt = *g_targets[static_cast<std::size_t>(s.color)];
        const u64 key = static_cast<u64>(s.src.view) | static_cast<u64>(dt.bp) << 4 | static_cast<u64>(fbmsk) << 20 | static_cast<u64>(flags & 0xFFFFu) << 52;
        if (++seen[key] == 1)
            std::fprintf(stderr, "[hwr-view] view %u reads target %d (bp %04X psm %02X) as tpsm %02X, writes target bp %04X psm %02X %ux%u fbmsk %08X flags %05X modes %08X\n", s.src.view,
                         s.src.target, g_targets[static_cast<std::size_t>(s.src.target)]->bp, g_targets[static_cast<std::size_t>(s.src.target)]->psm, w[14], dt.bp, dt.psm, dt.w, dt.h, fbmsk, flags, modes);
    }
    p[13] = s.src.view == 3 ? (t8h_mark ? 0xFFFFFFFFu : w[22]) : 0u;
    if (s.src.view == 4) {  // view 4: the palette slot and CSA, the texture's and the target's base, width and format, the target's scale
        const Target &st4 = *g_targets[static_cast<std::size_t>(s.src.target)];
        static const bool v4_mark = os::env("PS2_HWR_DIRECT") == "mark";
        p[13] = v4_mark ? 0xFFFFFFFFu : (w[22] & 0x7FFu) | (w[17] & 31u) << 11;
        p[14] = (w[12] & 0x3FFFu) | (w[13] & 63u) << 14 | (w[14] & 63u) << 20;
        p[15] = (st4.bp & 0x3FFFu) | (st4.bw & 63u) << 14 | (st4.psm & 63u) << 20;
        p[12] = fbits(src_sc.x);
        p[9] = fbits(src_sc.y);
    }
    // pipelines
    const u32 tk = s.src.kind == 1 ? 1u : s.src.kind == 2 ? 2u : 0u;
    PipeKey k = base;
    k.tex = tk;
    k.depth = false;  // set per pass (encode time decides from the pass's attachments)
    k.zcmp = !zte || ztst == 1 ? WGPUCompareFunction_Always : ztst == 2 ? WGPUCompareFunction_GreaterEqual : WGPUCompareFunction_Greater;
    s.k1 = k;
    s.k1.mask = pass_mask;
    s.k1.zwrite = pass_z;
    s.k1.atest = split == 0 ? 0u : 1u;
    s.k2 = k;
    s.k2.mask = fail_mask;
    s.k2.zwrite = fail_z;
    s.k2.atest = 2;
    if (!s.needs_depth) s.k1.zwrite = s.k2.zwrite = false;  // its depth writes dropped (a depth buffer held at another scale)
    if (split != 2) s.k2 = PipeKey{}, s.k2.mask = 0xFFu;  // marks "no second draw"
    if (!pass_mask && !pass_z) s.k1.mask = 0xFEu;          // the first draw writes nothing: only the second one is drawn
    s.kp1 = s.k1.mask == 0xFEu ? 0 : s.k1.pack() | kKeyOn;
    s.kp2 = s.k2.mask == 0xFFu ? 0 : s.k2.pack() | kKeyOn;
    s.zread = s.needs_depth && s.k1.zcmp != WGPUCompareFunction_Always;
    (void)st;
}

// the node (run of draws into one colour surface) that the next draw goes into: the open one when it has the same colour surface and no
// other depth surface (a node without depth takes the draw's depth surface; the render pass attaches it from the start)
void ensure_pass(DrawState &s) {
    Target &ct = *g_targets[static_cast<std::size_t>(s.color)];
    Surf *ds = s.needs_depth ? g_targets[static_cast<std::size_t>(s.depth)]->s : nullptr;
    if (g_open_pass >= 0) {
        Op &op = g_ops[static_cast<std::size_t>(g_open_pass)];
        if (op.cs == ct.s && (!ds || !op.ds || op.ds == ds)) {
            if (ds) op.ds = ds;
            return;
        }
    }
    Op &op = add_op(kOpPass);
    op.cs = ct.s;
    op.ds = ds;
    op.tgt = s.color;
    op.first_draw = static_cast<u32>(g_draws.size());
    g_open_pass = static_cast<int>(g_ops.size() - 1);
}

// what a shape drew into a target: a draw that reads a copy of the target made before this is out of date where they overlap
void add_dirty(std::vector<Rect> &list, const Rect &r) {
    // A rectangle that continues one already listed (the same columns and the next rows, or the same rows and the next columns) grows it: a
    // game that moves a picture by a strip at a time (Harry Potter 2 shifts its 704x512 effect buffer up 2 rows with 1408 strips of 64x2)
    // keeps one rectangle per column, where the bounding box of everything used to cover rows not written yet and force a copy per column.
    // PS2_HWR_DIRTY_MERGE=0 turns it off.
    static const bool merge = os::env("PS2_HWR_DIRTY_MERGE") != "0";
    if (merge)
        for (Rect &d : list) {
            if (d.x0 == r.x0 && d.x1 == r.x1 && r.y0 <= d.y1 + 1 && d.y0 <= r.y1 + 1) {
                d.y0 = std::min(d.y0, r.y0), d.y1 = std::max(d.y1, r.y1);
                return;
            }
            if (d.y0 == r.y0 && d.y1 == r.y1 && r.x0 <= d.x1 + 1 && d.x0 <= r.x1 + 1) {
                d.x0 = std::min(d.x0, r.x0), d.x1 = std::max(d.x1, r.x1);
                return;
            }
        }
    if (merge && list.size() >= 32) {
        // full: rectangles over the same columns become one first (a column of strips), so that the rest keep their own places
        std::vector<Rect> grouped;
        for (const Rect &d : list) {
            bool joined = false;
            for (Rect &g : grouped)
                if (g.x0 == d.x0 && g.x1 == d.x1) {
                    g.y0 = std::min(g.y0, d.y0), g.y1 = std::max(g.y1, d.y1);
                    joined = true;
                    break;
                }
            if (!joined) grouped.push_back(d);
        }
        list.swap(grouped);
    }
    if (list.size() >= 32) {
        Rect &b = list[0];
        for (const Rect &d : list) b.x0 = std::min(b.x0, d.x0), b.y0 = std::min(b.y0, d.y0), b.x1 = std::max(b.x1, d.x1), b.y1 = std::max(b.y1, d.y1);
        list.resize(1);
    }
    list.push_back(r);
}

void note_write(Target &t, int x0, int y0, int x1, int y1, bool alpha) {
    ++t.writes;
    if (!t.snap) return;
    const Rect r{x0, y0, x1, y1};
    add_dirty(t.dirty, r);
    if (alpha) add_dirty(t.dirty_a, r);
}

// a draw that reads the target it draws into reads a copy of it, made before the draw (outside the render pass). The copy is made again only
// when something was drawn into the rectangle the draw reads since the last copy (the glow ladder reads one part and writes another).
// alpha_only: the draw reads only the copy's alpha (a T8H palette conversion, view 3, that writes the colour of the same target with its
// alpha masked: Harry Potter 2 draws ~200 of them a frame). Writes that left alpha alone do not make the copy old for it, so the draws
// share one copy and one render pass instead of a copy and a pass break each. PS2_HWR_ALPHA_SNAP=0 turns it off.
WGPUBindGroup self_copy(int ti, const Rect &r, u64 &res, bool alpha_only, bool frame_stale = false) {
    Target &t = *g_targets[static_cast<std::size_t>(ti)];
    static const bool alpha_snap = os::env("PS2_HWR_ALPHA_SNAP") != "0";
    // frame_stale (a view-4 read of the target being drawn, r page-aligned): page by page as the write-back path reads its target
    // (store_for_memory once_per_frame): a page's part of the copy is brought up to date at its first read in a frame and read as it is for
    // the rest of that frame (counted "own target read as of earlier in the frame")
    if (frame_stale) {
        u32 pw, ph;
        page_dims(t.psm, pw, ph);
        const u32 per_row = std::max<u32>(1, t.w / pw), rows = (t.h + ph - 1) / ph;
        if (t.page_snap_frame.size() != static_cast<std::size_t>(per_row) * rows) t.page_snap_frame.assign(static_cast<std::size_t>(per_row) * rows, 0);
        const bool have = t.snap && t.snap_of == t.s;
        if (!have) {  // a new copy of the whole picture as it is now; its pages count as brought up to date when they are first read
            drop_snap(t);
            t.snap = get_surf(t.w, t.h, false, t.sc);
            t.snap_of = t.s;
            Op &op = add_op(kOpSelfCopy);
            op.tex = t.s->tex;
            op.tex2 = t.snap->tex;
            op.w = t.w, op.h = t.h;
            op.sc = t.sc;
            op.rd = t.s->id;
            op.wr = t.snap->id;
            op.tgt = ti;
            ++F.self_copies;
            F.copy_bytes += static_cast<u64>(card_x(t.w, t.sc)) * card_y(t.h, t.sc) * 4;
            t.dirty.clear(), t.dirty_a.clear();
            std::fill(t.page_snap_frame.begin(), t.page_snap_frame.end(), 0);
        }
        bool stale = false;
        const int x0 = std::max(r.x0, 0), y0 = std::max(r.y0, 0);
        const int x1 = std::min(r.x1, static_cast<int>(per_row * pw) - 1), y1 = std::min(r.y1, static_cast<int>(rows * ph) - 1);
        for (int py = y0 / static_cast<int>(ph); py <= y1 / static_cast<int>(ph); ++py)
            for (int px = x0 / static_cast<int>(pw); px <= x1 / static_cast<int>(pw); ++px) {
                u64 &f = t.page_snap_frame[static_cast<std::size_t>(py) * per_row + static_cast<std::size_t>(px)];
                if (f == g_frames + 1) {
                    stale = true;
                    continue;
                }
                f = g_frames + 1;
                if (!have) continue;  // the whole copy just made holds it as it is now
                const u32 gx = static_cast<u32>(px) * pw, gy = static_cast<u32>(py) * ph;
                const u32 cx0 = card_at_x(gx, t.sc), cy0 = card_at_y(gy, t.sc);
                const u32 cx1 = std::min(card_at_x(gx + pw, t.sc), card_x(t.w, t.sc)), cy1 = std::min(card_at_y(gy + ph, t.sc), card_y(t.h, t.sc));
                if (cx1 <= cx0 || cy1 <= cy0) continue;
                Op &op = add_op(kOpSelfCopy);
                op.tex = t.s->tex;
                op.tex2 = t.snap->tex;
                op.level = 1;
                op.offset = cx0, op.pitch = cy0;
                op.w = cx1 - cx0, op.h = cy1 - cy0;
                op.sc = t.sc;
                op.rd = t.s->id;
                op.wr = t.snap->id;
                op.tgt = ti;
                ++F.self_copies;
                F.copy_bytes += static_cast<u64>(op.w) * op.h * 4;
            }
        if (stale) ++F.approx[kStaleSelf];
        res = t.snap->id;
        return t.snap->sample_group;
    }
    bool fresh = t.snap && t.snap_of == t.s;
    if (fresh)
        for (const Rect &d : alpha_only && alpha_snap ? t.dirty_a : t.dirty)
            if (d.hits(r)) {
                fresh = false;
                break;
            }
    // A copy that exists already is brought up to date only where the draw reads (the rectangle r): Harry Potter 2 shifts its 704x512 effect
    // buffer with a draw per 64-pixel column that reads the column before, a whole copy per column before (15 MB a frame). Written rectangles
    // that lie inside r are no longer dirty; the others stay (they still differ from the copy). PS2_HWR_PARTIAL_COPY=0 turns it off.
    static const bool partial = os::env("PS2_HWR_PARTIAL_COPY") != "0";
    const int vw = std::clamp<int>(t.valid_w, 1, static_cast<int>(t.w)), vh = std::clamp<int>(t.valid_h, 1, static_cast<int>(t.h));
    Rect pr{std::max(r.x0, 0), std::max(r.y0, 0), std::min(r.x1, vw - 1), std::min(r.y1, vh - 1)};
    // the part copied reaches over all the rows of the written rectangles it meets, so that they can be cut back to what lies beside it
    // (one copy for a column of strips, not one per strip)
    if (!fresh && partial)
        for (const Rect &d : alpha_only && alpha_snap ? t.dirty_a : t.dirty)
            if (d.hits(pr)) pr.y0 = std::max(0, std::min(pr.y0, d.y0)), pr.y1 = std::min(vh - 1, std::max(pr.y1, d.y1));
    if (!fresh && partial && t.snap && t.snap_of == t.s && pr.x0 <= pr.x1 && pr.y0 <= pr.y1 &&
        static_cast<long>(pr.x1 - pr.x0 + 1) * (pr.y1 - pr.y0 + 1) * 2 < static_cast<long>(vw) * vh) {
        const u32 cx0 = card_at_x(static_cast<u32>(pr.x0), t.sc), cy0 = card_at_y(static_cast<u32>(pr.y0), t.sc);
        const u32 cx1 = std::min(card_at_x(static_cast<u32>(pr.x1 + 1), t.sc), card_x(t.w, t.sc)), cy1 = std::min(card_at_y(static_cast<u32>(pr.y1 + 1), t.sc), card_y(t.h, t.sc));
        if (cx1 > cx0 && cy1 > cy0) {
            Op &op = add_op(kOpSelfCopy);
            op.tex = t.s->tex;
            op.tex2 = t.snap->tex;
            op.level = 1;  // a part: offset, pitch are its card origin, w, h its card size
            op.offset = cx0, op.pitch = cy0;
            op.w = cx1 - cx0, op.h = cy1 - cy0;
            op.sc = t.sc;
            op.rd = t.s->id;
            op.wr = t.snap->id;
            op.tgt = ti;
            ++F.self_copies;
            F.copy_bytes += static_cast<u64>(op.w) * op.h * 4;
            const auto inside = [&](const Rect &d) { return d.x0 >= pr.x0 && d.x1 <= pr.x1 && d.y0 >= pr.y0 && d.y1 <= pr.y1; };
            t.dirty.erase(std::remove_if(t.dirty.begin(), t.dirty.end(), inside), t.dirty.end());
            t.dirty_a.erase(std::remove_if(t.dirty_a.begin(), t.dirty_a.end(), inside), t.dirty_a.end());
            // what is left of a written rectangle that reaches into r: its part outside r is enough (cut along the side it crosses)
            for (std::vector<Rect> *list : {&t.dirty, &t.dirty_a})
                for (Rect &d : *list) {
                    if (!d.hits(pr)) continue;
                    if (d.y0 >= pr.y0 && d.y1 <= pr.y1) {  // spans r's rows: keep the columns outside r on one side
                        if (d.x0 < pr.x0 && d.x1 <= pr.x1) d.x1 = pr.x0 - 1;
                        else if (d.x1 > pr.x1 && d.x0 >= pr.x0) d.x0 = pr.x1 + 1;
                    } else if (d.x0 >= pr.x0 && d.x1 <= pr.x1) {  // spans r's columns: keep the rows outside r on one side
                        if (d.y0 < pr.y0 && d.y1 <= pr.y1) d.y1 = pr.y0 - 1;
                        else if (d.y1 > pr.y1 && d.y0 >= pr.y0) d.y0 = pr.y1 + 1;
                    }
                }
            res = t.snap->id;
            return t.snap->sample_group;
        }
    }
    if (!fresh) {
        drop_snap(t);
        t.snap = get_surf(t.w, t.h, false, t.sc);
        t.snap_of = t.s;
        const u32 w = static_cast<u32>(std::clamp<int>(t.valid_w, 1, static_cast<int>(t.w))), h = static_cast<u32>(std::clamp<int>(t.valid_h, 1, static_cast<int>(t.h)));
        Op &op = add_op(kOpSelfCopy);
        op.tex = t.s->tex;
        op.tex2 = t.snap->tex;
        op.w = w;
        op.h = h;
        op.sc = t.sc;
        op.rd = t.s->id;
        op.wr = t.snap->id;
        op.tgt = ti;
        ++F.self_copies;
        F.copy_bytes += static_cast<u64>(card_x(w, t.sc)) * card_y(h, t.sc) * 4;
    }
    res = t.snap->id;
    return t.snap->sample_group;
}

// the texels a shape can read from a target (texel loads, docs: TEX=2): from the corners' coordinates for FST shapes, else all of it
Rect read_rect(const u32 *w, const Vertex *v, int n, int tw, int th) {
    const u32 flags = w[0], modes = w[7];
    Rect r{0, 0, tw - 1, th - 1};
    if (!(flags & 16u)) return r;
    float u0 = 1e9f, u1 = -1e9f, v0 = 1e9f, v1 = -1e9f;
    for (int i = 0; i < n; ++i) {
        const float u = static_cast<float>(v[i].u) / 16.0f, vv = static_cast<float>(v[i].v) / 16.0f;
        u0 = std::min(u0, u), u1 = std::max(u1, u), v0 = std::min(v0, vv), v1 = std::max(v1, vv);
    }
    const bool bil = flags & 16384u;
    int a0 = static_cast<int>(std::floor(bil ? u0 - 0.5f : u0)), a1 = static_cast<int>(std::floor(bil ? u1 - 0.5f : u1)) + (bil ? 1 : 0);
    int b0 = static_cast<int>(std::floor(bil ? v0 - 0.5f : v0)), b1 = static_cast<int>(std::floor(bil ? v1 - 0.5f : v1)) + (bil ? 1 : 0);
    const auto wrap = [](int &lo, int &hi, u32 mode, u32 mn, u32 mx, int size) {
        if (mode == 0) {
            if (lo < 0 || hi >= size) lo = 0, hi = size - 1;
        } else if (mode == 1) {
            lo = std::clamp(lo, 0, size - 1), hi = std::clamp(hi, 0, size - 1);
        } else if (mode == 2) {
            lo = std::clamp(lo, static_cast<int>(mn), static_cast<int>(mx)), hi = std::clamp(hi, static_cast<int>(mn), static_cast<int>(mx));
        } else {
            lo = 0, hi = size - 1;
        }
    };
    const int sw = 1 << (w[15] & 0xFFu), sh = 1 << (w[15] >> 8);
    wrap(a0, a1, (modes >> 24) & 3u, w[18] & 0xFFFFu, w[18] >> 16, sw);
    wrap(b0, b1, (modes >> 26) & 3u, w[19] & 0xFFFFu, w[19] >> 16, sh);
    r.x0 = std::max(a0, 0), r.x1 = std::min(a1, tw - 1), r.y0 = std::max(b0, 0), r.y1 = std::min(b1, th - 1);
    return r;
}

// Two-pass draws (alpha test whose failures still write) draw all passing pixels of a merged draw, then all failing ones. That is the GS's
// order only when no two shapes of the draw cover the same pixel, so a draw takes a shape only when it overlaps none of the draw's shapes
// (separating axis test on the triangles; shapes that only touch along an edge or a corner share no pixel). At most kSplitTris per draw.
struct Tri {
    float x[3], y[3];
};
std::vector<Tri> g_split_tris;
constexpr std::size_t kSplitTris = 96;

bool separated(const Tri &a, const Tri &b) {
    // bounding boxes first (most pairs end here)
    const float ax0 = std::min({a.x[0], a.x[1], a.x[2]}), ax1 = std::max({a.x[0], a.x[1], a.x[2]}), bx0 = std::min({b.x[0], b.x[1], b.x[2]}), bx1 = std::max({b.x[0], b.x[1], b.x[2]});
    if (ax1 <= bx0 || bx1 <= ax0) return true;
    const float ay0 = std::min({a.y[0], a.y[1], a.y[2]}), ay1 = std::max({a.y[0], a.y[1], a.y[2]}), by0 = std::min({b.y[0], b.y[1], b.y[2]}), by1 = std::max({b.y[0], b.y[1], b.y[2]});
    if (ay1 <= by0 || by1 <= ay0) return true;
    for (const Tri *t : {&a, &b})
        for (int i = 0; i < 3; ++i) {
            const int j = (i + 1) % 3;
            const float nx = t->y[i] - t->y[j], ny = t->x[j] - t->x[i];
            const float eps = 1e-3f * (std::fabs(nx) + std::fabs(ny));
            float a0 = 1e30f, a1 = -1e30f, b0 = 1e30f, b1 = -1e30f;
            for (int k = 0; k < 3; ++k) {
                const float pa = a.x[k] * nx + a.y[k] * ny, pb = b.x[k] * nx + b.y[k] * ny;
                a0 = std::min(a0, pa), a1 = std::max(a1, pa), b0 = std::min(b0, pb), b1 = std::max(b1, pb);
            }
            if (a1 <= b0 + eps || b1 <= a0 + eps) return true;
        }
    return false;
}

u32 add_params(const std::array<u32, 16> &p) {
    if (!g_params.empty() && g_params.back() == p) return static_cast<u32>(g_params.size() - 1);
    g_params.push_back(p);
    return static_cast<u32>(g_params.size() - 1);
}

// a mipmapped view-4 draw: its record and, right after it, the record of its levels 1..6 (base | width << 14) and its top level (word 6)
// that hwr.wgsl sample_v4_mip reads (never shared with the record before: the shader takes the next record as the levels)
u32 add_mip_record(const std::array<u32, 16> &p, const u32 *mip, u32 mip_top, u32 mem_mask = 0) {
    g_params.push_back(p);
    std::array<u32, 16> x{};
    for (int n = 1; n <= 6; ++n) x[static_cast<std::size_t>(n - 1)] = mip[n];
    x[6] = mip_top;
    x[7] = mem_mask;  // bit n: level n from graphics memory (hwr.wgsl from_mem)
    g_params.push_back(x);
    return static_cast<u32>(g_params.size() - 2);
}

// PS2_HWR_DIRECT_MIPMAP=1: mipmapped textures are read through view 4 too (otherwise written back, decoded and copied level by level)
bool v4_mipmaps() {
    // opt-in (PS2_HWR_DIRECT_MIPMAP=1): measured in Harry Potter 2 (docs/debugging-native.md): the spell sparkles' lower levels lie in memory that the game
    // uploads again between draws, so no exact gain; levels inside the target are read exactly
    static const bool on = os::env("PS2_HWR_DIRECT_MIPMAP") == "1";
    return on;
}

u32 shape_params(const DrawState &s, const u32 *mip, u32 mip_top) {
    if (s.skip) return 0u;
    if (s.src.view == 4 && mip_top && mip[0]) return add_mip_record(s.pw, mip, mip_top);
    return add_params(s.pw);
}

// a run of clear strips being drawn (shape(): becomes a clear of the target when it covers it)
struct ClearRun {
    bool on = false, zw = false, tex = false;
    u64 serial = 0, submit = 0;
    int color = -1, next_x = 0, y0 = 0, y1 = 0;
    u32 rgba = 0, z = 0;
    std::size_t ops_n = 0, draws_n = 0, verts_n = 0;
    int open = -1;
    u32 open_draws = 0, last_count = 0;
};
ClearRun g_clear;

// every page the target holds lies in rows 0..y1 (a clear of those rows over the whole width overwrites all of its picture)
bool covers(int ti, int y1) {
    const Target &t = *g_targets[static_cast<std::size_t>(ti)];
    u32 pw, ph;
    page_dims(t.psm, pw, ph);
    const u32 per_row = std::max<u32>(1, t.bw * 64 / pw);
    for (u32 p = 0; p < kPages; ++p) {
        if (g_owner[p] != ti) continue;
        const u32 rel = (p - t.bp / 32) & (kPages - 1), py = rel / per_row;
        if (static_cast<int>((py + 1) * ph) - 1 > y1) return false;
    }
    return true;
}

void timer_init();

}  // namespace

// ================================================================ interface

bool init(const Context &ctx) {
    C = ctx;
    if (os::env("PS2_HWR_NODUAL") == "1") C.dual = false;   // test aid: the device behaves as one without dual-source blending (the Mali phones)
    g_s = std::max<u32>(ctx.scale, 1u);
    g_fx = ctx.target_scale > 0.0f ? std::clamp(ctx.target_scale, 0.25f, 6.0f) : static_cast<float>(g_s);
    const float fy = static_cast<float>(std::atof(os::env("PS2_HWR_SCALE_Y").c_str()));
    g_fy = fy > 0.0f ? std::clamp(fy, 0.25f, 6.0f) : g_fx;
    const std::string small = os::env("PS2_HWR_SMALL_SCALE");
    const float cap = small.empty() ? 2.0f : static_cast<float>(std::atof(small.c_str()));
    g_small = cap > 0.0f && g_fx > cap ? cap / g_fx : 1.0f;
    g_rows = std::max(0.0f, static_cast<float>(std::atof(os::env("PS2_HWR_ROWS").c_str())));
    std::fprintf(stderr, "[hwr] targets at %.3fx across, %.3fx down (card texels per GS pixel), narrower than %u at %.3fx, %.3fx; graphics memory at %ux\n", g_fx, g_fy,
                 kSmallWidth, g_fx * g_small, g_fy * g_small, g_s);
    g_log = os::env("PS2_HWR_LOG") == "1";
    g_stream_order = os::env("PS2_HWR_ORDER") == "stream";
    if (const std::string mb = os::env("PS2_HWR_CACHE_MB"); !mb.empty()) g_cache_budget = std::strtoull(mb.c_str(), nullptr, 10) << 20;
    for (u32 b = 0; b < kBlocks; ++b) g_block[b] = mix(0x1A17ull, b);
    touch_all();
    std::fill(std::begin(g_owner), std::end(g_owner), -1);
    // the drawing program
    std::string src = kHwrShader;
    if (C.dual) {
        const auto at = src.find("//DUAL_ENABLE");
        if (at != std::string::npos) src.replace(at, 13, "enable dual_source_blending;");
    } else if (const auto b = src.find("//DUAL_BEGIN"), e = src.find("//DUAL_END"); b != std::string::npos && e != std::string::npos) {
        src.erase(b, e - b);
    }
    {
        WGPUShaderSourceWGSL ws = WGPU_SHADER_SOURCE_WGSL_INIT;
        ws.code = WGPUStringView{src.data(), src.size()};
        WGPUShaderModuleDescriptor md = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
        md.nextInChain = &ws.chain;
        g_module = wgpuDeviceCreateShaderModule(C.device, &md);
    }
    {
        const std::string sync = std::string(C.gs_shader) + kHwrSyncShader;
        WGPUShaderSourceWGSL ws = WGPU_SHADER_SOURCE_WGSL_INIT;
        ws.code = WGPUStringView{sync.data(), sync.size()};
        WGPUShaderModuleDescriptor md = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
        md.nextInChain = &ws.chain;
        g_sync_module = wgpuDeviceCreateShaderModule(C.device, &md);
    }
    if (!g_module || !g_sync_module) return false;
    // layouts
    {
        WGPUBindGroupLayoutEntry e = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        e.binding = 0;
        // OpenGL ES on Mali has no storage buffers in vertex shaders: there the fragment stage reads the records (vs_gl, fs_gl)
        e.visibility = C.gl ? WGPUShaderStage_Fragment : WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
        e.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
        // binding 1: the palettes, for a CT32 target read as T8H straight from the target (view 3) and view 4; binding 2: the address tables (view 4)
        // binding 3: graphics memory, for the mip levels of a view-4 read that memory holds (sample_v4_mip)
        WGPUBindGroupLayoutEntry pe[4] = {e, WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT, WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT, WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT};
        pe[1].binding = 1;
        pe[1].visibility = WGPUShaderStage_Fragment;
        pe[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
        pe[2] = pe[1];
        pe[2].binding = 2;
        pe[3] = pe[1];
        pe[3].binding = 3;
        g_params_layout = make_layout(pe, 4);
        std::vector<u32> at(kAddrWords, 0);
        if (C.swizzle && C.swizzle_size >= kInv32) {
            std::copy(C.swizzle, C.swizzle + kInv32, at.begin());
            for (u32 y = 0; y < 32; ++y)
                for (u32 x = 0; x < 64; ++x) at[kInv32 + (C.swizzle[0 + y * 64 + x] & 8191u) / 4] = x | y << 8;
            for (u32 y = 0; y < 64; ++y)
                for (u32 x = 0; x < 64; ++x) {
                    at[kInv16 + (C.swizzle[4096 + y * 64 + x] & 8191u) / 2] = x | y << 8;
                    at[kInv16S + (C.swizzle[8192 + y * 64 + x] & 8191u) / 2] = x | y << 8;
                }
        }
        WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
        bd.size = static_cast<u64>(kAddrWords) * 4;
        bd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
        g_addr = wgpuDeviceCreateBuffer(C.device, &bd);
        wgpuQueueWriteBuffer(C.queue, g_addr, 0, at.data(), bd.size);
    }
    {
        WGPUBindGroupLayoutEntry e[6] = {WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT, WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT, WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT,
                                         WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT, WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT, WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT};
        e[0].binding = 0;
        e[0].visibility = WGPUShaderStage_Fragment;
        e[0].texture.sampleType = WGPUTextureSampleType_Float;
        e[0].texture.viewDimension = WGPUTextureViewDimension_2D;
        for (u32 k = 1; k < 4; ++k) {
            e[k].binding = k;
            e[k].visibility = WGPUShaderStage_Fragment;
            e[k].sampler.type = WGPUSamplerBindingType_Filtering;
        }
        for (u32 k = 4; k < 6; ++k) e[k] = e[0], e[k].binding = k;
        g_tex_layout = make_layout(e, 6);
    }
    {
        WGPUBindGroupLayoutEntry e = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        e.binding = 0;
        e.visibility = WGPUShaderStage_Compute;
        e.texture.sampleType = WGPUTextureSampleType_Float;
        e.texture.viewDimension = WGPUTextureViewDimension_2D;
        g_store_layout = make_layout(&e, 1);
        WGPUBindGroupLayoutEntry z = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        z.binding = 0;
        z.visibility = WGPUShaderStage_Compute;
        z.texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;  // hwr_sync.wgsl reads it as texture_2d<f32> (OpenGL)
        z.texture.viewDimension = WGPUTextureViewDimension_2D;
        g_depth_layout = make_layout(&z, 1);
        WGPUBindGroupLayoutEntry l = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        l.binding = 0;
        l.visibility = WGPUShaderStage_Compute;
        l.storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
        l.storageTexture.format = WGPUTextureFormat_RGBA8Unorm;
        l.storageTexture.viewDimension = WGPUTextureViewDimension_2D;
        g_load_layout = make_layout(&l, 1);
    }
    {
        WGPUBindGroupLayout layouts[2] = {g_params_layout, g_tex_layout};
        WGPUPipelineLayoutDescriptor pd = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
        pd.bindGroupLayoutCount = 2;
        pd.bindGroupLayouts = layouts;
        g_draw_layout = wgpuDeviceCreatePipelineLayout(C.device, &pd);
    }
    g_load_pipe = compute("hwr_load", g_load_layout);
    g_store_pipe = compute("hwr_store", g_store_layout);
    g_store_z_pipe = compute("hwr_store_z", g_depth_layout);
    if (!g_load_pipe || !g_store_pipe || !g_store_z_pipe) return false;
    {
        WGPUTextureDescriptor d = WGPU_TEXTURE_DESCRIPTOR_INIT;
        d.size = {1, 1, 1};
        d.format = WGPUTextureFormat_RGBA8Unorm;
        d.usage = WGPUTextureUsage_TextureBinding;
        d.mipLevelCount = 1;
        d.sampleCount = 1;
        g_dummy = wgpuDeviceCreateTexture(C.device, &d);
        g_dummy_view = wgpuTextureCreateView(g_dummy, nullptr);
        g_dummy_group = tex_group(g_dummy_view, 0);
    }
    timer_init();
    // the pipelines of the built-in list and of the recorded list are made now (a pipeline made in the middle of a frame stalls it; tens of
    // milliseconds each on a phone)
    {
        const auto t0 = std::chrono::steady_clock::now();
        std::string path = os::env("PS2_HWR_PIPELINES");
        if (path.empty()) {
            const std::string rec = os::env("PS2_VU_RECORD");  // the folder of the recorded vector programs: the list lies next to it, as hwdraw's
            if (!rec.empty()) {
                std::error_code ec;
                const std::filesystem::path dir = std::filesystem::path(rec).parent_path() / "hw";
                std::filesystem::create_directories(dir, ec);
                path = (dir / "hwr_pipelines.txt").string();
            }
        }
        std::vector<u64> keys(std::begin(kBuiltinKeys), std::end(kBuiltinKeys));
        if (!path.empty())
            if (std::FILE *f = os::open(path.c_str(), "rb")) {
                char line[64];
                while (std::fgets(line, sizeof line, f)) {
                    char *end = nullptr;
                    const unsigned long long k = std::strtoull(line, &end, 16);
                    if (end != line && k) keys.push_back(k);
                }
                std::fclose(f);
            }
        g_pipe_loading = true;
        for (const u64 key : keys) {
            PipeKey k = PipeKey::unpack(key);
            if (k.dual && !C.dual) {  // made without dual-source blending: the factor is the source alpha (blend_map's other case)
                if (k.mask & 8u) continue;
                k.dual = false;
                if (k.src == WGPUBlendFactor_Src1) k.src = WGPUBlendFactor_SrcAlpha;
                if (k.src == WGPUBlendFactor_OneMinusSrc1) k.src = WGPUBlendFactor_OneMinusSrcAlpha;
                if (k.dst == WGPUBlendFactor_Src1) k.dst = WGPUBlendFactor_SrcAlpha;
                if (k.dst == WGPUBlendFactor_OneMinusSrc1) k.dst = WGPUBlendFactor_OneMinusSrcAlpha;
            }
            pipeline(k);
        }
        g_pipe_loading = false;
        g_pipe_path = path;
        std::fprintf(stderr, "[hwr] %zu pipelines made at the start (%zu keys, list %s) in %.1f ms\n", g_pipes.size(), keys.size(), path.empty() ? "-" : path.c_str(),
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    {
        const std::string dt = os::env("PS2_DRAW_TIME");
        drawtime::g_on = dt == "1" || dt == "ab" || dt == "ab-memo";
        g_ab_memo_only = dt == "ab-memo";
        static const bool ab = dt == "ab" || dt == "ab-memo";
        g_ab_on = ab;
    }
    g_on = true;
    std::fprintf(stderr, "[hwr] hardware renderer on (PS2_RENDERER=hw): dual-source blending %s, texture cache %llu MB\n", C.dual ? "yes" : "no",
                 static_cast<unsigned long long>(g_cache_budget >> 20));
    return true;
}

namespace {
long skip_tbp_value() {
    static const long v = os::env("PS2_HWR_SKIP_TBP").empty() ? -1 : std::strtol(os::env("PS2_HWR_SKIP_TBP").c_str(), nullptr, 16);
    return v;
}

// the draw state of the shapes that follow, worked out again when the setup, the epoch (uploads, palette loads, page owners) or the submit changed
// The states worked out last, by their setup words (a game sends the same registers again for each object: a new setup serial, the same
// state). An entry is good while nothing it depends on changed: the same epoch (uploads, palette loads, page owners, targets made, grown or
// renamed) and submit, and it did not read a target through a write-back to memory (no_memo: draws into that target do not move the epoch).
// build_state does not read the GS state beyond the setup words, the mip words and the kind of primitive. PS2_HWR_STATE_MEMO=0: off.
constexpr int kMemo = 8;
struct Memo {
    u64 epoch = 0, submit = ~0ull;
    u32 words[40 + 8 + 2];
    DrawState s;
};
Memo g_memo[kMemo];
int g_memo_next = 0;
bool memo_on() {
    static const bool on = os::env("PS2_HWR_STATE_MEMO") != "0";
    return on && !g_ab_off;
}
inline void memo_key(u32 *k, const u32 *w, const u32 *mip, u32 mip_top, u32 prim) {
    std::memcpy(k, w, 40 * sizeof(u32));
    std::memcpy(k + 40, mip, 8 * sizeof(u32));
    k[48] = mip_top;
    k[49] = prim < 3 ? 1u : 0u;
}

inline void prepare_state(DrawState &s, const State &st, const u32 *w, const u32 *mip, u32 mip_top, u64 serial, u32 prim) {
    if (s.serial != serial || s.epoch != g_epoch || s.submit != g_submit) {
        ++F.builds;
        if (s.serial == serial && s.submit == g_submit) ++F.builds_epoch;
        u32 key[50];
        const bool memo = memo_on() && !g_no_view4;
        if (memo) {
            memo_key(key, w, mip, mip_top, prim);
            for (Memo &m : g_memo)
                if (m.epoch == g_epoch && m.submit == g_submit && std::memcmp(m.words, key, sizeof key) == 0) {
                    s = m.s;
                    s.serial = serial;
                    ++F.memo_hits;
                    return;
                }
        }
        const auto b0 = g_log ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const prof::Scope build_scope(prof::kRBuild);
        const drawtime::Span build_span(drawtime::kBuild);
        build_state(s, st, w, mip, mip_top, prim);
        if (g_log) F.build_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - b0).count();
        s.serial = serial;
        s.epoch = g_epoch;
        s.submit = g_submit;
        s.param = shape_params(s, mip, mip_top);
        s.sc[0] = static_cast<int>(w[10] & 0xFFFFu), s.sc[2] = static_cast<int>(w[10] >> 16);
        s.sc[1] = static_cast<int>(w[11] & 0xFFFFu), s.sc[3] = static_cast<int>(w[11] >> 16);
        if (memo && !s.no_memo) {
            Memo &m = g_memo[g_memo_next];
            g_memo_next = (g_memo_next + 1) % kMemo;
            m.epoch = s.epoch, m.submit = s.submit;
            std::memcpy(m.words, key, sizeof key);
            m.s = s;
        }
    }
}

// one corner of a shape as the card's vertex (GS pixel x is the centre of the card's pixel x)
[[gnu::always_inline]] inline void put_vtx(Vtx &o, float x, float y, const Vertex &col, const Vertex &tv, u32 z, u8 fog, float tq, float sxs, float sys, u32 zmax,
                                           double zscale, bool fst, u32 param) {
    o.x = (x + 0.5f) * sxs - 1.0f;  // at PS2_SCALE the card's samples sit at the centres of the GS pixel's parts (x + 1/(2 scale) ...)
    o.y = 1.0f - (y + 0.5f) * sys;
    o.z = static_cast<float>(static_cast<double>(std::min(z, zmax)) / zscale);
    o.rgba = col.r | (static_cast<u32>(col.g) << 8) | (static_cast<u32>(col.b) << 16) | (static_cast<u32>(col.a) << 24);
    if (fst) o.s = static_cast<float>(tv.u) / 16.0f, o.t = static_cast<float>(tv.v) / 16.0f, o.q = 1.0f;
    else o.s = tv.s, o.t = tv.t, o.q = tq;
    o.fog = static_cast<float>(fog);
    o.di = param;
}
}  // namespace

void shape(const State &st, const u32 *w, const u32 *mip, u32 mip_top, u64 serial, const Vertex *v, int count) {
    const auto t0 = g_log ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};  // timed for PS2_HWR_LOG only: two clock reads per shape cost ~1 ms per frame on a phone
    const prof::Scope shape_scope(prof::kRShape);
    const drawtime::Span shape_span(drawtime::kShape);
    ++F.shapes;
    if (prov::g_on) prov::count(prov::kShapes);
    const u32 prim = static_cast<u32>(st.prim) & 7u;
    DrawState &s = g_ds;
    prepare_state(s, st, w, mip, mip_top, serial, prim);
    // A view-4 draw reads every texel from one target (the owner of the texture's first page). The texels this draw reads must all lie in
    // pages that target holds; otherwise (Harry Potter 2's sun haze: a 1408-wide T8 view over a 704-wide 16-bit buffer, whose further
    // pages belong to the frame, a blur target and memory) this one draw is set up again the old way (write-back and decode).
    // PS2_HWR_DIRECT_PAGES=0: no check (the earlier behaviour).
    static const bool view4_pages = os::env("PS2_HWR_DIRECT_PAGES") != "0";
    if (view4_pages && s.src.view == 4 && s.src.target >= 0) {
        const Rect tr = read_rect(w, v, std::min(count, prim == 6 ? 2 : 3), 1 << (w[15] & 0xFFu), 1 << (w[15] >> 8));
        bool inside = true;
        for_pages(w[14], w[12], w[13], tr.x0, tr.y0, tr.x1, tr.y1, [&](u32 page) {
            if (g_owner[page & (kPages - 1)] != s.src.target) inside = false;
        });
        // A mipmapped read: the same rectangle at every level (halved per level), in that level's base and width. Unless the draw cannot leave
        // level 0: the largest LOD its corners give (the GS's LOD = log2(1/q) * 2^L + K, or K alone with LCM; 1/q over a shape stays between
        // its corners' values) is 0 or less, so the shader reads level 0 (magnification) only. Harry Potter 2's spell sparkles: K = -7.9, their
        // level 1 base points into memory the frame does not hold.
        bool needs_levels = mip_top && mip[0];
        if (needs_levels) {
            const u32 m = mip[0];
            const float k = static_cast<float>(static_cast<int32_t>(((m >> 16) & 0xFFFu) << 20) >> 20) / 16.0f;
            float lod_max = k;
            if (!(m & 1u) && !(w[0] & 16u)) {  // per pixel, from q (a shape with u, v coordinates has q = 1)
                float qmin = 1e30f;
                for (int i = 0; i < std::min(count, prim == 6 ? 2 : 3); ++i) qmin = std::min(qmin, std::fabs(v[i].q));
                lod_max = qmin > 0.0f ? std::log2(1.0f / qmin) * static_cast<float>(1u << ((m >> 8) & 3u)) + k : 1e9f;
            }
            if (lod_max <= -0.01f) needs_levels = false;  // a margin: the shader's LOD per pixel is the same sum in float
        }
        // Each level is the target's (all its pages held by it) or graphics memory's (all its pages held by nobody and up to date there:
        // indexed formats, PS2_SCALE 1); a level split between them sets the draw up the old way.
        u32 mem_mask = 0;
        static const bool mem_levels = os::env("PS2_HWR_DIRECT_MEMLEVELS") == "1";  // opt-in, NOT exact: a level that memory holds is read when the pass runs, after the game's later uploads
        const bool mem_ok = mem_levels && C.vram && g_s == 1 && (w[14] == 0x13 || w[14] == 0x14);
        if (needs_levels)
            for (u32 n = 1; n <= mip_top && inside; ++n) {
                bool all_target = true, all_memory = true;
                for_pages(w[14], mip[n] & 0x3FFFu, (mip[n] >> 14) & 0x3Fu, tr.x0 >> n, tr.y0 >> n, tr.x1 >> n, tr.y1 >> n, [&](u32 page) {
                    const u32 pg = page & (kPages - 1);
                    if (g_owner[pg] != s.src.target) all_target = false;
                    if (g_owner[pg] != -1 || g_page_stale[pg]) all_memory = false;
                });
                if (all_target) continue;
                if (all_memory && mem_ok) {
                    mem_mask |= 1u << n;
                    for_pages(w[14], mip[n] & 0x3FFFu, (mip[n] >> 14) & 0x3Fu, tr.x0 >> n, tr.y0 >> n, tr.x1 >> n, tr.y1 >> n, [&](u32 page) { g_mem_read_submit[page & (kPages - 1)] = g_submit; });
                }
                else inside = false;
            }
        // a draw that reads levels from memory: a record pair of its own carrying which (the shader's from_mem); the pair the state made is
        // shared by the shapes that follow with other rectangles
        if (inside && mem_mask) {
            s.param = add_mip_record(s.pw, mip, mip_top, mem_mask);
            static const bool log_mem = os::env("PS2_HWR_LOG") == "6";
            static int nlog = 0;
            if (log_mem && nlog++ < 40)
                std::fprintf(stderr, "[hwr-v4-mem] frame %llu tex %04X/%u psm %02X rect %d,%d-%d,%d mem levels %02X top %u mip0 %08X fst %d color t%d (bp %04X psm %02X) lv1 %04X/%u lv2 %04X/%u lv3 %04X/%u\n",
                             static_cast<unsigned long long>(g_frames), w[12], w[13], w[14], tr.x0, tr.y0, tr.x1, tr.y1, mem_mask, mip_top, mip[0], (w[0] & 16u) ? 1 : 0, s.color,
                             g_targets[static_cast<std::size_t>(s.color)]->bp, g_targets[static_cast<std::size_t>(s.color)]->psm, mip[1] & 0x3FFFu, (mip[1] >> 14) & 63u, mip[2] & 0x3FFFu,
                             (mip[2] >> 14) & 63u, mip[3] & 0x3FFFu, (mip[3] >> 14) & 63u);
        }
        if (!inside) {
            static const bool log_out = os::env("PS2_HWR_LOG") == "6";
            if (log_out) {
                static int n = 0;
                if (n++ < 30) {
                    std::string owners;
                    for_pages(w[14], w[12], w[13], tr.x0, tr.y0, tr.x1, tr.y1, [&](u32 page) { owners += std::to_string(page) + ":" + std::to_string(g_owner[page & (kPages - 1)]) + " "; });
                    std::fprintf(stderr, "[hwr-v4-out] tex %04X/%u psm %02X rect %d,%d-%d,%d src t%d color t%d fst %d mip_top %u lod-needs %d pages %s\n", w[12], w[13], w[14], tr.x0, tr.y0, tr.x1, tr.y1,
                                 s.src.target, s.color, (w[0] & 16u) ? 1 : 0, mip_top, needs_levels ? 1 : 0, owners.c_str());
                }
            }
            ++F.view4_out;
            g_no_view4 = true;
            build_state(s, st, w, mip, mip_top, prim);
            g_no_view4 = false;
            s.param = shape_params(s, mip, mip_top);
            s.epoch = 0;  // the next shape is set up again (with view 4 if its texels allow it)
        }
    }
    if (s.skip == 2) {
        ++F.fallback[s.reason];
        if (prov::g_on) prov::drop(kReasonName[s.reason]);
        return;
    }
    // PS2_HWR_SKIP_TBP=<hex>: shapes reading that texture base are not drawn (a bisecting aid)
    const long skip_tbp = skip_tbp_value();
    if (skip_tbp >= 0 && (w[0] & 2u) && static_cast<long>(w[12]) == skip_tbp) return;
    if (s.skip == 1) {
        ++F.skipped;
        if (prov::g_on) prov::count(prov::kSkipped);
        return;
    }
    const int n = prim == 6 ? 2 : 3;
    if (count < n) return;
    u64 attr = st.prim;
    if (!(st.prmodecont & 1u)) attr = (st.prim & 7u) | (st.prmode & ~7ull);
    const ps2::gs::Context &gc = st.ctx[(attr >> 9) & 1u];
    const float ofx = static_cast<float>(gc.xyoffset & 0xFFFFu), ofy = static_cast<float>((gc.xyoffset >> 32) & 0xFFFFu);
    Target &ct = *g_targets[static_cast<std::size_t>(s.color)];
    const int W = static_cast<int>(ct.w);
    int H = static_cast<int>(ct.h);
    float px[3], py[3];
    float minx = 1e9f, miny = 1e9f, maxx = -1e9f, maxy = -1e9f;
    for (int i = 0; i < n; ++i) {
        px[i] = (static_cast<float>(v[i].x) - ofx) / 16.0f;
        py[i] = (static_cast<float>(v[i].y) - ofy) / 16.0f;
        minx = std::min(minx, px[i]), maxx = std::max(maxx, px[i]);
        miny = std::min(miny, py[i]), maxy = std::max(maxy, py[i]);
    }
    // the pixels the shape can cover, cut to the scissor and to the target
    int x0 = std::max(s.sc[0], static_cast<int>(std::ceil(minx))), x1 = std::min(s.sc[2], static_cast<int>(std::ceil(maxx)) - (prim == 6 ? 1 : 0));
    int y0 = std::max(s.sc[1], static_cast<int>(std::ceil(miny))), y1 = std::min(s.sc[3], static_cast<int>(std::ceil(maxy)) - (prim == 6 ? 1 : 0));
    x0 = std::max(x0, 0), y0 = std::max(y0, 0);
    if (x1 < x0 || y1 < y0) {
        ++F.skipped;
        return;
    }
    if (y1 >= H && H < static_cast<int>(kMaxRows)) {  // below the rows the target was made with: it (and its depth) become 512 rows, or the rows drawn (up to kMaxRows)
        const u32 rows = std::min<u32>(kMaxRows, std::max<u32>(512u, (static_cast<u32>(y1) + 32u) & ~31u));
        grow(s.color, rows);
        if (s.needs_depth) grow(s.depth, rows);
        H = static_cast<int>(ct.h);
    }
    if (x1 >= W || y1 >= H) {
        ++F.approx[kOutside];
        x1 = std::min(x1, W - 1), y1 = std::min(y1, H - 1);
        if (x1 < x0 || y1 < y0) return;
    }
    const u32 flags = w[0];
    Target *dt = s.needs_depth ? g_targets[static_cast<std::size_t>(s.depth)].get() : nullptr;
    const bool k2 = s.kp2 != 0;
    // Clears: the game clears a target with a row of untextured sprites side by side (16 strips of 32 pixels for the 512 wide targets). Such a
    // run is drawn as it comes; when it has covered every page the target holds, its draws are taken back and the target gets a new surface
    // that starts with the clear colour (the render pass's load op): the draws before it and the draws after it no longer depend on each other.
    const bool clear_part = prim == 6 && !(flags & 2u) && !s.k1.blend && !(flags & 4u) && !k2 && s.k1.atest == 0 && (s.k1.mask == 0 || s.color_full) &&
                            (s.k1.mask || s.k1.zwrite) && (!s.k1.zwrite || (s.k1.zcmp == WGPUCompareFunction_Always && dt && dt->w == ct.w && dt->h == ct.h));
    // the same with textured opaque sprites (the copy of the last picture into the display buffer): the draws stay, the target gets a new
    // surface that the render pass does not load
    const bool cover_part = prim == 6 && (flags & 2u) && s.src.kind && s.src.target != s.color && !s.k1.blend && !k2 && s.k1.atest == 0 && s.color_full;
    const Vertex &cv = v[1];
    const u32 crgba = cv.r | (static_cast<u32>(cv.g) << 8) | (static_cast<u32>(cv.b) << 16) | (static_cast<u32>(cv.a) << 24);
    ClearRun &cr = g_clear;
    const bool same_kind = cr.tex ? cover_part : (clear_part && cr.zw == s.k1.zwrite && cr.rgba == crgba && cr.z == cv.z);
    if (cr.on && !(same_kind && cr.serial == serial && cr.submit == g_submit && cr.color == s.color && x0 == cr.next_x && y0 == cr.y0 && y1 == cr.y1)) cr.on = false;
    if (!cr.on && (clear_part || cover_part) && x0 == 0 && y0 == 0 && !g_stream_order) {
        if (cover_part) close_pass();  // the run's draws go into a node of their own (it is moved to the new surface)
        cr = ClearRun{};
        cr.on = true, cr.tex = cover_part, cr.serial = serial, cr.submit = g_submit, cr.color = s.color, cr.zw = s.k1.zwrite, cr.y0 = y0, cr.y1 = y1, cr.rgba = crgba, cr.z = cv.z;
        cr.ops_n = g_ops.size(), cr.open = g_open_pass, cr.open_draws = g_open_pass >= 0 ? g_ops[static_cast<std::size_t>(g_open_pass)].draw_count : 0;
        cr.draws_n = g_draws.size(), cr.last_count = g_draws.empty() ? 0 : g_draws.back().count, cr.verts_n = g_verts.size();
    }
    WGPUBindGroup group = g_dummy_group;
    u64 res = 0;
    if (s.src.kind) {
        if (s.src.target >= 0 && s.src.target == s.color) {
            Rect rr;
            if (s.src.view == 4) {
                // a draw that reads its own target through memory addresses: the target pixels of the memory pages the texels it reads lie in
                const Target &st = *g_targets[static_cast<std::size_t>(s.src.target)];
                const Rect tr = read_rect(w, v, n, 1 << (w[15] & 0xFFu), 1 << (w[15] >> 8));
                u32 pw, ph;
                page_dims(st.psm, pw, ph);
                const u32 per_row = std::max<u32>(1, st.bw * 64 / pw);
                rr = Rect{1 << 30, 1 << 30, -1, -1};
                const auto add_page = [&](u32 page) {
                    if (g_owner[page & (kPages - 1)] != s.src.target) return;  // a level memory holds is read from memory
                    const int rel = static_cast<int>((page - st.bp / 32) & (kPages - 1));
                    const int px = static_cast<int>((static_cast<u32>(rel) % per_row) * pw), py = static_cast<int>((static_cast<u32>(rel) / per_row) * ph);
                    rr.x0 = std::min(rr.x0, px), rr.y0 = std::min(rr.y0, py);
                    rr.x1 = std::max(rr.x1, px + static_cast<int>(pw) - 1), rr.y1 = std::max(rr.y1, py + static_cast<int>(ph) - 1);
                };
                for_pages(w[14], w[12], w[13], tr.x0, tr.y0, tr.x1, tr.y1, add_page);
                // the mip levels the target holds too (the write-back path writes back every level's pages: mip_range)
                if (mip_top && mip[0])
                    for (u32 n = 1; n <= mip_top; ++n) for_pages(w[14], mip[n] & 0x3FFFu, (mip[n] >> 14) & 0x3Fu, tr.x0 >> n, tr.y0 >> n, tr.x1 >> n, tr.y1 >> n, add_page);
                if (rr.x1 < 0) rr = Rect{0, 0, static_cast<int>(st.w) - 1, static_cast<int>(st.h) - 1};
            } else {
                rr = read_rect(w, v, n, W, H);
            }
            group = self_copy(s.src.target, rr, res, s.src.view == 3, s.src.view == 4);
        }
        else group = s.src.group, res = s.src.res;
    }
    if (g_log && s.src.kind == 1 && mip_top && mip[0]) {  // how many levels the corners of this draw can sample (the same bound as the view-4 mip check)
        const u32 m = mip[0];
        const float k = static_cast<float>(static_cast<int32_t>(((m >> 16) & 0xFFFu) << 20) >> 20) / 16.0f;
        float lod_max = k;
        if (!(m & 1u) && !(w[0] & 16u)) {
            float qmin = 1e30f;
            for (int i = 0; i < n; ++i) qmin = std::min(qmin, std::fabs(v[i].q));
            lod_max = qmin > 0.0f ? std::log2(1.0f / qmin) * static_cast<float>(1u << ((m >> 8) & 3u)) + k : 1e9f;
        }
        const u32 need = lod_max <= -0.01f ? 1u : std::min<u32>(mip_top + 1u, static_cast<u32>(std::floor(lod_max)) + 2u);
        ++F.mipneed[std::min<u32>(need, 7u)];
        ++F.mipdraws;
    }
    ensure_pass(s);
    Op &pass = g_ops[static_cast<std::size_t>(g_open_pass)];
    if (g_dump_now && (s.src.target >= 0 || prim == 6))
        std::fprintf(stderr, "[hwr-dump] shape t%d %04x/%u prim %u rect %d,%d-%d,%d src t%d view %u uv %d,%d-%d,%d tbp %04x blend %d mask %x zw %d atest %u clearpart %d\n", s.color, ct.bp, ct.bw, prim, x0, y0, x1,
                     y1, s.src.target, s.src.view, v[0].u >> 4, v[0].v >> 4, v[1].u >> 4, v[1].v >> 4, w[12], s.k1.blend, s.k1.mask, s.k1.zwrite, s.k1.atest, clear_part);
    // corners in clip space (GS pixel x is the centre of the card's pixel x), as triangles
    const u32 first = static_cast<u32>(g_verts.size());
    const bool fst = flags & 16u, iip = flags & 1u;
    const float sxs = 2.0f / static_cast<float>(W), sys = 2.0f / static_cast<float>(H);
    // the vertices are written in place: the buffer grows once per shape (6 or 3), not per vertex
    g_verts.resize(first + (prim == 6 ? 6u : 3u));
    Vtx *out = g_verts.data() + first;
    const u32 zmax = s.zmax, param = s.param;
    const double zscale = s.zscale;
    const auto put = [&](float x, float y, const Vertex &col, const Vertex &tv, u32 z, u8 fog, float tq) {
        put_vtx(*out++, x, y, col, tv, z, fog, tq, sxs, sys, zmax, zscale, fst, param);
    };
    if (prim == 6) {
        const Vertex &a = v[0], &b = v[1];
        Vertex ab = a, ba = b;  // texture coordinates of the other two corners: s from one corner, t from the other
        ab.s = b.s, ab.u = b.u;  // (x1, y0)
        ba.s = a.s, ba.u = a.u;  // (x0, y1)
        put(px[0], py[0], b, a, b.z, b.f, b.q);
        put(px[1], py[0], b, ab, b.z, b.f, b.q);
        put(px[0], py[1], b, ba, b.z, b.f, b.q);
        put(px[1], py[0], b, ab, b.z, b.f, b.q);
        put(px[1], py[1], b, b, b.z, b.f, b.q);
        put(px[0], py[1], b, ba, b.z, b.f, b.q);
    } else {
        for (int i = 0; i < 3; ++i) put(px[i], py[i], iip ? v[i] : v[2], v[i], v[i].z, v[i].f, v[i].q);
    }
    const u32 nv = static_cast<u32>(g_verts.size()) - first;
    u32 sc[4] = {static_cast<u32>(std::max(s.sc[0], 0)), static_cast<u32>(std::max(s.sc[1], 0)), 0, 0};
    const u32 ex = static_cast<u32>(std::max(std::min(s.sc[2] + 1, W), 0)), ey = static_cast<u32>(std::max(std::min(s.sc[3] + 1, H), 0));
    sc[2] = ex > sc[0] ? ex - sc[0] : 0;
    sc[3] = ey > sc[1] ? ey - sc[1] : 0;
    bool merged = false;
    Tri tris[2];
    int ntris = 0;
    bool overlap = false;
    if (k2) {
        if (prim == 6) {
            tris[0] = Tri{{px[0], px[1], px[0]}, {py[0], py[0], py[1]}};
            tris[1] = Tri{{px[1], px[1], px[0]}, {py[0], py[1], py[1]}};
            ntris = 2;
        } else {
            tris[0] = Tri{{px[0], px[1], px[2]}, {py[0], py[1], py[2]}};
            ntris = 1;
        }
        overlap = g_split_tris.size() + static_cast<std::size_t>(ntris) > kSplitTris;
        for (int k = 0; k < ntris && !overlap; ++k)
            for (const Tri &o : g_split_tris)
                if (!separated(tris[k], o)) {
                    overlap = true;
                    break;
                }
    }
    if (pass.draw_count && !overlap) {
        DrawRec &last = g_draws.back();
        if (last.k1 == s.kp1 && last.k2 == s.kp2 && last.group == group && last.blend == s.blend && last.zuse == s.needs_depth && std::equal(sc, sc + 4, last.sc) &&
            last.first + last.count == first) {
            last.count += nv;
            merged = true;
        }
    }
    if (!merged) {
        g_split_tris.clear();
        DrawRec r;
        r.k1 = s.kp1, r.k2 = s.kp2, r.group = group, r.blend = s.blend;
        std::copy(sc, sc + 4, r.sc);
        r.first = first, r.count = nv;
        r.res = res;
        r.zuse = s.needs_depth;
        r.zread = s.zread;
        g_draws.push_back(r);
        ++pass.draw_count;
    }
    for (int k = 0; k < ntris; ++k) g_split_tris.push_back(tris[k]);
    // what the shape wrote: the pages it touched now belong to its targets
    if (s.cwrite_any) {
        ct.valid_w = std::max(ct.valid_w, x1 + 1), ct.valid_h = std::max(ct.valid_h, y1 + 1);
        for_pages(ct.psm, ct.bp, ct.bw, x0, y0, x1, y1, [&](u32 page) {
            if (static_cast<int>(page) == page_log())
                std::fprintf(stderr, "[hwr-page] frame %llu page %u written by shape into t%d rect %d,%d-%d,%d prim %u tex %04x/%u psm %02X src t%d view %u blend %d mask %x\n",
                             static_cast<unsigned long long>(g_frames), page, s.color, x0, y0, x1, y1, prim, w[12], w[13], w[14], s.src.target, s.src.view, s.k1.blend, s.k1.mask);
            take_page(s.color, page);
            ++g_page_write[page & (kPages - 1)];
        });
        note_write(ct, x0, y0, x1, y1, s.awrite_any);
    }
    if (s.zwrite_any && dt) {
        dt->valid_w = std::max(dt->valid_w, x1 + 1), dt->valid_h = std::max(dt->valid_h, y1 + 1);
        for_pages(dt->psm, dt->bp, dt->bw, x0, y0, x1, y1, [&](u32 page) {
            take_page(s.depth, page);
            ++g_page_write[page & (kPages - 1)];
        });
    }
    ++F.drawn;
    if (prov::g_on) prov::count(prov::kDrawn), prov::count(prov::kPixels, static_cast<std::uint64_t>(std::max(0, x1 - x0 + 1)) * static_cast<std::uint64_t>(std::max(0, y1 - y0 + 1)));
    F.vertices += nv;
    // the clear run goes on; when it reaches the right edge and covers every page of the target (and of the depth target), it becomes a clear
    if (cr.on) {
        cr.next_x = x1 + 1;
        if (x1 >= W - 1) {
            cr.on = false;
            bool only_nodes = true;  // what the run recorded: only draws into this target's surface
            for (std::size_t k = cr.ops_n; k < g_ops.size(); ++k)
                if (g_ops[k].kind != kOpPass || g_ops[k].cs != ct.s) only_nodes = false;
            if (cr.tex && only_nodes && cr.ops_n < g_ops.size() && covers(s.color, y1)) {
                rename(ct);
                for (std::size_t k = cr.ops_n; k < g_ops.size(); ++k) g_ops[k].cs = ct.s;
                g_ops[cr.ops_n].clear_c = true;  // what it held before is not needed (cleared to 0, then drawn over)
                ++F.overwrites;
            } else if (!cr.tex && only_nodes && (!s.k1.mask || covers(s.color, y1)) && (!s.k1.zwrite || covers(s.depth, y1))) {
                g_ops.resize(cr.ops_n);
                g_open_pass = cr.open;
                if (cr.open >= 0) g_ops[static_cast<std::size_t>(cr.open)].draw_count = cr.open_draws;
                g_draws.resize(cr.draws_n);
                if (!g_draws.empty()) g_draws.back().count = cr.last_count;
                g_verts.resize(cr.verts_n);
                Op op{};
                op.kind = kOpPass;
                op.pos = g_stream_order ? position() : 0;
                op.tgt = s.color;
                if (s.k1.mask) {
                    float rgb[4] = {static_cast<float>(cv.r), static_cast<float>(cv.g), static_cast<float>(cv.b), static_cast<float>(cv.a | ((flags & 4096u) ? 0x80 : 0))};
                    if (is16(ct.psm)) {
                        for (int k = 0; k < 3; ++k) rgb[k] = std::floor(rgb[k] / 8.0f) * 8.0f;
                        rgb[3] = rgb[3] >= 128.0f ? 255.0f : 0.0f;
                    }
                    rename(ct);
                    op.clear_c = true;
                    for (int k = 0; k < 4; ++k) op.cc[k] = rgb[k] / 255.0f;
                }
                if (s.k1.zwrite) {
                    rename(*dt);
                    op.ds = dt->s;
                    op.clear_d = true;
                    op.cz = static_cast<float>(static_cast<double>(std::min(cv.z, s.zmax)) / s.zscale);
                }
                op.cs = ct.s;
                op.first_draw = static_cast<u32>(g_draws.size());
                close_pass();
                g_ops.push_back(op);
                g_open_pass = static_cast<int>(g_ops.size() - 1);
                g_split_tris.clear();
                ++F.clears;
            }
        }
    }
    if (g_log) F.cpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

bool batch_on() {
    static const bool on = os::env("PS2_HWR_BATCH") != "0";
    return on && !(g_ab_off && !g_ab_memo_only);
}

namespace {
// A corner of a run converted once (a strip's corner belongs to up to three triangles): its position relative to the offset in GS
// sub-pixels (dx, dy: x - XYOFFSET, 1/16 pixel), in GS pixels (x, y) and its card vertex with its own colour (a flat-shaded triangle takes
// the colour of its last corner: shape()'s `iip ? v[i] : v[2]`).
struct Corner {
    int dx, dy;
    float x, y;
    Vtx o;
};
std::vector<Corner> g_corners;
std::vector<u32> g_corner_mark;  // == g_corner_pass: converted in this prim_run
u32 g_corner_pass = 0;

bool view4_pages_on() {  // shape()'s view4_pages (PS2_HWR_DIRECT_PAGES)
    static const bool on = os::env("PS2_HWR_DIRECT_PAGES") != "0";
    return on;
}

// ceil(d / 16) for a position in 1/16 pixels: the same as shape()'s std::ceil((float(x) - ofx) / 16.0f), which is exact in float (both
// below 2^16), without the float rounding call (a library call on x86 without SSE4.1)
inline int ceil16(int d) { return (d + 15) >> 4; }

// Primitives of one run (shapes()) while the draw state stays what it is: what shape() does for each, in the same order and with the same
// results, without what cannot change between them. Ends after the primitive that changed the epoch (a page taken from another owner, a
// target grown): the next one's state is worked out again, as shape() would. Primitive k has the corners v[idx[per*k + j]] (idx null:
// v[per*k + j]); per 3: triangles, per 2: sprites (only states that are no clear or cover run: shapes() checks). Returns how many it took.
int prim_run(DrawState &s, const State &st, const u32 *w, const u32 *mip, u32 mip_top, const Vertex *v, int nverts, const u16 *idx, int prims, int per) {
    const u64 epoch = g_epoch;
    const bool sprite = per == 2;
    const bool view4 = s.src.view == 4 && view4_pages_on();
    const u32 nv = sprite ? 6u : 3u;
    u64 attr = st.prim;
    if (!(st.prmodecont & 1u)) attr = (st.prim & 7u) | (st.prmode & ~7ull);
    const ps2::gs::Context &gc = st.ctx[(attr >> 9) & 1u];
    const int iofx = static_cast<int>(gc.xyoffset & 0xFFFFu), iofy = static_cast<int>((gc.xyoffset >> 32) & 0xFFFFu);
    const float ofx = static_cast<float>(iofx), ofy = static_cast<float>(iofy);
    Target &ct = *g_targets[static_cast<std::size_t>(s.color)];
    const int W = static_cast<int>(ct.w);
    int H = static_cast<int>(ct.h);
    const u32 flags = w[0];
    Target *dt = s.needs_depth ? g_targets[static_cast<std::size_t>(s.depth)].get() : nullptr;
    const WGPUBindGroup group = s.src.kind ? s.src.group : g_dummy_group;
    const u64 res = s.src.kind ? s.src.res : 0;
    const bool fst = flags & 16u, iip = flags & 1u;
    float sxs = 2.0f / static_cast<float>(W), sys = 2.0f / static_cast<float>(H);
    const u32 zmax = s.zmax, param = s.param;
    const double zscale = s.zscale;
    // the pages already taken in this run (page columns and rows, inclusive): a primitive inside them changes no owner (and counts as a
    // write to them already: g_page_write is only compared with what the last write-back saw). Page sizes are powers of two.
    u32 cpw, cph, dpw = 1, dph = 1;
    page_dims(ct.psm, cpw, cph);
    if (dt) page_dims(dt->psm, dpw, dph);
    const int csx = __builtin_ctz(cpw), csy = __builtin_ctz(cph), dsx = __builtin_ctz(dpw), dsy = __builtin_ctz(dph);
    int cpr[4] = {1, 1, 0, 0}, dpr[4] = {1, 1, 0, 0};  // empty
    DrawRec *cur = nullptr;  // the draw this run's primitives go into, once known
    // triangle corners converted on first use; the conversion depends on the state (scale, depth range, record), so once per call
    if (!sprite) {
        if (g_corners.size() < static_cast<std::size_t>(nverts)) g_corners.resize(static_cast<std::size_t>(nverts)), g_corner_mark.resize(static_cast<std::size_t>(nverts), 0u);
        if (++g_corner_pass == 0) {
            std::fill(g_corner_mark.begin(), g_corner_mark.end(), 0u);
            g_corner_pass = 1;
        }
    }
    u32 pass_id = g_corner_pass;
    Corner *corners = g_corners.data();
    u32 *mark = g_corner_mark.data();
    const auto corner = [&](u32 i) -> const Corner & {
        Corner &c = corners[i];
        if (mark[i] != pass_id) {
            mark[i] = pass_id;
            const Vertex &p = v[i];
            c.dx = static_cast<int>(p.x) - iofx;
            c.dy = static_cast<int>(p.y) - iofy;
            c.x = (static_cast<float>(p.x) - ofx) / 16.0f;
            c.y = (static_cast<float>(p.y) - ofy) / 16.0f;
            put_vtx(c.o, c.x, c.y, p, p, p.z, p.f, p.q, sxs, sys, zmax, zscale, fst, param);
        }
        return c;
    };
    const u32 first_all = static_cast<u32>(g_verts.size());
    g_verts.resize(first_all + static_cast<std::size_t>(prims) * nv);  // written in place; cut to what was drawn at the end
    u32 used = first_all;
    int k = 0;
    while (k < prims) {
        u32 ci[3];
        if (idx) {
            for (int j = 0; j < per; ++j) ci[j] = idx[per * k + j];
        } else {
            for (int j = 0; j < per; ++j) ci[j] = static_cast<u32>(per * k + j);
        }
        if (view4) {  // shape(): the texels must all lie in pages the source target holds, else this one is set up the old way there
            Vertex tv[3];
            for (int j = 0; j < per; ++j) tv[j] = v[ci[j]];
            const Rect tr = read_rect(w, tv, per, 1 << (w[15] & 0xFFu), 1 << (w[15] >> 8));
            bool inside = true;
            for_pages(w[14], w[12], w[13], tr.x0, tr.y0, tr.x1, tr.y1, [&](u32 page) {
                if (g_owner[page & (kPages - 1)] != s.src.target) inside = false;
            });
            if (!inside) break;
        }
        ++k;
        ++F.shapes;
        const Corner *c[3] = {nullptr, nullptr, nullptr};
        int x0, x1, y0, y1;
        if (sprite) {  // the box of the two corners, the right and bottom edges not drawn
            const Vertex &a = v[ci[0]], &b = v[ci[1]];
            const int ax = static_cast<int>(a.x) - iofx, ay = static_cast<int>(a.y) - iofy, bx = static_cast<int>(b.x) - iofx, by = static_cast<int>(b.y) - iofy;
            x0 = std::max(s.sc[0], ceil16(std::min(ax, bx))), x1 = std::min(s.sc[2], ceil16(std::max(ax, bx)) - 1);
            y0 = std::max(s.sc[1], ceil16(std::min(ay, by))), y1 = std::min(s.sc[3], ceil16(std::max(ay, by)) - 1);
        } else {
            c[0] = &corner(ci[0]), c[1] = &corner(ci[1]), c[2] = &corner(ci[2]);
            x0 = std::max(s.sc[0], ceil16(std::min({c[0]->dx, c[1]->dx, c[2]->dx}))), x1 = std::min(s.sc[2], ceil16(std::max({c[0]->dx, c[1]->dx, c[2]->dx})));
            y0 = std::max(s.sc[1], ceil16(std::min({c[0]->dy, c[1]->dy, c[2]->dy}))), y1 = std::min(s.sc[3], ceil16(std::max({c[0]->dy, c[1]->dy, c[2]->dy})));
        }
        x0 = std::max(x0, 0), y0 = std::max(y0, 0);
        if (x1 < x0 || y1 < y0) {
            ++F.skipped;
            continue;
        }
        if (y1 >= H && H < static_cast<int>(kMaxRows)) {
            const u32 rows = std::min<u32>(kMaxRows, std::max<u32>(512u, (static_cast<u32>(y1) + 32u) & ~31u));
            grow(s.color, rows);
            if (s.needs_depth) grow(s.depth, rows);
            H = static_cast<int>(ct.h);
            cur = nullptr;  // another surface: the pass and the scissor are found again
            // the card's y of every corner depends on the height: convert again (from this primitive on, as shape() would)
            if (2.0f / static_cast<float>(H) != sys) {
                sys = 2.0f / static_cast<float>(H);
                if (!sprite) {
                    if (++g_corner_pass == 0) {
                        std::fill(g_corner_mark.begin(), g_corner_mark.end(), 0u);
                        g_corner_pass = 1;
                    }
                    pass_id = g_corner_pass;
                    for (int i = 0; i < 3; ++i) c[i] = &corner(ci[i]);
                }
            }
        }
        if (x1 >= W || y1 >= H) {
            ++F.approx[kOutside];
            x1 = std::min(x1, W - 1), y1 = std::min(y1, H - 1);
            if (x1 < x0 || y1 < y0) {
                if (g_epoch != epoch) break;
                continue;
            }
        }
        g_clear.on = false;  // ends a run of clear strips (shape(): not the same kind; shapes() sends no clear or cover sprites here)
        if (g_log && s.src.kind == 1 && mip_top && mip[0]) {
            const u32 m = mip[0];
            const float kk = static_cast<float>(static_cast<int32_t>(((m >> 16) & 0xFFFu) << 20) >> 20) / 16.0f;
            float lod_max = kk;
            if (!(m & 1u) && !(w[0] & 16u)) {
                float qmin = 1e30f;
                for (int i = 0; i < per; ++i) qmin = std::min(qmin, std::fabs(v[ci[i]].q));
                lod_max = qmin > 0.0f ? std::log2(1.0f / qmin) * static_cast<float>(1u << ((m >> 8) & 3u)) + kk : 1e9f;
            }
            const u32 need = lod_max <= -0.01f ? 1u : std::min<u32>(mip_top + 1u, static_cast<u32>(std::floor(lod_max)) + 2u);
            ++F.mipneed[std::min<u32>(need, 7u)];
            ++F.mipdraws;
        }
        const u32 first = used;
        Vtx *out = g_verts.data() + first;
        if (sprite) {  // shape(): two triangles, colour, depth, fog and q from the second corner, s and t (u and v) from the corners they belong to
            const Vertex &a = v[ci[0]], &b = v[ci[1]];
            const float px0 = (static_cast<float>(a.x) - ofx) / 16.0f, py0 = (static_cast<float>(a.y) - ofy) / 16.0f;
            const float px1 = (static_cast<float>(b.x) - ofx) / 16.0f, py1 = (static_cast<float>(b.y) - ofy) / 16.0f;
            Vertex ab = a, ba = b;
            ab.s = b.s, ab.u = b.u;
            ba.s = a.s, ba.u = a.u;
            put_vtx(out[0], px0, py0, b, a, b.z, b.f, b.q, sxs, sys, zmax, zscale, fst, param);
            put_vtx(out[1], px1, py0, b, ab, b.z, b.f, b.q, sxs, sys, zmax, zscale, fst, param);
            put_vtx(out[2], px0, py1, b, ba, b.z, b.f, b.q, sxs, sys, zmax, zscale, fst, param);
            out[3] = out[1];
            put_vtx(out[4], px1, py1, b, b, b.z, b.f, b.q, sxs, sys, zmax, zscale, fst, param);
            out[5] = out[2];
        } else {
            out[0] = c[0]->o, out[1] = c[1]->o, out[2] = c[2]->o;
            if (!iip) out[0].rgba = out[1].rgba = out[2].rgba;
        }
        used += nv;
        if (cur) {
            cur->count += nv;
        } else {
            ensure_pass(s);
            Op &pass = g_ops[static_cast<std::size_t>(g_open_pass)];
            u32 sc[4] = {static_cast<u32>(std::max(s.sc[0], 0)), static_cast<u32>(std::max(s.sc[1], 0)), 0, 0};
            const u32 ex = static_cast<u32>(std::max(std::min(s.sc[2] + 1, W), 0)), ey = static_cast<u32>(std::max(std::min(s.sc[3] + 1, H), 0));
            sc[2] = ex > sc[0] ? ex - sc[0] : 0;
            sc[3] = ey > sc[1] ? ey - sc[1] : 0;
            bool merged = false;
            if (pass.draw_count) {
                DrawRec &last = g_draws.back();
                if (last.k1 == s.kp1 && last.k2 == s.kp2 && last.group == group && last.blend == s.blend && last.zuse == s.needs_depth && std::equal(sc, sc + 4, last.sc) &&
                    last.first + last.count == first) {
                    last.count += nv;
                    merged = true;
                }
            }
            if (!merged) {
                g_split_tris.clear();
                DrawRec r;
                r.k1 = s.kp1, r.k2 = s.kp2, r.group = group, r.blend = s.blend;
                std::copy(sc, sc + 4, r.sc);
                r.first = first, r.count = nv;
                r.res = res;
                r.zuse = s.needs_depth;
                r.zread = s.zread;
                g_draws.push_back(r);
                ++pass.draw_count;
            }
            cur = &g_draws.back();
        }
        if (s.cwrite_any) {
            ct.valid_w = std::max(ct.valid_w, x1 + 1), ct.valid_h = std::max(ct.valid_h, y1 + 1);
            const int a = x0 >> csx, b = y0 >> csy, cc = x1 >> csx, d = y1 >> csy;
            if (a < cpr[0] || b < cpr[1] || cc > cpr[2] || d > cpr[3]) {
                for_pages(ct.psm, ct.bp, ct.bw, x0, y0, x1, y1, [&](u32 page) {
                    take_page(s.color, page);
                    ++g_page_write[page & (kPages - 1)];
                });
                cpr[0] = a, cpr[1] = b, cpr[2] = cc, cpr[3] = d;
            }
            ++ct.writes;
            if (ct.snap) note_write(ct, x0, y0, x1, y1, s.awrite_any), --ct.writes;
        }
        if (s.zwrite_any && dt) {
            dt->valid_w = std::max(dt->valid_w, x1 + 1), dt->valid_h = std::max(dt->valid_h, y1 + 1);
            const int a = x0 >> dsx, b = y0 >> dsy, cc = x1 >> dsx, d = y1 >> dsy;
            if (a < dpr[0] || b < dpr[1] || cc > dpr[2] || d > dpr[3]) {
                for_pages(dt->psm, dt->bp, dt->bw, x0, y0, x1, y1, [&](u32 page) {
                    take_page(s.depth, page);
                    ++g_page_write[page & (kPages - 1)];
                });
                dpr[0] = a, dpr[1] = b, dpr[2] = cc, dpr[3] = d;
            }
        }
        ++F.drawn;
        F.vertices += nv;
        if (g_epoch != epoch) break;  // the next primitive's state is worked out again (shape(): prepare_state)
    }
    g_verts.resize(used);
    return k;
}
}  // namespace

void shapes(const State &st, const u32 *w, const u32 *mip, u32 mip_top, u64 serial, const Vertex *v, int nverts, const u16 *idx, int prims, int per) {
    const u32 prim = static_cast<u32>(st.prim) & 7u;
    const auto one = [&](int i) {  // primitive i through shape()
        if (!idx) {
            shape(st, w, mip, mip_top, serial, v + static_cast<std::size_t>(i) * static_cast<std::size_t>(per), per);
            return;
        }
        Vertex c[3];
        for (int j = 0; j < per; ++j) c[j] = v[idx[i * per + j]];
        shape(st, w, mip, mip_top, serial, c, per);
    };
    const bool kind_ok = (per == 3 && prim >= 3 && prim <= 5) || (per == 2 && prim == 6);
    if (!batch_on() || !kind_ok || g_dump_now || prov::g_on || page_log() >= 0 || skip_tbp_value() >= 0) {
        if (drawtime::g_on) drawtime::g_slow[3] += static_cast<u64>(prims);
        for (int i = 0; i < prims; ++i) one(i);
        return;
    }
    DrawState &s = g_ds;
    int i = 0;
    while (i < prims) {
        {
            const auto t0 = g_log ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            const prof::Scope shape_scope(prof::kRShape);
            const drawtime::Span shape_span(drawtime::kShape);
            prepare_state(s, st, w, mip, mip_top, serial, prim);
            const int rest = prims - i;
            if (s.skip == 2) {  // not drawn (the reason counted), nor anything after it in the run: nothing it does changes the state
                F.shapes += static_cast<u64>(rest);
                F.fallback[s.reason] += static_cast<u64>(rest);
                return;
            }
            if (s.skip == 1) {
                F.shapes += static_cast<u64>(rest);
                F.skipped += static_cast<u64>(rest);
                return;
            }
            // view 4: only without mip levels (shape() looks at each level's pages then); prim_run checks each one's texel pages
            bool fast = (s.src.view != 4 || !(mip_top && mip[0])) && !(s.src.kind && s.src.target >= 0 && s.src.target == s.color) && s.kp2 == 0;
            if (fast && per == 2) {
                // sprites that could be part of a clear run or a cover run (shape(): clear_part, cover_part) go one by one
                const Target &ct = *g_targets[static_cast<std::size_t>(s.color)];
                const Target *dt = s.needs_depth ? g_targets[static_cast<std::size_t>(s.depth)].get() : nullptr;
                const u32 flags = w[0];
                const bool clear_part = !(flags & 2u) && !s.k1.blend && !(flags & 4u) && s.k1.atest == 0 && (s.k1.mask == 0 || s.color_full) && (s.k1.mask || s.k1.zwrite) &&
                                        (!s.k1.zwrite || (s.k1.zcmp == WGPUCompareFunction_Always && dt && dt->w == ct.w && dt->h == ct.h));
                const bool cover_part = (flags & 2u) && s.src.kind && s.src.target != s.color && !s.k1.blend && s.k1.atest == 0 && s.color_full;
                fast = !clear_part && !cover_part && !g_stream_order;
            }
            if (fast) {
                const int took = idx ? prim_run(s, st, w, mip, mip_top, v, nverts, idx + static_cast<std::size_t>(i) * static_cast<std::size_t>(per), rest, per)
                                     : prim_run(s, st, w, mip, mip_top, v + static_cast<std::size_t>(i) * static_cast<std::size_t>(per), rest * per, nullptr, rest, per);
                if (g_log) F.cpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                i += took;
                if (took) continue;  // none: a view-4 primitive whose texels leave its target, for shape()
            }
        }
        // a draw that reads its own target, reads through memory addresses (view 4), is drawn twice (alpha test) or may clear: shape() by shape
        if (drawtime::g_on) ++drawtime::g_slow[s.kp2 ? 0 : s.src.view == 4 ? 2 : per == 2 ? 3 : 1];
        one(i);
        ++i;
    }
}

void upload(const u32 *rec, const u8 *bytes, std::size_t size) {
    const u32 dbp = rec[0], dbw = rec[1], dpsm = rec[2], dsax = rec[3], dsay = rec[4], rrw = rec[5], done = rec[6], pixels = rec[7], lead = rec[9];
    if (!rrw || !pixels) return;
    u64 uid = hash_bytes(bytes, size);
    uid = mix(uid, dpsm | dbw << 8 | lead << 16);
    uid = mix(uid, dsax | dsay << 16);
    uid = mix(uid, rrw | static_cast<u64>(done) << 16 | static_cast<u64>(pixels) << 40);
    const int first = static_cast<int>(done / rrw), last = static_cast<int>((done + pixels - 1) / rrw);
    const int full_first = static_cast<int>((done + rrw - 1) / rrw), full_last = static_cast<int>((done + pixels) / rrw) - 1;
    const int x0 = static_cast<int>(dsax), x1 = static_cast<int>(dsax + rrw - 1), y0 = static_cast<int>(dsay) + first, y1 = static_cast<int>(dsay) + last;
    for_pages(dpsm, dbp, dbw, x0, y0, x1, y1, [&](u32 page) {
        if (g_mem_read_submit[page & (kPages - 1)] == g_submit) ++g_mem_read_hits;
        lose_page(page);
        g_page_stale[page] = false;
    });
    const int fy0 = static_cast<int>(dsay) + full_first, fy1 = static_cast<int>(dsay) + full_last;
    for_blocks(dpsm, dbp, dbw, x0, y0, x1, y1, [&](u32 b, int bx, int by, u32 bwp, u32 bhp) {
        const bool full = bx >= x0 && bx + static_cast<int>(bwp) - 1 <= x1 && by >= fy0 && by + static_cast<int>(bhp) - 1 <= fy1;
        const u64 rel = (b - dbp) & (kBlocks - 1);
        const u64 nid = full ? mix(uid, rel) : mix(mix(g_block[b], uid), rel);
        if (g_mem_read_submit[(b / 32) & (kPages - 1)] == g_submit && nid != g_block[b]) ++g_mem_read_changes;   // a block a draw reads from memory gets other content
        g_block[b] = nid;
        if (full) g_blk_stale[b] = 0;
        touch_page(b / 32);
    });
    ++g_epoch;
}

void clut(u32 cbp, u32 cpsm, u32 csm, u32 csa, bool eight, u32 cbw, u32 cou, u32 cov) {
    u64 h = mix(0xC1ull, cpsm | csm << 8 | csa << 16 | (eight ? 1u : 0u) << 24);
    bool stale = false;
    const auto visit = [&](u32 b, int, int, u32, u32) {
        h = mix(h, g_block[b]);
        stale = stale || g_blk_stale[b] || g_owner[b / 32] >= 0;
    };
    if (csm) for_blocks(cpsm, cbp, cbw, static_cast<int>(cou * 16), static_cast<int>(cov), static_cast<int>(cou * 16 + (eight ? 255 : 15)), static_cast<int>(cov), visit);
    else for_blocks(cpsm, cbp, 1, 0, 0, eight ? 15 : 7, eight ? 15 : 1, visit);
    if (stale) h = mix(h, ++g_fresh);
    g_clut_id = eight ? h : mix(g_clut_id, h);
    ++g_epoch;
}

void pre_blocks(bool clut, const u32 *rec, std::vector<u32> &out) {
    const auto put = [&](u32 b, int, int, u32, u32) { out.push_back(b); };
    if (clut) {  // slot, next, cbp, cpsm, csm, csa, eight, cbw, cou, cov
        const u32 cbp = rec[2], cpsm = rec[3], csm = rec[4], cbw = rec[7], cou = rec[8], cov = rec[9];
        const bool eight = rec[6] != 0;
        if (csm) for_blocks(cpsm, cbp, cbw, static_cast<int>(cou * 16), static_cast<int>(cov), static_cast<int>(cou * 16 + (eight ? 255 : 15)), static_cast<int>(cov), put);
        else for_blocks(cpsm, cbp, 1, 0, 0, eight ? 15 : 7, eight ? 15 : 1, put);
        return;
    }
    const u32 dbp = rec[0], dbw = rec[1], dpsm = rec[2], dsax = rec[3], dsay = rec[4], rrw = rec[5], done = rec[6], pixels = rec[7];
    if (!rrw || !pixels) return;
    const int first = static_cast<int>(done / rrw), last = static_cast<int>((done + pixels - 1) / rrw);
    for_blocks(dpsm, dbp, dbw, static_cast<int>(dsax), static_cast<int>(dsay) + first, static_cast<int>(dsax + rrw - 1), static_cast<int>(dsay) + last, put);
}

void local_copy(u32 dbp, u32 dbw, u32 dpsm, int x0, int y0, int x1, int y1) {
    ++F.approx[kStaleTex];
    for_pages(dpsm, dbp, dbw, x0, y0, x1, y1, [&](u32 page) { lose_page(page); });
    for_blocks(dpsm, dbp, dbw, x0, y0, x1, y1, [&](u32 b, int, int, u32, u32) {
        g_block[b] = mix(0xC0B1ull, ++g_fresh);
        touch_page(b / 32);
    });
    ++g_epoch;
}

void before_display(u32 fbp, u32 fbw, u32 psm, u32 x, u32 y, u32 w, u32 h) {
    // the target that owns the page of the picture's first pixel, when it is that picture's buffer
    u32 f[5];
    C.fmt_words(psm, fbp, fbw, f);
    const int owner = g_owner[(block_at(f, x, y) / 32) & (kPages - 1)];
    if (owner < 0) return;
    Target &t = *g_targets[static_cast<std::size_t>(owner)];
    if (t.depth || t.bp != fbp || t.bw != fbw) return;
    const u32 x1 = std::min(x + w, t.w), y1 = std::min(y + h, t.h);
    if (x1 <= x || y1 <= y) return;
    Op &op = add_op(kOpStore);
    const u32 words[10] = {t.bp, t.bw, t.last_psm, x, y, x1 - x, y1 - y, 0, fbits(t.sc.x), fbits(t.sc.y)};
    op.param = C.params(words, 10);
    op.group = t.s->sync_group;
    op.rd = t.s->id;
    op.tgt = owner;
    op.w = x1 - x, op.h = y1 - y;
    for_pages(t.psm, t.bp, t.bw, static_cast<int>(x), static_cast<int>(y), static_cast<int>(x1) - 1, static_cast<int>(y1) - 1, [&](u32 page) {
        g_page_stale[page] = false;
        for (u32 b = page * 32; b < page * 32 + 32; ++b) g_blk_stale[b] = 0;
        touch_page(page);
    });
    ++F.stores;
}

namespace {
struct ShowPending {
    bool on = false;
    int owner = -1;
    u32 x = 0, y = 0, w = 0, h = 0;  // card texels
} g_show;
const bool g_present_memory = os::env("PS2_HWR_PRESENT") == "memory";
}  // namespace

bool display_target(u32 fbp, u32 fbw, u32 psm, u32 x, u32 y, u32 w, u32 h, u32 &card_w, u32 &card_h) {
    g_show.on = false;
    if (!g_on || g_present_memory || !w || !h) return false;
    u32 f[5];
    C.fmt_words(psm, fbp, fbw, f);
    const int owner = g_owner[(block_at(f, x, y) / 32) & (kPages - 1)];
    if (owner < 0) return false;
    const Target &t = *g_targets[static_cast<std::size_t>(owner)];
    if (!t.alive || t.depth || t.bp != fbp || t.bw != fbw || !t.s) return false;
    const u32 tw = card_x(t.w, t.sc), th = card_y(t.h, t.sc);
    const u32 x0 = card_at_x(x, t.sc), y0 = card_at_y(y, t.sc);
    const u32 x1 = std::min(card_at_x(std::min(x + w, t.w), t.sc), tw), y1 = std::min(card_at_y(std::min(y + h, t.h), t.sc), th);
    if (x1 <= x0 || y1 <= y0) return false;
    g_show = {true, owner, x0, y0, x1 - x0, y1 - y0};
    card_w = x1 - x0;
    card_h = y1 - y0;
    return true;
}

void show(WGPUTexture dst) {
    if (!g_show.on) return;
    g_show.on = false;
    const Target &t = *g_targets[static_cast<std::size_t>(g_show.owner)];
    Op &op = add_op(kOpShow);
    op.tex = t.s->tex;
    op.tex2 = dst;
    op.offset = g_show.x, op.pitch = g_show.y;
    op.w = g_show.w, op.h = g_show.h;
    op.rd = t.s->id;
    op.tgt = g_show.owner;
    close_pass();
}

void sync_all() {
    if (!g_on) return;
    g_clear.on = false;
    for (std::size_t i = 0; i < g_targets.size(); ++i) {
        Target *t = g_targets[i].get();
        if (!t || !t->alive || !t->valid_w || !t->valid_h) continue;
        const u32 w = static_cast<u32>(std::min<int>(t->valid_w, static_cast<int>(t->w))), h = static_cast<u32>(std::min<int>(t->valid_h, static_cast<int>(t->h)));
        Op &op = add_op(t->depth ? kOpStoreZ : kOpStore);
        const float scale = static_cast<float>(static_cast<double>(t->depth ? (t->psm == 0x30 ? 0xFFFFFFFFu : t->psm == 0x31 ? 0xFFFFFFu : 0xFFFFu) : 0u) + 1.0);
        u32 sbits;
        std::memcpy(&sbits, &scale, 4);
        const u32 words[10] = {t->bp, t->bw, t->depth ? t->psm : t->last_psm, 0, 0, w, h, sbits, fbits(t->sc.x), fbits(t->sc.y)};
        op.param = C.params(words, 10);
        op.group = t->s->sync_group;
        op.rd = t->s->id;
        op.tgt = static_cast<int>(i);
        op.w = w, op.h = h;
        ++F.stores;
    }
    std::fill(std::begin(g_page_stale), std::end(g_page_stale), false);
    std::fill(std::begin(g_blk_stale), std::end(g_blk_stale), u8{0});
    touch_all();
}

void state_loaded() {
    if (!g_on) return;
    g_clear.on = false;
    for (std::size_t i = 0; i < g_targets.size(); ++i)
        if (g_targets[i] && g_targets[i]->alive) release_target(static_cast<int>(i));
    std::fill(std::begin(g_owner), std::end(g_owner), -1);
    std::fill(std::begin(g_page_stale), std::end(g_page_stale), false);
    std::fill(std::begin(g_blk_stale), std::end(g_blk_stale), u8{0});
    touch_all();
    const u64 gen = ++g_fresh;
    for (u32 b = 0; b < kBlocks; ++b) g_block[b] = mix(mix(0x57A7Eull, gen), b);
    g_clut_id = mix(0xC1A7ull, gen);
    ++g_epoch;
    close_pass();
}

bool pending() { return g_on && !g_ops.empty(); }

bool before(u32 command) { return g_next_item < g_items.size() && g_items[g_next_item].pos <= command; }

namespace {

// ---------------------------------------------------------------- the schedule (docs/hw-renderer-design.md 3.3)
// The ops of the submit form a graph: an op that reads or writes a resource (a surface or a cache texture) comes after the op that wrote it
// last, and an op that writes it after the ops that read the previous picture. Loads and stores of graphics memory keep their place among
// gpu.cpp's compute commands ("anchored"); passes and copies ("floating") go as late as they can: just before the first anchored op that
// needs them, else after all compute commands. Each such group is ordered by a list schedule that keeps drawing into the same colour surface
// as long as it can (nodes into one surface become one render pass), and when it has to change, first takes a surface whose remaining nodes
// can all be drawn now (the shadow buffers, the blur buffer) before one that would have to be broken again later (the main picture).
struct Graph {
    std::vector<std::vector<int>> succ, pred;
};

void build_graph(Graph &g) {
    const std::size_t n = g_ops.size();
    g.succ.assign(n, {});
    g.pred.assign(n, {});
    std::unordered_map<u64, int> writer;
    std::unordered_map<u64, std::vector<int>> readers;
    std::vector<u64> rd, wr;
    const auto edge = [&](int a, int b) {
        if (a == b) return;
        g.succ[static_cast<std::size_t>(a)].push_back(b);
        g.pred[static_cast<std::size_t>(b)].push_back(a);
    };
    for (std::size_t i = 0; i < n; ++i) {
        const Op &op = g_ops[i];
        rd.clear(), wr.clear();
        if (op.kind == kOpPass) {
            for (u32 d = op.first_draw; d < op.first_draw + op.draw_count; ++d)
                if (g_draws[d].res && std::find(rd.begin(), rd.end(), g_draws[d].res) == rd.end()) rd.push_back(g_draws[d].res);
            if (op.cs) wr.push_back(op.cs->id);
            if (op.ds) wr.push_back(op.ds->id);
        } else {
            if (op.rd) rd.push_back(op.rd);
            if (op.wr) wr.push_back(op.wr);
        }
        const int me = static_cast<int>(i);
        for (const u64 r : rd) {
            if (const auto it = writer.find(r); it != writer.end()) edge(it->second, me);
            readers[r].push_back(me);
        }
        for (const u64 w : wr) {
            if (const auto it = writer.find(w); it != writer.end()) edge(it->second, me);
            if (const auto it = readers.find(w); it != readers.end()) {
                for (const int r : it->second) edge(r, me);
                it->second.clear();
            }
            writer[w] = me;
        }
    }
}

bool anchored(const Op &op) { return op.kind == kOpLoad || op.kind == kOpStore || op.kind == kOpStoreZ; }

// one group of floating ops (all with the same deadline) in schedule order; passes are runs of pass ops (`breaks` marks the first op of each)
void order_group(const Graph &g, const std::vector<int> &ops, std::vector<int> &out, std::vector<u8> &starts) {
    const std::size_t n = g_ops.size();
    static std::vector<int> inpred;
    static std::vector<u8> in_group, done;
    inpred.assign(n, 0);
    in_group.assign(n, 0);
    done.assign(n, 0);
    for (const int i : ops) in_group[static_cast<std::size_t>(i)] = 1;
    for (const int i : ops)
        for (const int p : g.pred[static_cast<std::size_t>(i)])
            if (in_group[static_cast<std::size_t>(p)]) ++inpred[static_cast<std::size_t>(i)];
    std::vector<int> ready;  // kept in record order
    for (const int i : ops)
        if (!inpred[static_cast<std::size_t>(i)]) ready.push_back(i);
    const auto take = [&](std::size_t k, bool start) {
        const int i = ready[k];
        ready.erase(ready.begin() + static_cast<std::ptrdiff_t>(k));
        done[static_cast<std::size_t>(i)] = 1;
        out.push_back(i);
        starts.push_back(start ? 1 : 0);
        for (const int s : g.succ[static_cast<std::size_t>(i)]) {
            if (!in_group[static_cast<std::size_t>(s)]) continue;
            if (--inpred[static_cast<std::size_t>(s)] == 0) ready.insert(std::lower_bound(ready.begin(), ready.end(), s), s);
        }
    };
    // all remaining nodes of surface cs can be drawn one after another from now on (no other op has to come in between)
    std::vector<u8> reach(n, 0);
    const auto complete = [&](const Surf *cs) {
        bool all = true;
        for (const int i : ops) {
            const Op &op = g_ops[static_cast<std::size_t>(i)];
            if (done[static_cast<std::size_t>(i)] || op.kind != kOpPass || op.cs != cs) continue;
            bool ok = true;
            for (const int p : g.pred[static_cast<std::size_t>(i)])
                if (in_group[static_cast<std::size_t>(p)] && !done[static_cast<std::size_t>(p)] && !reach[static_cast<std::size_t>(p)]) {
                    ok = false;
                    break;
                }
            reach[static_cast<std::size_t>(i)] = ok;
            if (!ok) all = false;
        }
        for (const int i : ops) reach[static_cast<std::size_t>(i)] = 0;
        return all;
    };
    const Surf *cur_c = nullptr, *cur_d = nullptr;
    bool open = false;
    while (!ready.empty()) {
        // the open render pass goes on with a ready node of its surface
        if (open) {
            std::size_t k = 0;
            for (; k < ready.size(); ++k) {
                const Op &op = g_ops[static_cast<std::size_t>(ready[k])];
                if (op.kind == kOpPass && op.cs == cur_c && (!op.ds || !cur_d || op.ds == cur_d)) break;
            }
            if (k < ready.size()) {
                if (g_ops[static_cast<std::size_t>(ready[k])].ds) cur_d = g_ops[static_cast<std::size_t>(ready[k])].ds;
                take(k, false);
                continue;
            }
            open = false;
        }
        // copies that are ready go now (between render passes)
        bool copied = false;
        for (std::size_t k = 0; k < ready.size();)
            if (g_ops[static_cast<std::size_t>(ready[k])].kind != kOpPass) take(k, true), copied = true;
            else ++k;
        if (copied) continue;
        // a new render pass: a surface whose nodes can all be drawn now, else the first ready node
        std::size_t pick = 0;
        for (std::size_t k = 0; k < ready.size(); ++k) {
            const Surf *cs = g_ops[static_cast<std::size_t>(ready[k])].cs;
            bool seen = false;
            for (std::size_t j = 0; j < k; ++j) seen = seen || g_ops[static_cast<std::size_t>(ready[j])].cs == cs;
            if (!seen && complete(cs)) {
                pick = k;
                break;
            }
        }
        cur_c = g_ops[static_cast<std::size_t>(ready[pick])].cs;
        cur_d = g_ops[static_cast<std::size_t>(ready[pick])].ds;
        open = true;
        take(pick, true);
    }
}

void add_pass_item(u32 pos, const std::vector<int> &ops, std::size_t a, std::size_t b) {
    Item it;
    it.pos = pos;
    it.first = static_cast<u32>(g_pass_ops.size());
    it.n = static_cast<u32>(b - a);
    for (std::size_t k = a; k < b; ++k) {
        const Op &op = g_ops[static_cast<std::size_t>(ops[k])];
        g_pass_ops.push_back(ops[k]);
        it.cs = op.cs;
        if (op.ds && !it.ds) {
            it.ds = op.ds;
            it.clear_d = op.clear_d;
            it.cz = op.cz;
        }
        if (k == a && op.clear_c) {
            it.clear_c = true;
            std::copy(op.cc, op.cc + 4, it.cc);
        }
    }
    g_items.push_back(it);
}

void schedule() {
    const prof::Scope schedule_scope(prof::kRSchedule);
    const auto t0 = std::chrono::steady_clock::now();
    g_items.clear();
    g_pass_ops.clear();
    g_final.clear();
    g_vorder.clear();
    const std::size_t n = g_ops.size();
    std::vector<int> seq;
    std::vector<u8> starts;
    std::vector<u32> pos_of;  // the encode position of each op in seq
    if (g_stream_order) {
        for (std::size_t i = 0; i < n; ++i) {
            seq.push_back(static_cast<int>(i));
            starts.push_back(1);
            pos_of.push_back(g_ops[i].pos);
        }
    } else {
        // depth that a node writes and that nothing uses afterwards (no target holds the surface, no later node attaches it, the node's own
        // draws do not test it): the node draws without depth, so it can share a render pass with other depth (the glow composite before the
        // HUD's depth clear, the display copy's dead depth writes)
        bool store_z = false;
        std::unordered_map<const Surf *, std::size_t> zlast;
        for (std::size_t i = 0; i < n; ++i) {
            store_z = store_z || g_ops[i].kind == kOpStoreZ;
            if (g_ops[i].kind == kOpPass && g_ops[i].ds) zlast[g_ops[i].ds] = i;
        }
        for (std::size_t i = 0; i < n && !store_z; ++i) {
            Op &op = g_ops[i];
            if (op.kind != kOpPass || !op.ds || !op.ds->retired || zlast[op.ds] != i || op.clear_d) continue;
            bool zread = false;
            for (u32 d = op.first_draw; d < op.first_draw + op.draw_count; ++d) zread = zread || g_draws[d].zread;
            if (zread) continue;
            for (u32 d = op.first_draw; d < op.first_draw + op.draw_count; ++d) g_draws[d].zuse = false;
            op.ds = nullptr;
            ++F.dead_depth;
        }
        Graph g;
        build_graph(g);
        // a load whose surface nothing draws into, reads or copies afterwards and that no target holds any more is not done: a target
        // made over current memory and cleared (renamed) right after (the menus: ~50 MB a frame at the screen's resolution)
        std::vector<u8> dead(n, 0);
        for (std::size_t i = 0; i < n; ++i) {
            const Op &op = g_ops[i];
            if (op.kind != kOpLoad || !g.succ[i].empty()) continue;
            for (const auto &sf : g_surfs)
                if (sf->id == op.wr) {
                    if (sf->retired) dead[i] = 1, ++F.approx[kDeadLoad];
                    break;
                }
        }
        std::vector<int> anchor_index(n, -1), anchors;
        for (std::size_t i = 0; i < n; ++i)
            if (anchored(g_ops[i]) && !dead[i]) anchor_index[i] = static_cast<int>(anchors.size()), anchors.push_back(static_cast<int>(i));
        const int end = static_cast<int>(anchors.size());
        std::vector<int> dl(n, end);
        for (std::size_t r = n; r-- > 0;) {
            if (anchor_index[r] >= 0) {
                dl[r] = anchor_index[r];
                continue;
            }
            for (const int s : g.succ[r]) dl[r] = std::min(dl[r], dl[static_cast<std::size_t>(s)]);
        }
        std::vector<std::vector<int>> groups(static_cast<std::size_t>(end) + 1);
        for (std::size_t i = 0; i < n; ++i)
            if (anchor_index[i] < 0 && !dead[i]) groups[static_cast<std::size_t>(dl[i])].push_back(static_cast<int>(i));
        for (int k = 0; k <= end; ++k) {
            const u32 pos = k < end ? g_ops[static_cast<std::size_t>(anchors[static_cast<std::size_t>(k)])].pos : ~0u;
            const std::size_t from = seq.size();
            order_group(g, groups[static_cast<std::size_t>(k)], seq, starts);
            for (std::size_t j = from; j < seq.size(); ++j) pos_of.push_back(pos);
            if (k < end) {
                seq.push_back(anchors[static_cast<std::size_t>(k)]);
                starts.push_back(1);
                pos_of.push_back(pos);
            }
        }
    }
    // items: runs of pass ops become one render pass each
    for (std::size_t k = 0; k < seq.size();) {
        const Op &op = g_ops[static_cast<std::size_t>(seq[k])];
        if (op.kind != kOpPass) {
            Item it;
            it.pos = pos_of[k];
            it.op = seq[k];
            g_items.push_back(it);
            ++k;
            continue;
        }
        std::size_t e = k + 1;
        while (e < seq.size() && !starts[e] && g_ops[static_cast<std::size_t>(seq[e])].kind == kOpPass) ++e;
        add_pass_item(pos_of[k], seq, k, e);
        k = e;
    }
    // the last use of each surface: a pass whose attachment is not used again and no target holds it stores nothing
    std::unordered_map<const Surf *, std::size_t> last;
    for (std::size_t i = 0; i < g_items.size(); ++i) {
        const Item &it = g_items[i];
        if (it.op >= 0) continue;
        if (it.cs) last[it.cs] = i;
        if (it.ds) last[it.ds] = i;
    }
    std::unordered_map<u64, const Surf *> by_id;  // surfaces read by draws and copies (they are not pass attachments there)
    for (const auto &s : g_surfs) by_id[s->id] = s.get();
    for (std::size_t i = 0; i < g_items.size(); ++i) {
        const Item &it = g_items[i];
        if (it.op >= 0) {
            const auto f = by_id.find(g_ops[static_cast<std::size_t>(it.op)].rd);
            if (f != by_id.end()) last[f->second] = std::max(last[f->second], i);
            continue;
        }
        for (u32 k = 0; k < it.n; ++k) {
            const Op &op = g_ops[static_cast<std::size_t>(g_pass_ops[it.first + k])];
            for (u32 d = op.first_draw; d < op.first_draw + op.draw_count; ++d)
                if (g_draws[d].res) {
                    const auto f = by_id.find(g_draws[d].res);
                    if (f != by_id.end()) last[f->second] = std::max(last[f->second], i);
                }
        }
    }
    if (g_reorder_verts) g_vorder.reserve(g_verts.size());
    for (std::size_t i = 0; i < g_items.size(); ++i) {
        Item &it = g_items[i];
        if (it.op >= 0) continue;
        it.store_c = !(it.cs->retired && last[it.cs] == i);
        it.store_d = it.ds && !(it.ds->retired && last[it.ds] == i);
        // the merged draw calls: the nodes' draws one after another; neighbours with the same state become one call
        it.draw_first = static_cast<u32>(g_final.size());
        for (u32 k = 0; k < it.n; ++k) {
            const Op &op = g_ops[static_cast<std::size_t>(g_pass_ops[it.first + k])];
            for (u32 d = op.first_draw; d < op.first_draw + op.draw_count; ++d) {
                const DrawRec &r = g_draws[d];
                if (r.zread) it.zread = true;
                u32 at = r.first;
                if (g_reorder_verts) {
                    at = static_cast<u32>(g_vorder.size());
                    g_vorder.insert(g_vorder.end(), g_verts.begin() + r.first, g_verts.begin() + r.first + r.count);
                }
                if (g_final.size() > it.draw_first) {
                    DrawRec &l = g_final.back();
                    if (!l.k2 && !r.k2 && l.k1 == r.k1 && l.group == r.group && l.blend == r.blend && l.zuse == r.zuse && std::equal(l.sc, l.sc + 4, r.sc) &&
                        (g_reorder_verts || l.first + l.count == r.first)) {
                        l.count += r.count;
                        continue;
                    }
                }
                DrawRec f = r;
                f.first = at;
                g_final.push_back(f);
            }
        }
        it.draw_n = static_cast<u32>(g_final.size()) - it.draw_first;
    }
    F.nodes += n;
    F.sched_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (g_dump_now) {
        const char *kn[] = {"pass", "copylevel", "load", "store", "storez", "selfcopy", "show"};
        std::fprintf(stderr, "[hwr-dump] submit %llu: %zu ops -> %zu items\n", static_cast<unsigned long long>(g_submit), n, g_items.size());
        for (const Item &it : g_items) {
            if (it.op >= 0) {
                const Op &op = g_ops[static_cast<std::size_t>(it.op)];
                std::fprintf(stderr, "  %-9s pos %u t%d\n", kn[op.kind], it.pos, op.tgt);
                continue;
            }
            const Op &f = g_ops[static_cast<std::size_t>(g_pass_ops[it.first])];
            const Target *t = f.tgt >= 0 && g_targets[static_cast<std::size_t>(f.tgt)] ? g_targets[static_cast<std::size_t>(f.tgt)].get() : nullptr;
            std::fprintf(stderr, "  pass      pos %u t%d %04x/%u c%llu d%llu nodes %u draws %u clear %d%d store %d%d zread %d\n", it.pos, f.tgt, t ? t->bp : 0, t ? t->bw : 0,
                         static_cast<unsigned long long>(it.cs->id), it.ds ? static_cast<unsigned long long>(it.ds->id) : 0ull, it.n, it.draw_n, it.clear_c, it.clear_d,
                         it.store_c, it.store_d, it.zread);
        }
    }
}

}  // namespace

void prepare(bool defer_writes) {
    const prof::Scope prepare_scope(prof::kRPrepare);
    schedule();
    const auto grow = [](WGPUBuffer &buf, u64 &size, u64 need, WGPUBufferUsage usage) {
        if (need <= size && buf) return false;
        if (buf) wgpuBufferRelease(buf);
        size = std::max<u64>(std::max<u64>(size * 2, need), 1u << 16);
        WGPUBufferDescriptor d = WGPU_BUFFER_DESCRIPTOR_INIT;
        d.size = size;
        d.usage = usage;
        buf = wgpuDeviceCreateBuffer(C.device, &d);
        return true;
    };
    const u64 vbytes = vsource().size() * sizeof(Vtx), pbytes = std::max<u64>(g_params.size(), 1) * 64;
    grow(g_vbuf, g_vbuf_size, vbytes, WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst);
    if (grow(g_pbuf, g_pbuf_size, pbytes, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst) || !g_param_group) {
        if (g_param_group) wgpuBindGroupRelease(g_param_group);
        WGPUBindGroupEntry e[4] = {WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT};
        e[0].binding = 0;
        e[0].buffer = g_pbuf;
        e[0].size = WGPU_WHOLE_SIZE;
        e[1].binding = 1;
        e[1].buffer = C.clut;
        e[1].size = WGPU_WHOLE_SIZE;
        e[2].binding = 2;
        e[2].buffer = g_addr;
        e[2].size = WGPU_WHOLE_SIZE;
        e[3].binding = 3;
        e[3].buffer = C.vram ? C.vram : g_addr;  // no memory (PS2_SCALE > 1): the levels are never read from it (shape() keeps them on the old way)
        e[3].size = WGPU_WHOLE_SIZE;
        WGPUBindGroupDescriptor d = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        d.layout = g_params_layout;
        d.entryCount = 4;
        d.entries = e;
        g_param_group = wgpuDeviceCreateBindGroup(C.device, &d);
    }
    if (defer_writes) return;
    if (vbytes) wgpuQueueWriteBuffer(C.queue, g_vbuf, 0, vsource().data(), vbytes);
    if (!g_params.empty()) wgpuQueueWriteBuffer(C.queue, g_pbuf, 0, g_params.data(), g_params.size() * 64);
}

namespace {

// PS2_GPU_PROFILE=1 (or PS2_HWR_TIME=1): the card's time of every piece of the renderer's work, from timestamps the card writes at the start
// and end of each render pass and compute pass (copies are bracketed by two empty compute passes). Each piece gets a name: its kind, the
// target (address, format, size in GS pixels and card texels) and, for a render pass, its draws and whether it loads/stores. The gaps
// between pieces are the gpu.cpp work between them (texture uploads, palettes, decodes) and are named after the piece that follows.
// Every PS2_GPU_PROFILE_N frames (default 50) the log gets the span of a frame on the card, the sum of the pieces, and the 30 dearest names
// in ms per frame; a frame whose span passes 20 ms gets a [gpu-slow] line with its 6 dearest pieces. The frame waits for its timestamps,
// so the frame rate drops a little while it is on; the pieces' own times stay true.
struct GpuProf {
    bool on = false;
    WGPUQuerySet set = nullptr;
    WGPUBuffer resolve = nullptr, read = nullptr;
    static constexpr u32 kMax = 1024;  // timestamps a frame (pairs: 512 pieces)
    u32 used = 0;                      // timestamps written this frame
    struct Piece {
        std::string name;
        u32 begin, end;  // timestamp indices
    };
    std::vector<Piece> pieces;
    struct Sum {
        double ms = 0;
        u64 n = 0;
    };
    std::unordered_map<std::string, Sum> sums;
    double span = 0, worst = 0, sum_pieces = 0, sum_gaps = 0;
    u32 frames = 0, every = 50, slow = 0;
    double period_ns = 1.0;
} g_gp;

void timer_init() {
    g_gp.on = os::env("PS2_GPU_PROFILE") == "1" || os::env("PS2_HWR_TIME") == "1";
    if (!g_gp.on) return;
    if (!wgpuDeviceHasFeature(C.device, WGPUFeatureName_TimestampQuery)) {
        std::fprintf(stderr, "[gpu-prof] the device has no timestamp queries: PS2_GPU_PROFILE is off\n");
        g_gp.on = false;
        return;
    }
    if (const long n = std::atol(os::env("PS2_GPU_PROFILE_N").c_str()); n > 0) g_gp.every = static_cast<u32>(n);
    WGPUQuerySetDescriptor qd = WGPU_QUERY_SET_DESCRIPTOR_INIT;
    qd.type = WGPUQueryType_Timestamp;
    qd.count = GpuProf::kMax;
    g_gp.set = wgpuDeviceCreateQuerySet(C.device, &qd);
    WGPUBufferDescriptor d = WGPU_BUFFER_DESCRIPTOR_INIT;
    d.size = GpuProf::kMax * 8;
    d.usage = WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc;
    g_gp.resolve = wgpuDeviceCreateBuffer(C.device, &d);
    d.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    g_gp.read = wgpuDeviceCreateBuffer(C.device, &d);
    g_gp.period_ns = wgpuQueueGetTimestampPeriod(C.queue);
    if (!(g_gp.period_ns > 0)) g_gp.period_ns = 1.0;
    if (!g_gp.set) g_gp.on = false;
    std::fprintf(stderr, "[gpu-prof] on: every render pass, compute pass and copy of the renderer is timed (%.3f ns per tick), report every %u frames\n",
                 g_gp.period_ns, g_gp.every);
}

// two timestamp slots for a piece named `name`; false when the frame's slots are used up
bool prof_take(std::string name, u32 &b, u32 &e) {
    if (!g_gp.on || g_gp.used + 2 > GpuProf::kMax) return false;
    b = g_gp.used++, e = g_gp.used++;
    g_gp.pieces.push_back({std::move(name), b, e});
    return true;
}

// an empty compute pass that writes its start (begin) or its end (end) timestamp: brackets a copy
void prof_mark(WGPUCommandEncoder enc, u32 index, bool at_end) {
    WGPUPassTimestampWrites tw = WGPU_PASS_TIMESTAMP_WRITES_INIT;
    tw.querySet = g_gp.set;
    if (at_end) tw.beginningOfPassWriteIndex = index;   // the copy's end: the next pass starts after it
    else tw.endOfPassWriteIndex = index;                // the copy's start: this pass ends before it
    WGPUComputePassDescriptor cd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
    cd.timestampWrites = &tw;
    WGPUComputePassEncoder cp = wgpuCommandEncoderBeginComputePass(enc, &cd);
    wgpuComputePassEncoderEnd(cp);
    wgpuComputePassEncoderRelease(cp);
}

std::string target_name(int ti) {
    if (ti < 0 || static_cast<std::size_t>(ti) >= g_targets.size() || !g_targets[static_cast<std::size_t>(ti)]) return "t?";
    const Target &t = *g_targets[static_cast<std::size_t>(ti)];
    char b[96];
    std::snprintf(b, sizeof b, "%s %04X/%u psm %02X %ux%u (%ux%u)", t.depth ? "Z" : "C", t.bp, t.bw, t.psm, t.w, t.h, card_x(t.w, t.sc), card_y(t.h, t.sc));
    return b;
}

void timer_resolve(WGPUCommandEncoder enc) {
    if (!g_gp.on || !g_gp.used) return;
    wgpuCommandEncoderResolveQuerySet(enc, g_gp.set, 0, g_gp.used, g_gp.resolve, 0);
    wgpuCommandEncoderCopyBufferToBuffer(enc, g_gp.resolve, 0, g_gp.read, 0, static_cast<u64>(g_gp.used) * 8);
}

void timer_read() {
    if (!g_gp.on) return;
    const prof::Scope prof_scope(prof::kRProfile);
    if (!g_gp.used) {
        g_gp.pieces.clear();
        return;
    }
    bool done = false;
    WGPUBufferMapCallbackInfo cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_AllowProcessEvents;
    cb.callback = [](WGPUMapAsyncStatus, WGPUStringView, void *flag, void *) { *static_cast<bool *>(flag) = true; };
    cb.userdata1 = &done;
    wgpuBufferMapAsync(g_gp.read, WGPUMapMode_Read, 0, static_cast<u64>(g_gp.used) * 8, cb);
    while (!done) wgpuDevicePoll(C.device, true, nullptr);
    const u64 *t = static_cast<const u64 *>(wgpuBufferGetConstMappedRange(g_gp.read, 0, static_cast<u64>(g_gp.used) * 8));
    const double ms_per_tick = g_gp.period_ns * 1e-6;
    const auto ms = [&](u64 a, u64 b) { return b > a ? static_cast<double>(b - a) * ms_per_tick : 0.0; };
    if (t && !g_gp.pieces.empty()) {
        std::vector<std::pair<double, const std::string *>> frame;
        u64 first = ~0ull, last = 0;
        for (std::size_t i = 0; i < g_gp.pieces.size(); ++i) {
            const auto &p = g_gp.pieces[i];
            const u64 a = t[p.begin], b = t[p.end];
            if (!a || !b) continue;
            first = std::min(first, a), last = std::max(last, b);
            const double d = ms(a, b);
            if (p.name == "compute") continue;  // a compute pass that ran no program (begun and ended around the renderer's work)
            GpuProf::Sum &s = g_gp.sums[p.name];
            s.ms += d, ++s.n;
            g_gp.sum_pieces += d;
            frame.push_back({d, &p.name});
            if (i + 1 < g_gp.pieces.size()) {  // what ran between this piece and the next: gpu.cpp's uploads, palettes and decodes
                const u64 nb = t[g_gp.pieces[i + 1].begin];
                const double gap = nb > b ? ms(b, nb) : 0.0;
                if (gap > 0.002) {
                    GpuProf::Sum &g = g_gp.sums["(card idle: waiting for the next submit) before " + g_gp.pieces[i + 1].name.substr(0, g_gp.pieces[i + 1].name.find(' '))];
                    g.ms += gap, ++g.n;
                    g_gp.sum_gaps += gap;
                }
            }
        }
        const double span = first < last ? ms(first, last) : 0.0;
        g_gp.span += span, g_gp.worst = std::max(g_gp.worst, span);
        if (span > 20.0 && g_gp.slow++ < 40) {
            std::sort(frame.begin(), frame.end(), [](const auto &x, const auto &y) { return x.first > y.first; });
            std::fprintf(stderr, "[gpu-slow] hwr frame %llu: %.2f ms on the card (%zu pieces); dearest:", static_cast<unsigned long long>(g_frames), span, frame.size());
            for (std::size_t k = 0; k < frame.size() && k < 6; ++k) std::fprintf(stderr, " | %.2f %s", frame[k].first, frame[k].second->c_str());
            std::fprintf(stderr, "\n");
        }
    }
    wgpuBufferUnmap(g_gp.read);
    g_gp.used = 0;
    g_gp.pieces.clear();
    if (++g_gp.frames >= g_gp.every) {
        const double f = static_cast<double>(g_gp.frames);
        std::vector<std::pair<std::string, GpuProf::Sum>> rows(g_gp.sums.begin(), g_gp.sums.end());
        std::sort(rows.begin(), rows.end(), [](const auto &x, const auto &y) { return x.second.ms > y.second.ms; });
        std::fprintf(stderr, "[gpu-prof] %u frames: card WORK %.2f ms a frame (the sum of the timed pieces below), idle between submits %.2f ms, "
                             "first to last timestamp %.2f ms (worst %.2f); frames whose work passed 20 ms: %u\n",
                     g_gp.frames, g_gp.sum_pieces / f, g_gp.sum_gaps / f, g_gp.span / f, g_gp.worst, g_gp.slow);
        for (std::size_t k = 0; k < rows.size() && k < 30; ++k)
            std::fprintf(stderr, "[gpu-prof]   %7.3f ms  %5.1f x  %s\n", rows[k].second.ms / f, static_cast<double>(rows[k].second.n) / f, rows[k].first.c_str());
        g_gp.sums.clear();
        g_gp.span = g_gp.worst = g_gp.sum_pieces = g_gp.sum_gaps = 0;
        g_gp.frames = 0, g_gp.slow = 0;
    }
}

WGPURenderPipeline draw_pipeline(u64 key, bool zuse, bool pass_depth) {
    PipeKey k = PipeKey::unpack(key & ~kKeyOn);
    k.depth = pass_depth;
    if (!zuse || !pass_depth) k.zcmp = WGPUCompareFunction_Always, k.zwrite = false;
    return pipeline(k);
}

void encode_pass(WGPUCommandEncoder enc, const Item &it) {
    if (!it.draw_n && !it.clear_c && !it.clear_d) return;
    static const bool no_pass = os::env("PS2_HWR_NOPASS") == "1";  // measuring aid: render passes are not encoded (the picture is wrong)
    if (no_pass || g_sweep == 2) return;
    const u64 cbytes = static_cast<u64>(card_x(it.cs->w, it.cs->sc)) * card_y(it.cs->h, it.cs->sc) * 4;
    WGPURenderPassColorAttachment ca = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
    ca.view = it.cs->view;
    ca.loadOp = it.clear_c ? WGPULoadOp_Clear : WGPULoadOp_Load;
    ca.storeOp = it.store_c ? WGPUStoreOp_Store : WGPUStoreOp_Discard;
    ca.clearValue = WGPUColor{it.cc[0], it.cc[1], it.cc[2], it.cc[3]};
    if (!it.clear_c) ++F.att_loads, F.att_load_bytes += cbytes;
    if (it.store_c) ++F.att_stores, F.att_store_bytes += cbytes;
    WGPURenderPassDepthStencilAttachment da = WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_INIT;
    WGPURenderPassDescriptor rd = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
    rd.colorAttachmentCount = 1;
    rd.colorAttachments = &ca;
    if (it.ds) {
        // depth that no draw reads and that is not kept needs no load (the F0 copy's dead depth writes)
        const bool load = !it.clear_d && (it.zread || it.store_d);
        da.view = it.ds->view;
        da.depthLoadOp = load ? WGPULoadOp_Load : WGPULoadOp_Clear;
        da.depthStoreOp = it.store_d ? WGPUStoreOp_Store : WGPUStoreOp_Discard;
        da.depthClearValue = it.clear_d ? it.cz : 0.0f;
        rd.depthStencilAttachment = &da;
        const u64 dbytes = static_cast<u64>(card_x(it.ds->w, it.ds->sc)) * card_y(it.ds->h, it.ds->sc) * 4;
        if (load) ++F.att_loads, F.att_load_bytes += dbytes;
        if (it.store_d) ++F.att_stores, F.att_store_bytes += dbytes;
    }
    WGPUPassTimestampWrites tw = WGPU_PASS_TIMESTAMP_WRITES_INIT;
    if (g_gp.on) {
        const Op &f = g_ops[static_cast<std::size_t>(g_pass_ops[it.first])];
        std::string name = "pass " + target_name(f.tgt) + (it.clear_c ? " clear" : " load") + (it.store_c ? "+store" : "+discard");
        if (it.ds) name += it.clear_d ? " +Z clear" : " +Z";
        u32 b, e;
        if (prof_take(std::move(name), b, e)) {
            tw.querySet = g_gp.set;
            tw.beginningOfPassWriteIndex = b;
            tw.endOfPassWriteIndex = e;
            rd.timestampWrites = &tw;
        }
    }
    WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(enc, &rd);
    ++F.passes;
    static const bool no_draw = os::env("PS2_HWR_NOPASS") == "draws";  // measuring aid: render passes without their draws
    if (it.draw_n && !no_draw && g_sweep != 1) {
        wgpuRenderPassEncoderSetVertexBuffer(rp, 0, g_vbuf, 0, vsource().size() * sizeof(Vtx));
        wgpuRenderPassEncoderSetBindGroup(rp, 0, g_param_group, 0, nullptr);
        WGPURenderPipeline cur = nullptr;
        WGPUBindGroup cur_group = nullptr;
        float cur_blend = -1.0f;
        u32 cur_sc[4] = {~0u, ~0u, ~0u, ~0u};
        const bool pass_depth = it.ds != nullptr;
        for (u32 i = it.draw_first; i < it.draw_first + it.draw_n; ++i) {
            const DrawRec &r = g_final[i];
            if (!r.sc[2] || !r.sc[3]) continue;
            if (r.group != cur_group) {
                wgpuRenderPassEncoderSetBindGroup(rp, 1, r.group, 0, nullptr);
                cur_group = r.group;
                ++F.groups;
            }
            if (!std::equal(r.sc, r.sc + 4, cur_sc)) {
                const Sc &tw = it.cs->sc;
                wgpuRenderPassEncoderSetScissorRect(rp, card_at_x(r.sc[0], tw), card_at_y(r.sc[1], tw), card_at_x(r.sc[0] + r.sc[2], tw) - card_at_x(r.sc[0], tw),
                                                    card_at_y(r.sc[1] + r.sc[3], tw) - card_at_y(r.sc[1], tw));
                std::copy(r.sc, r.sc + 4, cur_sc);
            }
            if (r.blend != cur_blend) {
                const WGPUColor bc{r.blend, r.blend, r.blend, r.blend};
                wgpuRenderPassEncoderSetBlendConstant(rp, &bc);
                cur_blend = r.blend;
            }
            for (const u64 key : {r.k1, r.k2}) {
                if (!key) continue;
                WGPURenderPipeline p = draw_pipeline(key, r.zuse, pass_depth);
                if (p != cur) {
                    wgpuRenderPassEncoderSetPipeline(rp, p);
                    cur = p;
                    ++F.pipelines;
                }
                wgpuRenderPassEncoderDraw(rp, r.count, 1, r.first, 0);
                ++F.draws;
            }
        }
    }
    wgpuRenderPassEncoderEnd(rp);
    wgpuRenderPassEncoderRelease(rp);
}

void encode_op(WGPUCommandEncoder enc, const Op &op) {
    // PS2_GPU_SWEEP parts 5, 6, 7: the target <-> memory transfers, the copies of targets and the copies of decoded texture levels are left out
    if ((g_sweep == 5 && (op.kind == kOpLoad || op.kind == kOpStore || op.kind == kOpStoreZ)) || (g_sweep == 6 && op.kind == kOpSelfCopy) ||
        (g_sweep == 7 && op.kind == kOpCopyLevel))
        return;
    switch (op.kind) {
    case kOpPass:
        return;
    case kOpCopyLevel: {
        WGPUTexelCopyBufferInfo src = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
        src.buffer = C.decoded;
        src.layout.offset = static_cast<u64>(op.offset) * 4;
        src.layout.bytesPerRow = op.pitch * 4;
        src.layout.rowsPerImage = op.h;
        WGPUTexelCopyTextureInfo dst = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
        dst.texture = op.tex;
        dst.mipLevel = op.level;
        const WGPUExtent3D size{op.w, op.h, 1};
        wgpuCommandEncoderCopyBufferToTexture(enc, &src, &dst, &size);
        return;
    }
    case kOpSelfCopy: {
        WGPUTexelCopyTextureInfo src = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
        src.texture = op.tex;
        WGPUTexelCopyTextureInfo dst = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
        dst.texture = op.tex2;
        if (op.level == 1) {  // a part (self_copy): card origin and size
            src.origin = {op.offset, op.pitch, 0};
            dst.origin = {op.offset, op.pitch, 0};
            const WGPUExtent3D part{op.w, op.h, 1};
            wgpuCommandEncoderCopyTextureToTexture(enc, &src, &dst, &part);
            return;
        }
        const WGPUExtent3D size{card_x(op.w, op.sc), card_y(op.h, op.sc), 1};
        wgpuCommandEncoderCopyTextureToTexture(enc, &src, &dst, &size);
        return;
    }
    case kOpLoad: case kOpStore: case kOpStoreZ: {
        WGPUComputePassEncoder cp = wgpuCommandEncoderBeginComputePass(enc, nullptr);
        wgpuComputePassEncoderSetPipeline(cp, op.kind == kOpLoad ? g_load_pipe : op.kind == kOpStore ? g_store_pipe : g_store_z_pipe);
        const u32 offset = op.param * C.param_stride;
        wgpuComputePassEncoderSetBindGroup(cp, 0, *C.group, 1, &offset);
        wgpuComputePassEncoderSetBindGroup(cp, 1, op.group, 0, nullptr);
        // stores: one thread per memory sample; loads: one per target texel (the rectangle's texels, +1 for the rounding at its edges)
        const u32 tw = op.kind == kOpLoad ? card_x(op.w, op.sc) + 1 : op.w * g_s, th = op.kind == kOpLoad ? card_y(op.h, op.sc) + 1 : op.h * g_s;
        wgpuComputePassEncoderDispatchWorkgroups(cp, (tw + 7) / 8, (th + 7) / 8, 1);
        wgpuComputePassEncoderEnd(cp);
        wgpuComputePassEncoderRelease(cp);
        if (op.kind == kOpLoad) {  // the whole staging picture (every texel of it is inside the target) into the surface
            WGPUTexelCopyTextureInfo src = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
            src.texture = op.tex;
            WGPUTexelCopyTextureInfo dst = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
            dst.texture = op.tex2;
            const WGPUExtent3D size{card_x(op.w, op.sc), card_y(op.h, op.sc), 1};
            wgpuCommandEncoderCopyTextureToTexture(enc, &src, &dst, &size);
        }
        return;
    }
    case kOpShow: {  // the displayed rectangle of a target (card texels at offset, pitch) into gpu.cpp's field texture at 0, 0
        WGPUTexelCopyTextureInfo src = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
        src.texture = op.tex;
        src.origin = {op.offset, op.pitch, 0};
        WGPUTexelCopyTextureInfo dst = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
        dst.texture = op.tex2;
        const WGPUExtent3D size{op.w, op.h, 1};
        wgpuCommandEncoderCopyTextureToTexture(enc, &src, &dst, &size);
        return;
    }
    }
}

}  // namespace

int prof_compute(WGPUPassTimestampWrites &tw) {
    if (!g_gp.on) return -1;
    u32 b, e;
    if (!prof_take("compute", b, e)) return -1;
    tw = WGPU_PASS_TIMESTAMP_WRITES_INIT;
    tw.querySet = g_gp.set;
    tw.beginningOfPassWriteIndex = b;
    tw.endOfPassWriteIndex = e;
    return static_cast<int>(g_gp.pieces.size() - 1);
}

void prof_name(int piece, const char *program) {
    if (piece < 0 || static_cast<std::size_t>(piece) >= g_gp.pieces.size()) return;
    std::string &n = g_gp.pieces[static_cast<std::size_t>(piece)].name;
    const std::string p = std::string(" ") + program;
    if (n.find(p) == std::string::npos) n += p;   // "compute decode palette ...": the programs the pass runs, once each
}

WGPUComputePassEncoder encode(WGPUCommandEncoder enc, u32 upto, WGPUComputePassEncoder pass) {
    const prof::Scope encode_scope(prof::kREncode);
    if (pass) {
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    }
    while (g_next_item < g_items.size() && g_items[g_next_item].pos <= upto) {
        const Item &it = g_items[g_next_item++];
        if (it.op >= 0) {
            const Op &op = g_ops[static_cast<std::size_t>(it.op)];
            u32 b = 0, e = 0;
            bool timed = false;
            if (g_gp.on && op.kind != kOpPass) {
                static const char *const kKind[] = {"pass", "load", "store", "storeZ", "self copy", "level copy", "show"};
                const char *kind = "op";
                switch (op.kind) {
                case kOpLoad: kind = kKind[1]; break;
                case kOpStore: kind = kKind[2]; break;
                case kOpStoreZ: kind = kKind[3]; break;
                case kOpSelfCopy: kind = kKind[4]; break;
                case kOpCopyLevel: kind = kKind[5]; break;
                case kOpShow: kind = kKind[6]; break;
                default: break;
                }
                char extra[48] = "";
                if (op.kind == kOpCopyLevel) std::snprintf(extra, sizeof extra, " %ux%u", op.w, op.h);
                timed = prof_take(std::string(kind) + " " + (op.kind == kOpCopyLevel ? std::string("texture") : target_name(op.tgt)) + extra, b, e);
                if (timed) prof_mark(enc, b, false);
            }
            encode_op(enc, op);
            if (timed) prof_mark(enc, e, true);
        } else {
            encode_pass(enc, it);
        }
    }
    if (upto == ~0u) timer_resolve(enc);
    return nullptr;
}

struct Handoff {
    std::vector<WGPUBindGroup> groups;
    std::vector<WGPUTextureView> views;
    std::vector<WGPUTexture> texs;
    std::vector<Vtx> verts;                    // handoff(true): this submit's vertices and parameter records, written by the job
    std::vector<std::array<u32, 16>> params;
    WGPUBuffer vbuf = nullptr, pbuf = nullptr;  // (referenced for the job: the buffers may grow and be replaced before it runs)
};
namespace {
std::mutex g_handoff_mutex;
std::vector<Handoff *> g_handoff_pool;  // finished hand-overs, their vectors keep their room (a frame's vertices are megabytes)
}
Handoff *handoff(bool take_writes) {
    Handoff *h = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_handoff_mutex);
        if (!g_handoff_pool.empty()) h = g_handoff_pool.back(), g_handoff_pool.pop_back();
    }
    if (!h) h = new Handoff;
    h->groups.swap(g_release_group);
    h->views.swap(g_release_view);
    h->texs.swap(g_release_tex);
    if (take_writes) {  // (taken after encode: nothing of this submit reads them from hwr any more; submitted() clears what is left)
        std::vector<Vtx> &source = g_reorder_verts ? g_vorder : g_verts;
        h->verts.swap(source);
        h->params.swap(g_params);
        h->vbuf = g_vbuf;
        h->pbuf = g_pbuf;
        if (h->vbuf) wgpuBufferAddRef(h->vbuf);
        if (h->pbuf) wgpuBufferAddRef(h->pbuf);
    }
    return h;
}
void handoff_write(Handoff *h) {
    if (!h->verts.empty() && h->vbuf) wgpuQueueWriteBuffer(C.queue, h->vbuf, 0, h->verts.data(), h->verts.size() * sizeof(Vtx));
    if (!h->params.empty() && h->pbuf) wgpuQueueWriteBuffer(C.queue, h->pbuf, 0, h->params.data(), h->params.size() * 64);
}
void handoff_release(Handoff *h) {
    for (WGPUBindGroup g : h->groups) wgpuBindGroupRelease(g);
    for (WGPUTextureView v : h->views) wgpuTextureViewRelease(v);
    for (WGPUTexture t : h->texs) wgpuTextureDestroy(t), wgpuTextureRelease(t);
    h->groups.clear();
    h->views.clear();
    h->texs.clear();
    if (h->vbuf) wgpuBufferRelease(h->vbuf);
    if (h->pbuf) wgpuBufferRelease(h->pbuf);
    h->vbuf = h->pbuf = nullptr;
    h->verts.clear();
    h->params.clear();
    std::lock_guard<std::mutex> lock(g_handoff_mutex);
    g_handoff_pool.push_back(h);
}

void submitted() {
    g_ops.clear();
    g_draws.clear();
    g_verts.clear();
    g_params.clear();
    g_items.clear();
    g_pass_ops.clear();
    g_final.clear();
    g_vorder.clear();
    g_next_item = 0;
    g_open_pass = -1;
    g_clear.on = false;
    ++g_submit;
    g_dec_used = 0;
    for (WGPUBindGroup g : g_release_group) wgpuBindGroupRelease(g);
    for (WGPUTextureView v : g_release_view) wgpuTextureViewRelease(v);
    for (WGPUTexture t : g_release_tex) wgpuTextureDestroy(t), wgpuTextureRelease(t);
    g_release_group.clear();
    g_release_view.clear();
    g_release_tex.clear();
    for (Surf *s : g_retired) g_pool.push_back(s);
    g_retired.clear();
    for (int i : g_dead_slots)
        if (g_targets[static_cast<std::size_t>(i)] && !g_targets[static_cast<std::size_t>(i)]->alive) g_targets[static_cast<std::size_t>(i)].reset();
}

void frame_end() {
    if (!g_on) return;
    ++g_frames;
#if defined(PS2_DRAW_SAMPLER)
    if (g_frames == 100) {  // after the start (pipelines, translations)
        const std::string path = os::env("PS2_DRAW_SAMPLE");
        if (!path.empty()) drawtime::start_sampler(path.c_str());
    }
#endif
    if (drawtime::g_on) {  // PS2_DRAW_TIME: the drawing thread's GS stream time and its processor time per picture (drawtime.hpp)
        // Per picture into a bucket (1: the A/B switch had the shortcuts off for it), summed from picture 100 on (the first ones set up and
        // compile); printed every 100 pictures. With PS2_DRAW_TIME=ab the switch flips every PS2_DRAW_TIME_EVERY pictures (10).
        static const u64 every = [] {
            const long n = std::strtol(os::env("PS2_DRAW_TIME_EVERY").c_str(), nullptr, 10);
            return static_cast<u64>(n > 0 ? n : 10);
        }();
        static u64 frames = 0, tsc0 = drawtime::now();
        static auto clock0 = std::chrono::steady_clock::now();
        static double thread_last = -1.0, thread_sum[2] = {}, sum[2][drawtime::kParts] = {};
        static u64 count[2][drawtime::kParts] = {}, pictures[2] = {};
        const int b = g_ab_off ? 1 : 0;
        const double thread_now = drawtime::thread_ms(1.0);  // Windows: cycles (converted below), else ms
        if (thread_last >= 0.0 && frames >= 100) {
            thread_sum[b] += thread_now - thread_last;
            for (int k = 0; k < drawtime::kParts; ++k) sum[b][k] += static_cast<double>(drawtime::g_sum[k]), count[b][k] += drawtime::g_count[k];
            ++pictures[b];
        }
        thread_last = thread_now;
        for (int k = 0; k < drawtime::kParts; ++k) drawtime::g_sum[k] = drawtime::g_count[k] = 0;
        ++frames;
        if (g_ab_on && frames % every == 0) g_ab_off = !g_ab_off;
        if (frames % 100 == 0 && frames > 100) {
            const double ms_per_tick = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - clock0).count() / static_cast<double>(drawtime::now() - tsc0);
#if defined(_WIN32)
            const double thread_scale = ms_per_tick;
#else
            const double thread_scale = 1.0;
#endif
            for (int k = 0; k < 2; ++k) {
                if (!pictures[k]) continue;
                const double n = static_cast<double>(pictures[k]);
                std::fprintf(stderr, "[draw-time] pictures 100-%llu%s (%llu): drawing thread processor %.3f ms per picture | gif %.3f ms (%.0f messages), shape %.3f ms (%.0f calls), build %.3f ms (%.0f), PACKED runs %.3f ms (%.0f)\n",
                             static_cast<unsigned long long>(frames), g_ab_on ? (k ? (g_ab_memo_only ? ", memo OFF" : ", shortcuts OFF") : ", shortcuts on") : "",
                             static_cast<unsigned long long>(pictures[k]), thread_sum[k] * thread_scale / n, sum[k][drawtime::kGif] * ms_per_tick / n,
                             static_cast<double>(count[k][drawtime::kGif]) / n, sum[k][drawtime::kShape] * ms_per_tick / n, static_cast<double>(count[k][drawtime::kShape]) / n,
                             sum[k][drawtime::kBuild] * ms_per_tick / n, static_cast<double>(count[k][drawtime::kBuild]) / n,
                             sum[k][drawtime::kPacked] * ms_per_tick / n, static_cast<double>(count[k][drawtime::kPacked]) / n);
            }
            std::fprintf(stderr, "[draw-time] last 100 pictures, per picture: runs closed by register writes %.0f, single PACKED writes %.0f, image data %.0f, other messages %.0f, other PACKED tags %.0f; "
                                 "primitives one by one: two-pass alpha test %.0f, own target read %.0f, view 4 %.0f, other kinds %.0f\n",
                         drawtime::g_close[0] / 100.0, drawtime::g_close[1] / 100.0, drawtime::g_close[2] / 100.0, drawtime::g_close[3] / 100.0, drawtime::g_close[4] / 100.0,
                         drawtime::g_slow[0] / 100.0, drawtime::g_slow[1] / 100.0, drawtime::g_slow[2] / 100.0, drawtime::g_slow[3] / 100.0);
            for (auto &c : drawtime::g_close) c = 0;
            for (auto &c : drawtime::g_slow) c = 0;
        }
    }
    timer_read();
    static const long dump_frame = os::env("PS2_HWR_DUMP").empty() ? -1 : std::strtol(os::env("PS2_HWR_DUMP").c_str(), nullptr, 10);
    g_dump_now = dump_frame >= 0 && static_cast<long>(g_frames) == dump_frame;
    cache_evict();
    if (g_log) {
        std::fprintf(stderr, "[hwr] frame %llu: shapes %llu drawn %llu skipped %llu clears %llu renames %llu | nodes %llu passes %llu draws %llu pipeline switches %llu group switches %llu vertices %llu | "
                             "attachment loads %llu (%.2f MB) stores %llu (%.2f MB) | tex lookups %llu hits %llu misses %llu (%llu texels) target reads %llu self copies %llu (%.2f MB) | loads %llu stores %llu new targets %llu new surfaces %llu | cpu %.2f ms schedule %.2f ms | cache %zu entries %.1f MB",
                     static_cast<unsigned long long>(g_frames), static_cast<unsigned long long>(F.shapes), static_cast<unsigned long long>(F.drawn), static_cast<unsigned long long>(F.skipped),
                     static_cast<unsigned long long>(F.clears), static_cast<unsigned long long>(F.renames), static_cast<unsigned long long>(F.nodes), static_cast<unsigned long long>(F.passes),
                     static_cast<unsigned long long>(F.draws), static_cast<unsigned long long>(F.pipelines), static_cast<unsigned long long>(F.groups), static_cast<unsigned long long>(F.vertices),
                     static_cast<unsigned long long>(F.att_loads), static_cast<double>(F.att_load_bytes) / 1048576.0, static_cast<unsigned long long>(F.att_stores),
                     static_cast<double>(F.att_store_bytes) / 1048576.0, static_cast<unsigned long long>(F.lookups), static_cast<unsigned long long>(F.hits),
                     static_cast<unsigned long long>(F.misses), static_cast<unsigned long long>(F.decoded_texels), static_cast<unsigned long long>(F.target_reads),
                     static_cast<unsigned long long>(F.self_copies), static_cast<double>(F.copy_bytes) / 1048576.0, static_cast<unsigned long long>(F.loads), static_cast<unsigned long long>(F.stores),
                     static_cast<unsigned long long>(F.targets_made), static_cast<unsigned long long>(F.surfaces_made), F.cpu_ms, F.sched_ms, g_cache.size(),
                     static_cast<double>(g_cache_bytes) / 1048576.0);
        for (int r = 0; r < kReasons; ++r)
            if (F.fallback[r]) std::fprintf(stderr, " | FALLBACK %s %llu", kReasonName[r], static_cast<unsigned long long>(F.fallback[r]));
        for (int r = 0; r < kApproxes; ++r)
            if (F.approx[r]) std::fprintf(stderr, " | approx %s %llu", kApproxName[r], static_cast<unsigned long long>(F.approx[r]));
        if (F.view4_out) std::fprintf(stderr, " | view 4 outside its target %llu", static_cast<unsigned long long>(F.view4_out));
        if (F.mipdraws)
            std::fprintf(stderr, " | mipmapped draws %llu needing 1/2/3/4+ levels: %llu/%llu/%llu/%llu", static_cast<unsigned long long>(F.mipdraws), static_cast<unsigned long long>(F.mipneed[1]),
                         static_cast<unsigned long long>(F.mipneed[2]), static_cast<unsigned long long>(F.mipneed[3]),
                         static_cast<unsigned long long>(F.mipneed[4] + F.mipneed[5] + F.mipneed[6] + F.mipneed[7]));
        if (g_mem_read_hits) std::fprintf(stderr, " | uploads into pages a draw reads from memory %llu", static_cast<unsigned long long>(g_mem_read_hits));
        if (g_mem_read_changes) std::fprintf(stderr, " | of them with other content %llu", static_cast<unsigned long long>(g_mem_read_changes));
        g_mem_read_hits = g_mem_read_changes = 0;
        std::fprintf(stderr, " | states built %llu (%.2f ms, %llu for the epoch, %llu from the memo)", static_cast<unsigned long long>(F.builds), F.build_ms,
                     static_cast<unsigned long long>(F.builds_epoch), static_cast<unsigned long long>(F.memo_hits));
        std::fprintf(stderr, "\n");
    }
    T.shapes += F.shapes, T.drawn += F.drawn, T.skipped += F.skipped, T.clears += F.clears, T.draws += F.draws, T.passes += F.passes, T.pipelines += F.pipelines;
    T.groups += F.groups, T.vertices += F.vertices, T.lookups += F.lookups, T.hits += F.hits, T.misses += F.misses, T.decoded_texels += F.decoded_texels;
    T.target_reads += F.target_reads, T.self_copies += F.self_copies, T.loads += F.loads, T.stores += F.stores, T.targets_made += F.targets_made, T.cpu_ms += F.cpu_ms;
    T.nodes += F.nodes, T.renames += F.renames, T.surfaces_made += F.surfaces_made, T.copy_bytes += F.copy_bytes, T.sched_ms += F.sched_ms;
    T.att_loads += F.att_loads, T.att_stores += F.att_stores, T.att_load_bytes += F.att_load_bytes, T.att_store_bytes += F.att_store_bytes;
    for (int r = 0; r < kReasons; ++r) T.fallback[r] += F.fallback[r];
    for (int r = 0; r < kApproxes; ++r) T.approx[r] += F.approx[r];
    F = Counters{};
}

void stats() {
    if (!g_on || !g_frames) return;
    const double n = static_cast<double>(g_frames);
    u64 fb = 0;
    for (int r = 0; r < kReasons; ++r) fb += T.fallback[r];
    std::fprintf(stderr, "[hwr] %llu frames, per frame: shapes %.0f drawn %.0f, clears %.1f renames %.1f, nodes %.1f, render passes %.1f, draws %.1f, pipeline switches %.1f, group switches %.1f, vertices %.0f, "
                         "attachment loads %.1f (%.2f MB) stores %.1f (%.2f MB), self copies %.1f (%.2f MB), stores to memory %.1f, processor %.3f ms (schedule %.3f ms); texture lookups %.1f: cache hits %.1f, "
                         "misses %.2f (hit rate %.1f %% of cache lookups), target reads %.1f; fallback shapes %llu; %zu pipelines, %zu surfaces\n",
                 static_cast<unsigned long long>(g_frames), static_cast<double>(T.shapes) / n, static_cast<double>(T.drawn) / n, static_cast<double>(T.clears) / n,
                 static_cast<double>(T.renames) / n, static_cast<double>(T.nodes) / n, static_cast<double>(T.passes) / n, static_cast<double>(T.draws) / n,
                 static_cast<double>(T.pipelines) / n, static_cast<double>(T.groups) / n, static_cast<double>(T.vertices) / n, static_cast<double>(T.att_loads) / n,
                 static_cast<double>(T.att_load_bytes) / n / 1048576.0, static_cast<double>(T.att_stores) / n, static_cast<double>(T.att_store_bytes) / n / 1048576.0,
                 static_cast<double>(T.self_copies) / n, static_cast<double>(T.copy_bytes) / n / 1048576.0, static_cast<double>(T.stores) / n, T.cpu_ms / n, T.sched_ms / n,
                 static_cast<double>(T.lookups) / n, static_cast<double>(T.hits) / n, static_cast<double>(T.misses) / n,
                 T.hits + T.misses ? 100.0 * static_cast<double>(T.hits) / static_cast<double>(T.hits + T.misses) : 0.0, static_cast<double>(T.target_reads) / n,
                 static_cast<unsigned long long>(fb), g_pipes.size(), g_surfs.size());
    for (int r = 0; r < kReasons; ++r)
        if (T.fallback[r]) std::fprintf(stderr, "[hwr]   fallback %s: %llu shapes\n", kReasonName[r], static_cast<unsigned long long>(T.fallback[r]));
    for (int r = 0; r < kApproxes; ++r)
        if (T.approx[r]) std::fprintf(stderr, "[hwr]   approximation %s: %llu\n", kApproxName[r], static_cast<unsigned long long>(T.approx[r]));
}

}  // namespace ps2::gs::gpu::hwr

#endif

#ifdef __EMSCRIPTEN__
// the browser build has no hardware renderer: PS2_RENDERER is ignored there
namespace ps2::gs::gpu::hwr {
bool g_on = false;
int g_sweep = 0;
bool requested() { return false; }
bool init(const Context &) { return false; }
void shape(const State &, const std::uint32_t *, const std::uint32_t *, std::uint32_t, std::uint64_t, const Vertex *, int) {}
void shapes(const State &, const std::uint32_t *, const std::uint32_t *, std::uint32_t, std::uint64_t, const Vertex *, int, const std::uint16_t *, int, int) {}
bool batch_on() { return false; }
void upload(const std::uint32_t *, const std::uint8_t *, std::size_t) {}
void clut(std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, bool, std::uint32_t, std::uint32_t, std::uint32_t) {}
void pre_blocks(bool, const std::uint32_t *, std::vector<std::uint32_t> &) {}
void local_copy(std::uint32_t, std::uint32_t, std::uint32_t, int, int, int, int) {}
void before_display(std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t) {}
void sync_all() {}
void state_loaded() {}
bool pending() { return false; }
bool before(std::uint32_t) { return false; }
void prepare(bool) {}
WGPUComputePassEncoder encode(WGPUCommandEncoder, std::uint32_t, WGPUComputePassEncoder pass) { return pass; }
struct Handoff {};
Handoff *handoff(bool) { return nullptr; }
void handoff_write(Handoff *) {}
void handoff_release(Handoff *) {}
void submitted() {}
void frame_end() {}
int prof_compute(WGPUPassTimestampWrites &) { return -1; }
void prof_name(int, const char *) {}
void stats() {}
}
#endif
