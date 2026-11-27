#pragma once

#include "cpu.hpp"

#include <cstdint>
#include <cstring>

namespace ps2 {

inline constexpr std::uint32_t kRamSize = 32u << 20;
inline constexpr std::uint32_t kScratchpadBase = 0x70000000u;
inline constexpr std::uint32_t kScratchpadSize = 16u << 10;

#ifdef PS2_LIVE_EE
// A library of main processor code compiled while the game runs (ee_live_abi.hpp) reaches the memory through a pointer it is handed
// when it is loaded: it is not linked to the runner's g_ram.
extern std::uint8_t *g_live_ram;
extern std::uint8_t *g_live_spr;
#define PS2_RAM g_live_ram
#define PS2_SPR g_live_spr
#else
extern std::uint8_t g_ram[];
extern std::uint8_t g_spr[];
#define PS2_RAM g_ram
#define PS2_SPR g_spr
#endif

extern bool g_watch_on;
extern std::uint32_t g_watch_base, g_watch_end;
void watch_hit(std::uint32_t a, std::uint32_t v, const char *via);
void watch_span(std::uint32_t a, std::uint32_t n, const char *via);
void init_watch();

template <typename T>
inline bool watched(std::uint32_t a, std::uint32_t v) {
    if (g_watch_on && a >= g_watch_base && a < g_watch_end) [[unlikely]] watch_hit(a, v, "store");
    return false;
}

std::uint8_t slow_ld8(std::uint32_t a);
std::uint16_t slow_ld16(std::uint32_t a);
std::uint32_t slow_ld32(std::uint32_t a);
std::uint64_t slow_ld64(std::uint32_t a);
void slow_ld128(std::uint32_t a, Reg128 &out);
void slow_st8(std::uint32_t a, std::uint8_t v);
void slow_st16(std::uint32_t a, std::uint16_t v);
void slow_st32(std::uint32_t a, std::uint32_t v);
void slow_st64(std::uint32_t a, std::uint64_t v);
void slow_st128(std::uint32_t a, const Reg128 &v);

template <typename T>
inline T *fast(std::uint32_t a) {
    if (a < kRamSize) [[likely]] return reinterpret_cast<T *>(PS2_RAM + a);
    if ((a >> 14) == (kScratchpadBase >> 14)) return reinterpret_cast<T *>(PS2_SPR + (a & (kScratchpadSize - 1)));
    return nullptr;
}

template <typename T>
inline T load(const void *p) {
    T v;
    std::memcpy(&v, p, sizeof v);
    return v;
}

template <typename T>
inline void store(void *p, T v) { std::memcpy(p, &v, sizeof v); }

inline std::uint8_t ld8(std::uint32_t a) {
    if (auto *p = fast<std::uint8_t>(a)) return *p;
    return slow_ld8(a);
}

inline std::uint16_t ld16(std::uint32_t a) {
    if (auto *p = fast<std::uint8_t>(a)) return load<std::uint16_t>(p);
    return slow_ld16(a);
}

inline std::uint32_t ld32(std::uint32_t a) {
    if (auto *p = fast<std::uint8_t>(a)) return load<std::uint32_t>(p);
    return slow_ld32(a);
}

inline std::uint64_t ld64(std::uint32_t a) {
    if (auto *p = fast<std::uint8_t>(a)) return load<std::uint64_t>(p);
    return slow_ld64(a);
}

inline void ld128(std::uint32_t a, Reg128 &out) {
    if (auto *p = fast<std::uint8_t>(a)) { std::memcpy(&out, p, 16); return; }
    slow_ld128(a, out);
}

#ifdef PS2_LIVE_EE
// A library of code compiled while the game runs keeps the plain store path: its watch switch is a copy (ee_live_abi.hpp).
inline void st8(std::uint32_t a, std::uint8_t v) {
    watched<std::uint8_t>(a, v);
    if (auto *p = fast<std::uint8_t>(a)) { *p = v; return; }
    slow_st8(a, v);
}

inline void st16(std::uint32_t a, std::uint16_t v) {
    watched<std::uint16_t>(a, v);
    if (auto *p = fast<std::uint8_t>(a)) { store(p, v); return; }
    slow_st16(a, v);
}

inline void st32(std::uint32_t a, std::uint32_t v) {
    watched<std::uint32_t>(a, v);
    if (auto *p = fast<std::uint8_t>(a)) { store(p, v); return; }
    slow_st32(a, v);
}

inline void st64(std::uint32_t a, std::uint64_t v) {
    watched<std::uint32_t>(a, static_cast<std::uint32_t>(v));
    if (auto *p = fast<std::uint8_t>(a)) { store(p, v); return; }
    slow_st64(a, v);
}

inline void st128(std::uint32_t a, const Reg128 &v) {
    watched<std::uint32_t>(a, v.w[0]);
    if (auto *p = fast<std::uint8_t>(a)) { std::memcpy(p, &v, 16); return; }
    slow_st128(a, v);
}
#else
// A guest store is one compare and one write when it goes to main memory: g_store_ram_end is the end of main memory, or 0
// while the write watch (PS2_WATCH) is on. Every other store (scratchpad, registers, mirrors, and every store while the
// watch is on) takes st*_other in memory.cpp, which is the old path: the watch check, then scratchpad or main memory, then
// slow_st*. The order of what happens is the same as before; only the test for the common case is shorter. It is an inline
// variable so the generated code reaches it directly (an extern variable costs a load of its address on mingw).
inline std::uint32_t g_store_ram_end = kRamSize;

void st8_other(std::uint32_t a, std::uint8_t v);
void st16_other(std::uint32_t a, std::uint16_t v);
void st32_other(std::uint32_t a, std::uint32_t v);
void st64_other(std::uint32_t a, std::uint64_t v);
void st128_other(std::uint32_t a, const Reg128 &v);

inline void st8(std::uint32_t a, std::uint8_t v) {
    if (a < g_store_ram_end) [[likely]] { PS2_RAM[a] = v; return; }
    st8_other(a, v);
}

inline void st16(std::uint32_t a, std::uint16_t v) {
    if (a < g_store_ram_end) [[likely]] { store(PS2_RAM + a, v); return; }
    st16_other(a, v);
}

inline void st32(std::uint32_t a, std::uint32_t v) {
    if (a < g_store_ram_end) [[likely]] { store(PS2_RAM + a, v); return; }
    st32_other(a, v);
}

inline void st64(std::uint32_t a, std::uint64_t v) {
    if (a < g_store_ram_end) [[likely]] { store(PS2_RAM + a, v); return; }
    st64_other(a, v);
}

inline void st128(std::uint32_t a, const Reg128 &v) {
    if (a < g_store_ram_end) [[likely]] { std::memcpy(PS2_RAM + a, &v, 16); return; }
    st128_other(a, v);
}
#endif

inline std::uint64_t mask_below(std::uint32_t bits) { return bits >= 64 ? ~0ull : (1ull << bits) - 1; }

inline void lwl(Reg128 &rt, std::uint32_t a) {
    const std::uint32_t shift = (3u - (a & 3u)) * 8u;
    const std::uint32_t mem = ld32(a & ~3u);
    const std::uint32_t v = (mem << shift) | (rt.w[0] & static_cast<std::uint32_t>(mask_below(shift)));
    rt.sd[0] = static_cast<std::int32_t>(v);
}

inline void lwr(Reg128 &rt, std::uint32_t a) {
    const std::uint32_t shift = (a & 3u) * 8u;
    const std::uint32_t mem = ld32(a & ~3u);
    const std::uint32_t v = (mem >> shift) | (rt.w[0] & ~static_cast<std::uint32_t>(mask_below(32 - shift)));
    if (shift == 0) rt.sd[0] = static_cast<std::int32_t>(v);
    else rt.w[0] = v;
}

inline void ldl(Reg128 &rt, std::uint32_t a) {
    const std::uint32_t shift = (7u - (a & 7u)) * 8u;
    const std::uint64_t mem = ld64(a & ~7u);
    rt.d[0] = (mem << shift) | (rt.d[0] & mask_below(shift));
}

inline void ldr(Reg128 &rt, std::uint32_t a) {
    const std::uint32_t shift = (a & 7u) * 8u;
    const std::uint64_t mem = ld64(a & ~7u);
    rt.d[0] = (mem >> shift) | (rt.d[0] & ~mask_below(64 - shift));
}

inline void swl(const Reg128 &rt, std::uint32_t a) {
    const std::uint32_t shift = (3u - (a & 3u)) * 8u;
    const std::uint32_t mem = ld32(a & ~3u);
    st32(a & ~3u, (rt.w[0] >> shift) | (mem & ~static_cast<std::uint32_t>(mask_below(32 - shift))));
}

inline void swr(const Reg128 &rt, std::uint32_t a) {
    const std::uint32_t shift = (a & 3u) * 8u;
    const std::uint32_t mem = ld32(a & ~3u);
    st32(a & ~3u, (rt.w[0] << shift) | (mem & static_cast<std::uint32_t>(mask_below(shift))));
}

inline void sdl(const Reg128 &rt, std::uint32_t a) {
    const std::uint32_t shift = (7u - (a & 7u)) * 8u;
    const std::uint64_t mem = ld64(a & ~7u);
    st64(a & ~7u, (rt.d[0] >> shift) | (mem & ~mask_below(64 - shift)));
}

inline void sdr(const Reg128 &rt, std::uint32_t a) {
    const std::uint32_t shift = (a & 7u) * 8u;
    const std::uint64_t mem = ld64(a & ~7u);
    st64(a & ~7u, (rt.d[0] << shift) | (mem & mask_below(shift)));
}

// The generated code's own memory accesses (recomp/codegen.cpp). A unit function takes the processor state as
// `Cpu &PS2_GEN_RESTRICT c`: nothing but c reaches the state while the unit runs, so the compiler may keep guest registers and
// the budget in host registers across guest loads and stores instead of writing them back after each one. Only the slow paths
// can see the state behind the compiler's back (the INTC_STAT wait skip in hw.cpp compares g_cpu.r and moves g_cpu.budget, a timer
// read takes the time from g_cpu.budget, request_iop_sync lowers the budget), so they are fenced: every cached value is written back
// before the slow path and read again after it, and the game computes exactly what it did. Calls that take c are fences by themselves.
// -DPS2_NO_GEN_RESTRICT (PS2_LIVE_FLAGS for a live library) leaves the state in memory as before, for comparisons.
#if (defined(__clang__) || defined(__GNUC__)) && !defined(__EMSCRIPTEN__) && !defined(PS2_NO_GEN_RESTRICT)
#define PS2_GEN_RESTRICT __restrict__
inline void fence(Cpu &c) { asm volatile("" : : "r"(&c) : "memory"); }
#else
#define PS2_GEN_RESTRICT
inline void fence(Cpu &) {}
#endif

inline std::uint8_t ld8(Cpu &c, std::uint32_t a) {
    if (auto *p = fast<std::uint8_t>(a)) return *p;
    fence(c);
    const std::uint8_t v = slow_ld8(a);
    fence(c);
    return v;
}

inline std::uint16_t ld16(Cpu &c, std::uint32_t a) {
    if (auto *p = fast<std::uint8_t>(a)) return load<std::uint16_t>(p);
    fence(c);
    const std::uint16_t v = slow_ld16(a);
    fence(c);
    return v;
}

inline std::uint32_t ld32(Cpu &c, std::uint32_t a) {
    if (auto *p = fast<std::uint8_t>(a)) return load<std::uint32_t>(p);
    fence(c);
    const std::uint32_t v = slow_ld32(a);
    fence(c);
    return v;
}

inline std::uint64_t ld64(Cpu &c, std::uint32_t a) {
    if (auto *p = fast<std::uint8_t>(a)) return load<std::uint64_t>(p);
    fence(c);
    const std::uint64_t v = slow_ld64(a);
    fence(c);
    return v;
}

inline void ld128(Cpu &c, std::uint32_t a, Reg128 &out) {
    if (auto *p = fast<std::uint8_t>(a)) { std::memcpy(&out, p, 16); return; }
    fence(c);
    slow_ld128(a, out);
    fence(c);
}

// A store: the common case as in st8..st128 above; anything else (the watch, scratchpad, registers, mirrors) takes the plain path fenced.
#ifdef PS2_LIVE_EE
#define PS2_GEN_STORE(T, plain)                                                    \
    if (!g_watch_on) if (auto *p = fast<std::uint8_t>(a)) { store<T>(p, v); return; } \
    fence(c);                                                                      \
    plain(a, v);                                                                   \
    fence(c);
#else
#define PS2_GEN_STORE(T, plain)                                                    \
    if (a < g_store_ram_end) [[likely]] { store<T>(PS2_RAM + a, v); return; }       \
    fence(c);                                                                      \
    plain(a, v);                                                                   \
    fence(c);
#endif

inline void st8(Cpu &c, std::uint32_t a, std::uint8_t v) { PS2_GEN_STORE(std::uint8_t, st8) }
inline void st16(Cpu &c, std::uint32_t a, std::uint16_t v) { PS2_GEN_STORE(std::uint16_t, st16) }
inline void st32(Cpu &c, std::uint32_t a, std::uint32_t v) { PS2_GEN_STORE(std::uint32_t, st32) }
inline void st64(Cpu &c, std::uint32_t a, std::uint64_t v) { PS2_GEN_STORE(std::uint64_t, st64) }
inline void st128(Cpu &c, std::uint32_t a, const Reg128 &v) { PS2_GEN_STORE(Reg128, st128) }
#undef PS2_GEN_STORE

inline void lwl(Cpu &c, Reg128 &rt, std::uint32_t a) {
    const std::uint32_t shift = (3u - (a & 3u)) * 8u;
    const std::uint32_t mem = ld32(c, a & ~3u);
    const std::uint32_t v = (mem << shift) | (rt.w[0] & static_cast<std::uint32_t>(mask_below(shift)));
    rt.sd[0] = static_cast<std::int32_t>(v);
}

inline void lwr(Cpu &c, Reg128 &rt, std::uint32_t a) {
    const std::uint32_t shift = (a & 3u) * 8u;
    const std::uint32_t mem = ld32(c, a & ~3u);
    const std::uint32_t v = (mem >> shift) | (rt.w[0] & ~static_cast<std::uint32_t>(mask_below(32 - shift)));
    if (shift == 0) rt.sd[0] = static_cast<std::int32_t>(v);
    else rt.w[0] = v;
}

inline void ldl(Cpu &c, Reg128 &rt, std::uint32_t a) {
    const std::uint32_t shift = (7u - (a & 7u)) * 8u;
    const std::uint64_t mem = ld64(c, a & ~7u);
    rt.d[0] = (mem << shift) | (rt.d[0] & mask_below(shift));
}

inline void ldr(Cpu &c, Reg128 &rt, std::uint32_t a) {
    const std::uint32_t shift = (a & 7u) * 8u;
    const std::uint64_t mem = ld64(c, a & ~7u);
    rt.d[0] = (mem >> shift) | (rt.d[0] & ~mask_below(64 - shift));
}

inline void swl(Cpu &c, const Reg128 &rt, std::uint32_t a) {
    const std::uint32_t shift = (3u - (a & 3u)) * 8u;
    const std::uint32_t mem = ld32(c, a & ~3u);
    st32(c, a & ~3u, (rt.w[0] >> shift) | (mem & ~static_cast<std::uint32_t>(mask_below(32 - shift))));
}

inline void swr(Cpu &c, const Reg128 &rt, std::uint32_t a) {
    const std::uint32_t shift = (a & 3u) * 8u;
    const std::uint32_t mem = ld32(c, a & ~3u);
    st32(c, a & ~3u, (rt.w[0] << shift) | (mem & static_cast<std::uint32_t>(mask_below(shift))));
}

inline void sdl(Cpu &c, const Reg128 &rt, std::uint32_t a) {
    const std::uint32_t shift = (7u - (a & 7u)) * 8u;
    const std::uint64_t mem = ld64(c, a & ~7u);
    st64(c, a & ~7u, (rt.d[0] >> shift) | (mem & ~mask_below(64 - shift)));
}

inline void sdr(Cpu &c, const Reg128 &rt, std::uint32_t a) {
    const std::uint32_t shift = (a & 7u) * 8u;
    const std::uint64_t mem = ld64(c, a & ~7u);
    st64(c, a & ~7u, (rt.d[0] << shift) | (mem & mask_below(shift)));
}

}
