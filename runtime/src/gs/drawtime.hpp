#pragma once
// PS2_DRAW_TIME=1: where the drawing thread's time goes while it plays the GS stream back, measured with the processor's cycle counter (a
// read costs a few nanoseconds, unlike two clock reads per shape). Summed per presented picture and printed every 100 pictures by hwr
// (frame_end): GIF messages in all (gif), the shapes handed to the hardware renderer (shape, incl. build), their state set-ups (build).
// Off: one predictable branch per message and per shape.
#include <chrono>
#include <cstdint>
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <time.h>
#endif
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <x86intrin.h>
#endif

namespace ps2::gs::drawtime {

enum Part { kGif, kShape, kBuild, kParts };

inline bool g_on = false;
inline std::uint64_t g_sum[kParts] = {};
inline std::uint64_t g_count[kParts] = {};

inline std::uint64_t now() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    return __rdtsc();
#elif defined(__aarch64__)
    std::uint64_t v;
    asm volatile("mrs %0, cntvct_el0" : "=r"(v));
    return v;
#else
    return static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
#endif
}

// This thread's processor time in ms (not the time it was preempted or waited): the drawing thread's own cost, on a busy machine too.
// Windows: its cycles (QueryThreadCycleTime) at the rate `ms_per_tick` of now() (both the time-stamp counter).
inline double thread_ms(double ms_per_tick) {
#if defined(_WIN32)
    ULONG64 cycles = 0;
    QueryThreadCycleTime(GetCurrentThread(), &cycles);
    return static_cast<double>(cycles) * ms_per_tick;
#else
    (void)ms_per_tick;
    timespec ts{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return static_cast<double>(ts.tv_sec) * 1e3 + static_cast<double>(ts.tv_nsec) * 1e-6;
#endif
}

struct Span {
    Part part;
    std::uint64_t from;
    explicit Span(Part p) : part(p), from(g_on ? now() : 0) {}
    ~Span() {
        if (g_on) g_sum[part] += now() - from, ++g_count[part];
    }
    Span(const Span &) = delete;
    Span &operator=(const Span &) = delete;
};

}  // namespace ps2::gs::drawtime
