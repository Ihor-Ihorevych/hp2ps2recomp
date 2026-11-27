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
#include <string>
#include <thread>
#include <unordered_map>
#include <cstdio>
#include <cstring>
#else
#include <time.h>
#endif
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <x86intrin.h>
#endif

namespace ps2::gs::drawtime {

enum Part { kGif, kShape, kBuild, kPacked, kParts };

inline bool g_on = false;
inline std::uint64_t g_sum[kParts] = {};
inline std::uint64_t g_count[kParts] = {};
// why gs.cpp closed an open run of triangles: a register write, a PACKED write outside a run, image data, another message (debug counts)
inline std::uint64_t g_close[5] = {};
// primitives of runs that hwr::shapes handed to shape() one by one: two-pass alpha test, own target read, view 4, sprites/other kinds
inline std::uint64_t g_slow[4] = {};

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


#if defined(_WIN32) && (defined(__x86_64__) || defined(_M_X64))
#define PS2_DRAW_SAMPLER 1
// PS2_DRAW_SAMPLE=<file> (Windows, a measuring aid: Windows has no SIGPROF): a thread looks at the calling thread's instruction pointer
// about once a millisecond and writes how often each address was seen to <file> every few thousand looks (address as linked, base
// 0x140000000: map with llvm-nm). Started once, from the drawing thread.
inline void start_sampler(const char *path) {
    HANDLE self = nullptr;
    DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &self, THREAD_ALL_ACCESS, FALSE, 0);
    const std::uint64_t base = reinterpret_cast<std::uint64_t>(GetModuleHandleA(nullptr));
    std::string file(path);
    std::thread([self, base, file] {
        std::unordered_map<std::uint64_t, std::uint32_t> seen;
        for (std::uint64_t n = 1;; ++n) {
            Sleep(1);
            if (SuspendThread(self) == static_cast<DWORD>(-1)) return;
            CONTEXT ctx{};
            ctx.ContextFlags = CONTEXT_CONTROL;
            const bool ok = GetThreadContext(self, &ctx);
            ResumeThread(self);
            if (ok) ++seen[ctx.Rip];
            if (n % 500 == 0) {
                if (std::FILE *f = std::fopen(file.c_str(), "w")) {
                    for (const auto &[at, count] : seen) {
                        HMODULE m = nullptr;
                        char name[MAX_PATH] = "?";
                        GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCSTR>(at), &m);
                        const std::uint64_t mb = reinterpret_cast<std::uint64_t>(m);
                        if (mb == base) std::fprintf(f, "%llx %u\n", static_cast<unsigned long long>(at - base + 0x140000000ull), count);
                        else {
                            if (m) GetModuleFileNameA(m, name, MAX_PATH);
                            const char *short_name = std::strrchr(name, '\\') ? std::strrchr(name, '\\') + 1 : name;
                            std::fprintf(f, "%s+%llx %u\n", short_name, static_cast<unsigned long long>(at - mb), count);
                        }
                    }
                    std::fclose(f);
                }
            }
        }
    }).detach();
}
#endif

}  // namespace ps2::gs::drawtime
