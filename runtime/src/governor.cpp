// The governor's state (runtime/include/ps2/governor.hpp): fed by the host's speed limiter (emulation thread) and the app's device line
// (the app's own thread), asked by the hardware renderer (drawing thread).
#include "ps2/governor.hpp"
#include "ps2/os.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>

namespace ps2::governor {

namespace {
Pace g_pace;  // emulation thread only
Skipper g_skip;  // drawing thread only
std::atomic<bool> g_pace_behind{false}, g_hot{false}, g_disabled{false};
std::atomic<int> g_status{-1};
std::atomic<float> g_headroom{-1.0f};
std::atomic<std::uint64_t> g_shown{0}, g_skipped{0}, g_behind_windows{0};
const bool g_log = os::env("PS2_FRAMESKIP_LOG") == "1";
}  // namespace

Mode mode() {
    static const Mode m = [] {
        const std::string v = os::env("PS2_FRAMESKIP");
        const Mode r = v == "0" ? kOff : v == "force1" ? kForce : kAuto;
        if (r == kForce) std::fprintf(stderr, "[governor] PS2_FRAMESKIP=force1: every other picture is skipped (a test aid)\n");
        else if (r == kOff) std::fprintf(stderr, "[governor] PS2_FRAMESKIP=0: every picture is drawn\n");
        return r;
    }();
    return m;
}

void note_pace(double slack_ms, double period_ms, bool restarted, bool stretched) {
    const bool was = g_pace.behind;
    const int fields = g_pace.fields;
    g_pace.field(slack_ms, period_ms, restarted, stretched);
    if (fields + 1 == Pace::kWindow && g_pace.behind) g_behind_windows.fetch_add(1, std::memory_order_relaxed);
    if (was != g_pace.behind) {
        g_pace_behind.store(g_pace.behind, std::memory_order_relaxed);
        if (g_log) std::fprintf(stderr, "[governor] %s\n", g_pace.behind ? "behind: pictures may be skipped" : "keeping up again: every picture is drawn");
    }
    if (g_log) {
        static auto since = std::chrono::steady_clock::now();
        if (std::chrono::steady_clock::now() - since >= std::chrono::seconds(1)) {
            since = std::chrono::steady_clock::now();
            report();
        }
    }
}

void note_device(const char *text) {
    const Device d = parse_device(text);
    g_status.store(d.status, std::memory_order_relaxed);
    g_headroom.store(d.headroom, std::memory_order_relaxed);
    const bool hot = device_hot(d);
    if (hot != g_hot.exchange(hot, std::memory_order_relaxed))
        std::fprintf(stderr, "[governor] the phone is %s (thermal status %d, headroom %.2f)\n", hot ? "hot: pictures may be skipped" : "cool again", d.status, static_cast<double>(d.headroom));
}

bool behind() { return g_pace_behind.load(std::memory_order_relaxed) || g_hot.load(std::memory_order_relaxed); }

bool skip_picture() {
    g_skip.mode = mode();
    if (g_skip.mode == kOff || g_disabled.load(std::memory_order_relaxed)) {
        g_skip.last = false;
        return false;
    }
    return g_skip.decide(behind());
}

void picture_done(bool skipped) { (skipped ? g_skipped : g_shown).fetch_add(1, std::memory_order_relaxed); }

void disable(const char *why) {
    if (g_disabled.exchange(true)) return;
    std::fprintf(stderr, "[governor] picture skipping is off for the rest of the run: %s\n", why);
}

bool disabled() { return g_disabled.load(std::memory_order_relaxed); }

void report() {
    static std::uint64_t last_shown = 0, last_skipped = 0;
    const std::uint64_t shown = g_shown.load(), skipped = g_skipped.load();
    std::fprintf(stderr, "[governor] pictures drawn %llu, skipped %llu (in all %llu / %llu); %s%s%s, thermal status %d, headroom %.2f\n",
                 static_cast<unsigned long long>(shown - last_shown), static_cast<unsigned long long>(skipped - last_skipped), static_cast<unsigned long long>(shown),
                 static_cast<unsigned long long>(skipped), g_pace_behind.load() ? "behind" : "keeping up", g_hot.load() ? ", hot" : "", g_disabled.load() ? ", skipping disabled" : "",
                 g_status.load(), static_cast<double>(g_headroom.load()));
    last_shown = shown, last_skipped = skipped;
}

}  // namespace ps2::governor
