#pragma once
// The governor: a weak phone keeps the game at full speed (every field emulated, the sound never stretched) by
// drawing fewer pictures when it cannot keep up. The host's speed limiter tells it each field how much time was left (or how late the
// field was), the phone's app tells it once a second how hot the phone is (nativeDeviceNote), and the hardware renderer asks it at the
// start of each picture whether this picture's draws may be skipped (gs/gpu/hwr.cpp: a skipped picture is not shown, the last one stays).
//
// PS2_FRAMESKIP=0: never skip. Unset or 1: skip at most every other picture, only while the game is behind. force1: skip every other
// picture always (a test aid: the game must go exactly the same way, scripts/replay.py).
//
// The rules are the small pure parts below (tests/governor.cpp checks them without a game); runtime/src/governor.cpp keeps the state.
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

namespace ps2::governor {

enum Mode { kOff = 0, kAuto = 1, kForce = 2 };

// What the app's device line says: "thermal status 2 (...), headroom 0.73 (...), ..." (android SpikeActivity.deviceTick). -1: not given
// (the headroom is NaN on phones that do not forecast it).
struct Device {
    int status = -1;
    float headroom = -1.0f;
};

inline Device parse_device(const char *text) {
    Device d;
    if (!text) return d;
    // "thermal status 2" (SpikeActivity) or "thermal 2" (Ps2Activity.deviceNote)
    for (const char *key : {"thermal status ", "thermal "}) {
        const char *p = std::strstr(text, key);
        if (!p) continue;
        const std::size_t n = std::strlen(key);
        char *end = nullptr;
        const long v = std::strtol(p + n, &end, 10);
        if (end != p + n) {
            d.status = static_cast<int>(v);
            break;
        }
    }
    if (const char *p = std::strstr(text, "headroom ")) {
        char *end = nullptr;
        const float v = std::strtof(p + 9, &end);
        if (end != p + 9 && std::isfinite(v)) d.headroom = v;
    }
    return d;
}

// Hot enough to draw less before the phone slows itself down: status severe (3) or worse, or the forecast headroom at 0.9 of the
// point where throttling starts.
inline bool device_hot(const Device &d) { return d.status >= 3 || d.headroom >= 0.9f; }

// The speed limiter's view, a window of fields at a time: `slack_ms` is what the limiter waited at a field (negative: the field came
// that late), `restarted` that it gave up catching up (the game fell several fields behind). Behind only when the game really lost time:
// the clock had to restart, the sound was being stretched, or the fields of a whole window came more than a period late on average.
// Little or no wait is not behind: a game at full load (Harry Potter 2 draws in one field and none in the next, so the drawn field comes
// late and the next one on time) keeps real time; skipping then only halves the pictures (the phone showed 50 fields a second and a
// quarter of the pictures skipped under the earlier "under 5 % unused" rule). Caught up again after two windows that lost no time and
// came less than half a period late on average.
struct Pace {
    static constexpr int kWindow = 25;  // fields (half a second PAL)
    int fields = 0, calm = 0;
    double slack = 0, period = 0;
    bool late = false;  // a restart or stretched sound in this window
    bool behind = false;
    void field(double slack_ms, double period_ms, bool restarted, bool stretched) {
        slack += slack_ms;
        period += period_ms;
        late = late || restarted || stretched;
        if (++fields < kWindow) return;
        const double avg = slack / kWindow, per = period / kWindow;
        if (late || avg < -per) {
            behind = true;
            calm = 0;
        } else if (avg > -per / 2) {
            if (++calm >= 2) behind = false;
        } else {
            calm = 0;
        }
        fields = 0, slack = 0, period = 0, late = false;
    }
};

// One decision per picture: never two skipped in a row (half the pictures at the least: 12.5 a second for a game that draws 25).
struct Skipper {
    Mode mode = kAuto;
    bool last = false;
    bool decide(bool behind) {
        const bool skip = !last && (mode == kForce || (mode == kAuto && behind));
        last = skip;
        return skip;
    }
};

// ---------------------------------------------------------------- the runtime's state (runtime/src/governor.cpp)
Mode mode();                                                                  // PS2_FRAMESKIP, read once
void note_pace(double slack_ms, double period_ms, bool restarted, bool stretched);  // the host's limiter, every field
void note_device(const char *text);                                           // the app's device line, once a second
bool behind();                                                                // the limiter or the phone's heat says draw less
bool skip_picture();                                                          // the renderer, at the start of each picture
void picture_done(bool skipped);                                              // the renderer: a picture was shown or skipped (counted)
void disable(const char *why);                                                // skipping lost something a later picture needed: no more this run
bool disabled();
void report();                                                                // once a second with PS2_FRAMESKIP_LOG=1, and at the end

}  // namespace ps2::governor
