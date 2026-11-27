// The governor's rules (runtime/include/ps2/governor.hpp), no game needed: ninja ps2governortest && ./ps2governortest
#include "ps2/governor.hpp"

#include <cstdio>

namespace {

int g_failed = 0;

void check(bool ok, const char *what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++g_failed;
}

using namespace ps2::governor;

// `windows` windows of fields that each wait `slack` ms of a 20 ms period
bool run(Pace &p, int windows, double slack, bool restarted = false, bool stretched = false) {
    for (int i = 0; i < windows * Pace::kWindow; ++i) p.field(slack, 20.0, restarted && i == 3, stretched && i == 3);
    return p.behind;
}

}  // namespace

int main() {
    {
        const char *line = "thermal status 3 (0 none, 1 light, 2 moderate, 3 severe, 4 critical), headroom 0.62 (1.0 = throttling starts), display 90.0 Hz, battery 38.5 C, plugged 0, power saver false";
        const Device d = parse_device(line);
        check(d.status == 3 && d.headroom > 0.61f && d.headroom < 0.63f, "the app's line: status 3, headroom 0.62");
        check(device_hot(d), "status 3 is hot");
        check(!device_hot(parse_device("thermal status 2 (...), headroom 0.89 (...)")), "status 2, headroom 0.89 is not hot");
        check(device_hot(parse_device("thermal status 1 (...), headroom 0.9 (...)")), "headroom 0.9 is hot");
        const Device nan = parse_device("thermal status 0 (...), headroom NaN (1.0 = throttling starts)");
        check(nan.status == 0 && nan.headroom < 0 && !device_hot(nan), "a NaN headroom counts as not given");
        check(!device_hot(parse_device("")) && !device_hot(parse_device(nullptr)), "no line: not hot");
    }
    {
        Pace p;
        check(!run(p, 4, 8.0), "8 ms of 20 left each field: keeping up");
        check(run(p, 1, 0.5), "0.5 ms of 20 left: behind after one window");
        check(run(p, 1, 3.0), "15 % left: still behind (between the bars)");
        check(run(p, 1, 8.0), "40 % left once: still behind (two windows needed)");
        check(!run(p, 1, 8.0), "40 % left twice: keeping up again");
        check(run(p, 1, 8.0, true), "a restart of the clock is behind, whatever is left");
        Pace q;
        check(run(q, 1, 8.0, false, true), "stretched sound is behind");
        Pace r;
        // Harry Potter 2: a drawn field takes two periods, the empty one after it none: no wait at all at full load
        for (int i = 0; i < Pace::kWindow; ++i) r.field(i % 2 ? 0.0 : -19.0, 20.0, false, false);
        check(r.behind, "no wait in a whole window: behind");
    }
    {
        Skipper s;
        s.mode = kAuto;
        int skipped = 0;
        for (int i = 0; i < 10; ++i) skipped += s.decide(false);
        check(skipped == 0, "auto, keeping up: nothing skipped");
        bool two = false, last = false;
        skipped = 0;
        for (int i = 0; i < 10; ++i) {
            const bool k = s.decide(true);
            two = two || (k && last);
            last = k;
            skipped += k;
        }
        check(skipped == 5 && !two, "auto, behind: every other picture, never two in a row");
        Skipper f;
        f.mode = kForce;
        skipped = 0;
        for (int i = 0; i < 10; ++i) skipped += f.decide(false);
        check(skipped == 5, "force1: every other picture even when keeping up");
        Skipper o;
        o.mode = kOff;
        skipped = 0;
        for (int i = 0; i < 10; ++i) skipped += o.decide(true);
        check(skipped == 0, "off: nothing skipped when behind");
    }
    std::printf(g_failed ? "%d FAILED\n" : "all passed\n", g_failed);
    return g_failed ? 1 : 0;
}
