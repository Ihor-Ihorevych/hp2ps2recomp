// JNI bridge of the touch overlay (android-spike/.../TouchOverlay.java): the buttons are the PS2 pad's bit mask (host/window.cpp Button).
#include "android_touch.hpp"
#include <jni.h>
#include "ps2/video.hpp"
#include "ps2/prof.hpp"
#include "ps2/governor.hpp"
#include <android/log.h>
#include <atomic>
#include <cstdio>

// The Java classes that declare these natives: their names are part of the symbols. Another app names its own with
// -DPS2_JNI_TOUCH=io_github_flipendo_spike_Ps2Pad -DPS2_JNI_ACTIVITY=io_github_flipendo_spike_Ps2Activity (underscores for dots).
#ifndef PS2_JNI_TOUCH
#define PS2_JNI_TOUCH org_ps2recomp_spike_TouchOverlay
#endif
#ifndef PS2_JNI_ACTIVITY
#define PS2_JNI_ACTIVITY org_ps2recomp_spike_SpikeActivity
#endif
#define PS2_JNI_NAME2(cls, fn) Java_##cls##_##fn
#define PS2_JNI_NAME(cls, fn) PS2_JNI_NAME2(cls, fn)

namespace {
std::atomic<std::uint32_t> g_buttons{0};
std::atomic<std::uint32_t> g_stick{0x8080};  // ly << 8 | lx
std::atomic<std::uint32_t> g_logged{~0u};
}

void android_touch_merge(std::uint16_t &pressed, std::uint8_t &lx, std::uint8_t &ly) {
    const std::uint32_t b = g_buttons.load(), s = g_stick.load();
    pressed |= static_cast<std::uint16_t>(b);
    if ((s & 0xFF) != 0x80) lx = static_cast<std::uint8_t>(s & 0xFF);
    if (((s >> 8) & 0xFF) != 0x80) ly = static_cast<std::uint8_t>((s >> 8) & 0xFF);
}

extern "C" JNIEXPORT void JNICALL PS2_JNI_NAME(PS2_JNI_TOUCH, nativeSet)(JNIEnv *, jclass, jint buttons, jint lx, jint ly) {
    g_buttons.store(static_cast<std::uint32_t>(buttons) & 0xFFFF);
    g_stick.store(static_cast<std::uint32_t>((ly & 0xFF) << 8 | (lx & 0xFF)));
    const std::uint32_t key = (static_cast<std::uint32_t>(buttons) & 0xFFFF) | static_cast<std::uint32_t>((lx & 0xFF) << 16) | static_cast<std::uint32_t>((ly & 0xFF) << 24);
    if (g_logged.exchange(key) != key) {
        __android_log_print(ANDROID_LOG_INFO, "ps2touch", "buttons %04X stick %d %d", buttons & 0xFFFF, lx, ly);
        std::fprintf(stderr, "[touch] buttons %04X left stick %d %d\n", buttons & 0xFFFF, lx, ly);
    }
}

namespace {
std::atomic<float> g_fps{0.0f}, g_hz{50.0f};
}

void android_note_fps(float fps, float hz) {
    g_fps.store(fps);
    g_hz.store(hz);
}

// the FPS counter of the app: { pictures per second, the console's rate (50 PAL / 59.94 NTSC) }
extern "C" JNIEXPORT jfloatArray JNICALL PS2_JNI_NAME(PS2_JNI_ACTIVITY, nativeFps)(JNIEnv *env, jclass) {
    const jfloat v[2] = {g_fps.load(), g_hz.load()};
    jfloatArray out = env->NewFloatArray(2);
    env->SetFloatArrayRegion(out, 0, 2, v);
    return out;
}

// the heavy runtime threads' kernel ids, for the performance hint session (PerformanceHintManager)
extern "C" JNIEXPORT jintArray JNICALL PS2_JNI_NAME(PS2_JNI_ACTIVITY, nativeThreadIds)(JNIEnv *env, jclass) {
    int ids[16];
    const int n = ps2::prof::thread_ids(ids, 16);
    jintArray out = env->NewIntArray(n);
    if (n > 0) env->SetIntArrayRegion(out, 0, n, reinterpret_cast<const jint *>(ids));
    return out;
}

// The phone's thermal state, battery temperature and charging, once a second from the activity: for PS2_PROF's output and the governor
extern "C" JNIEXPORT void JNICALL PS2_JNI_NAME(PS2_JNI_ACTIVITY, nativeDeviceNote)(JNIEnv *env, jclass, jstring text) {
    if (!text) return;
    const char *chars = env->GetStringUTFChars(text, nullptr);
    if (chars) {
        ps2::prof::set_note(chars);
        ps2::governor::note_device(chars);  // a hot phone draws fewer pictures (ps2/governor.hpp)
        env->ReleaseStringUTFChars(text, chars);
    }
}

// live display mode (ps2::display_mode: 0 original 4:3, 1 stretch, 2 zoom), from the settings panel
extern "C" JNIEXPORT void JNICALL PS2_JNI_NAME(PS2_JNI_TOUCH, nativeDisplayMode)(JNIEnv *, jclass, jint mode) { ps2::set_display_mode(mode); }
