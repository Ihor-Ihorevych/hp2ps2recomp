# Flipendo Droid's PS2 support

This fork of [ps2-native-recomp](https://github.com/exoteric33/ps2-native-recomp) (MIT, by Samir Nuri; this repository keeps its history and
licence) is what the **PS2 tab of [Flipendo Droid](https://github.com/Ihor-Ihorevych/flipendo-droid)** runs: Harry Potter and the Chamber of
Secrets (PS2) on Android phones. Nothing from any game is in this repository; the phone translates the game from the player's own disc.

* `flipendo/build-toolchain.sh`: clang and lld for the phone (LLVM 20.1.8, AArch64 only) and a small sysroot, which the phone uses to compile
  the translated game. Takes hours; run once.
* `flipendo/build-ps2.sh`: the runner (`libps2main.so`) and SDL2's API over the app's SDL3 (`libSDL2ps2.so`), and the folder
  `flipendo-deps/ps2out` Flipendo Droid's `android/build-apk.sh` takes everything from. Git Bash on Windows, Android NDK 28, CMake 3.22.1.
* `.github/workflows/flipendo-android.yml`: builds both on GitHub and publishes them as releases: `toolchain-llvm20.1.8-ndk28.2` (the
  compiler, built only when that release is missing) and `runner-<commit>` (one per build). Flipendo Droid's workflow downloads them.

The graphics library is the upstream wgpu-native release `v29.0.1.1` for Android (downloaded by `build-ps2.sh`, SHA-256 checked).
Runtime switches, profiling and the VU1 notes are in `docs/` (`debugging-native.md`, `debugging-vu1.md`, `hw-renderer-progress.md`).
