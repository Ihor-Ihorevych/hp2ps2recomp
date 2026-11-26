#!/usr/bin/env bash
# Builds what Flipendo Droid's PS2 tab needs and puts it in flipendo-deps/ps2out, where its build-apk.sh takes it from:
#   libps2main.so    the recompiling runner (ps2-native-recomp, no game code in it), with SDL2's API
#   libSDL2ps2.so    SDL2's API over this app's SDL3 (sdl2-compat; renamed, the HP3 engine has a real libSDL2.so)
#   libps2clang.so, libps2lld.so, sysroot.zip   the compiler the phone translates a game with (flipendo/build-toolchain.sh)
#
#   bash flipendo/build-ps2.sh        (Git Bash on Windows; after flipendo/build-toolchain.sh, or with its three files in flipendo-deps/ps2cc/out)
# Needs the NDK 28 and CMake 3.22.1 of the Android SDK and git. The Android wgpu-native (the pinned upstream release) and SDL3's
# headers are fetched on the first run.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
PS2RECOMP="${PS2RECOMP:-$ROOT}"
SDK="$(cygpath -u "${ANDROID_HOME:-$(cygpath "$LOCALAPPDATA")/Android/Sdk}")"
NDK="${ANDROID_NDK:-$(ls -d "$SDK"/ndk/28.* | sort | tail -n1)}"
CM="$SDK/cmake/3.22.1/bin"
DEPS="$ROOT/flipendo-deps"
OUT="$DEPS/ps2out"
mkdir -p "$OUT"
JOBS="${PS2_JOBS:-$(nproc)}"

# 0. SDL3's headers (sdl2-compat is built against them) and the Android wgpu-native (the release this runtime's graphics code is written for)
if [ ! -d "$DEPS/SDL" ]; then
	git clone -q --depth 1 --branch release-3.2.20 https://github.com/libsdl-org/SDL "$DEPS/SDL"
fi
WG="$PS2RECOMP/third_party/wgpu-native-android"
WGPU_VERSION=v29.0.1.1
WGPU_SHA256=721741f1b05a20c1738166bedf7a5efb2ba4b382da689526d3fc33de22bdd573
if [ ! -f "$WG/lib/libwgpu_native.a" ]; then
	mkdir -p "$WG"
	curl -fsSL -o "$WG/wgpu.zip" "https://github.com/gfx-rs/wgpu-native/releases/download/$WGPU_VERSION/wgpu-android-aarch64-release.zip"
	got="$(sha256sum "$WG/wgpu.zip" | cut -d' ' -f1)"
	[ "$got" = "$WGPU_SHA256" ] || { echo "wgpu-native: wrong SHA-256 $got (expected $WGPU_SHA256)"; exit 1; }
	powershell -NoProfile -Command "Expand-Archive -Force -Path '$(cygpath -w "$WG/wgpu.zip")' -DestinationPath '$(cygpath -w "$WG")'"
	rm -f "$WG/wgpu.zip"
fi

# 1. SDL2's API over SDL3: sdl2-compat finds SDL3 at run time (the app's libSDL3.so), so only SDL3's headers are needed to build it
if [ ! -d "$DEPS/sdl2-compat" ]; then
	git clone -q --depth 1 --branch release-2.32.66 https://github.com/libsdl-org/sdl2-compat "$DEPS/sdl2-compat"
	# the HP3 engine brings a real libSDL2.so: this one must not share its name
	printf '\nset_target_properties(SDL2 PROPERTIES OUTPUT_NAME "SDL2ps2")\n' >> "$DEPS/sdl2-compat/CMakeLists.txt"
fi
if [ ! -f "$DEPS/sdl2-compat-install/lib/libSDL2ps2.so" ]; then
	rm -rf "$DEPS/sdl2-compat-build"
	"$CM/cmake.exe" -S "$DEPS/sdl2-compat" -B "$DEPS/sdl2-compat-build" -G Ninja -DCMAKE_MAKE_PROGRAM="$CM/ninja.exe" \
		-DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29 \
		-DCMAKE_BUILD_TYPE=Release -DSDL3_INCLUDE_DIRS="$(cygpath -m "$DEPS/SDL/include")" -DSDL2COMPAT_TESTS=OFF -DSDL2COMPAT_STATIC=OFF \
		-DSDL2COMPAT_X11=OFF -DSDL2COMPAT_INSTALL=ON -DCMAKE_INSTALL_PREFIX="$(cygpath -m "$DEPS/sdl2-compat-install")"
	"$CM/ninja.exe" -C "$DEPS/sdl2-compat-build" install
fi

# 2. the runner, with no game linked in (host/empty_game): the JNI names are those of Flipendo's Java classes
RB="$PS2RECOMP/build/android-ps2"
"$CM/cmake.exe" -S "$PS2RECOMP" -B "$RB" -G Ninja -DCMAKE_MAKE_PROGRAM="$CM/ninja.exe" \
	-DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29 \
	-DCMAKE_BUILD_TYPE=Release -DCMAKE_SHARED_LINKER_FLAGS=-flto=thin -DPS2_GENERATED="$(cygpath -m "$PS2RECOMP/host/empty_game")" \
	-DPS2_WGPU_DIR="$(cygpath -m "$PS2RECOMP/third_party/wgpu-native-android")" \
	-DSDL2_DIR="$(cygpath -m "$DEPS/sdl2-compat-install/lib/cmake/SDL2")" -DPS2_ANDROID_LIB_NAME=ps2main \
	"-DCMAKE_CXX_FLAGS=-flto=thin -DPS2_JNI_TOUCH=io_github_flipendo_spike_Ps2Pad -DPS2_JNI_ACTIVITY=io_github_flipendo_spike_Ps2Activity"
"$CM/ninja.exe" -C "$RB" -j"$JOBS" ps2run
STRIP="$NDK/toolchains/llvm/prebuilt/windows-x86_64/bin/llvm-strip.exe"
cp "$RB/libps2main.so" "$OUT/libps2main.so"
cp "$DEPS/sdl2-compat-install/lib/libSDL2ps2.so" "$OUT/libSDL2ps2.so"

# 3. the compiler (android/ps2/build-toolchain.sh made it; it takes a long time, so it is not started from here)
for f in libps2clang.so libps2lld.so sysroot.zip; do
	[ -f "$DEPS/ps2cc/out/$f" ] || { echo "missing $DEPS/ps2cc/out/$f: run flipendo/build-toolchain.sh first"; exit 1; }
	cp "$DEPS/ps2cc/out/$f" "$OUT/$f"
done
"$STRIP" --strip-unneeded "$OUT/libps2main.so" "$OUT/libSDL2ps2.so"
ls -la "$OUT"
