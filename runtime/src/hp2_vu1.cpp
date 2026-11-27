// Harry Potter 2 (SLES-51192): native versions of its VU1 microcode stages (the "native dispatcher", plan M4).
//
// HP2's VU1 microcode is a dispatcher (0x0010) that runs up to four stages per batch from a table, the last one a transform that writes
// the GIF packet (ST, RGBA, XYZ per vertex) and hands over to the kick program (0x0140). The compiled programs (recomp/vurecomp_gen.cpp)
// do every pair as the console does, with the pipeline's timing; the stages here do the same arithmetic per vertex in a plain loop.
// vu_micro.cpp calls them in place of a compiled program when the program's key (the code from its entry) is the one written here; a
// stage that cannot take a batch (a triangle that has to be clipped, an odd count) returns false and the compiled program runs.
//
// PS2_HP2_NATIVE (fast vector unit only, vu::fast_mode): unset or 1 = on, with the guard band below; exact = on without it (the results
// are those of the compiled fast-mode program, bit for bit); check = run the stage on a copy, then the compiled program, and compare
// everything (registers, flags, the 16 KB of data memory), printing the first differences; 0 = off.
//
// The guard band: the microcode sends every triangle with a corner outside the view volume (any clip flag of its three corners) through
// its clipper (0x0CB8), which is more than half of all the program switches in a busy scene. A triangle whose corners are all in front
// of the camera, between the near and far planes, and inside the GS's coordinate range (0 .. 4095.9375 after the viewport) needs no
// clipping on this renderer: the card clips it to the screen. Such a triangle is drawn as it is; only a triangle that really needs the
// clipper makes the whole batch go to the compiled program.
#include "ps2/cpu.hpp"
#include "ps2/fpu.hpp"
#include "ps2/vu.hpp"
#include "ps2/vu_ops.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ps2::hp2 {

using vuops::v4u;
typedef float v4f __attribute__((vector_size(16)));

namespace {

inline std::uint8_t *qw(std::uint8_t *mem, std::uint32_t index) { return mem + (index & 1023u) * 16u; }
inline v4u ld(const std::uint8_t *p) {
    v4u v;
    std::memcpy(&v, p, 16);
    return v;
}
inline void st(std::uint8_t *p, v4u v) { std::memcpy(p, &v, 16); }
inline v4u rv(const Reg128 &r) {
    v4u v;
    std::memcpy(&v, &r, 16);
    return v;
}
inline void wv(Reg128 &r, v4u v) { std::memcpy(&r, &v, 16); }
inline v4u splat(std::uint32_t x) { return v4u{x, x, x, x}; }
// what the fast programs' operands go through (vshadow / vqd: an exponent of 255 is the largest float with its sign)
inline v4u clampv(v4u x) {
    const v4u e255 = (x & 0x7F800000u) == 0x7F800000u;
    return (x & ~e255) | (((x & 0x80000000u) | 0x7F7FFFFFu) & e255);
}
// A result as the compiled fast programs leave it: unclamped on arm64 (vu_ops.hpp PS2_VU_NOCLAMP with flush-to-zero), clamped where the
// live libraries are built without SSE4.1 (the PC's x86-64 default: vclamp_result is the whole vfix there)
#if defined(__aarch64__)
inline v4u res(v4u x) { return x; }
#else
inline v4u res(v4u x) { return clampv(x); }
#endif
inline v4u mul(v4u a, v4u b) { return res((v4u)((v4f)a * (v4f)b)); }
inline v4u add(v4u a, v4u b) { return res((v4u)((v4f)a + (v4f)b)); }
inline v4u sel(v4u old, v4u now, unsigned dest) {  // dest: x = 8, y = 4, z = 2, w = 1
    const v4u m{dest & 8u ? ~0u : 0u, dest & 4u ? ~0u : 0u, dest & 2u ? ~0u : 0u, dest & 1u ? ~0u : 0u};
    return (now & m) | (old & ~m);
}
inline std::uint32_t clip_flags(v4u s, std::uint32_t w) { return vuops::clip_flags(s, w); }  // CLIP fs.xyz, ft.w (vu_ops.hpp)
// MAC flags of a result (FlagPipeV::mac_of without the SSE form): zero, sign, overflow (exponent 255 with an all-ones mantissa)
inline std::uint32_t mac_of(v4u v) {
    std::uint32_t z = 0, s = 0, o = 0;
    for (int i = 0; i < 4; ++i) {
        const std::uint32_t bit = 8u >> i;
        if (!(v[i] & 0x7F800000u)) z |= bit;
        if (v[i] & 0x80000000u) s |= bit;
        if ((v[i] & 0x7FFFFFFFu) == 0x7FFFFFFFu) o |= bit;
    }
    return z | (s << 4) | (o << 12);
}

struct Mode {
    bool guard = true;
};

// GS coordinates after FTOI4 (12.4 fixed point): a corner is inside the range when 0 <= x, y < 65536 (sixteenths)
inline bool in_range(v4u xyz) {
    return static_cast<std::int32_t>(xyz[0]) >= 0 && static_cast<std::int32_t>(xyz[0]) < 65536 && static_cast<std::int32_t>(xyz[1]) >= 0 &&
           static_cast<std::int32_t>(xyz[1]) < 65536;
}

// ---- clipping (the guard band's triangles that cannot be drawn as they are)
// A corner of a triangle being cut: its clip-space position, texture coordinates (ITOF12, before the divide by w) and colour.
struct Corner {
    float c[4], t[3], col[4];
};
inline Corner lerp(const Corner &a, const Corner &b, float f) {
    Corner o;
    for (int i = 0; i < 4; ++i) o.c[i] = a.c[i] + (b.c[i] - a.c[i]) * f;
    for (int i = 0; i < 3; ++i) o.t[i] = a.t[i] + (b.t[i] - a.t[i]) * f;
    for (int i = 0; i < 4; ++i) o.col[i] = a.col[i] + (b.col[i] - a.col[i]) * f;
    return o;
}
inline float fl(std::uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

}  // namespace

}  // namespace ps2::hp2
namespace ps2::vu {
void hp2_send_packet(const std::uint8_t *data, std::uint32_t qwords);  // vu_micro.cpp: a GIF packet from a buffer, as an XGKICK sends one
}
namespace ps2::hp2 {
namespace {

// The triangles of the strip that end at the corners in `ends` (corners e-2, e-1, e), cut against w > 0, the near and far planes
// (|z| <= w, the CLIP test's z) and a guard band in x and y that keeps every corner inside the GS's coordinate range after the viewport;
// the pieces are sent as one GIF packet before the strip's own (the microcode's clipper also kicks its pieces before the strip).
// Every piece is three corners, the first two with ADC set, so it draws one triangle whatever primitive the batch's GIF tag sets up
// (a strip or a list) and nothing with the corners before it. False: nothing could be sent (the stage then leaves the batch to the
// compiled program).
bool clip_and_send(std::uint8_t *mem, std::uint32_t E7, std::uint32_t N, v4u tag, const v4u *clipv, v4u s30, v4u s31, const std::uint32_t *ends,
                   std::uint32_t nends) {
    const float sx = fl(s30[0]), sy = fl(s30[1]), ox = fl(s31[0]), oy = fl(s31[1]);
    if (!(sx != 0.0f) || !(sy != 0.0f)) return false;
    // |x / w| <= gx keeps x * sx + ox inside 0 .. 4095 (a sixteenth of a pixel inside, for the FTOI4)
    const float gx = std::fmin(ox, 4095.0f - ox) / std::fabs(sx) * 0.999f, gy = std::fmin(oy, 4095.0f - oy) / std::fabs(sy) * 0.999f;
    if (!(gx > 0.0f) || !(gy > 0.0f)) return false;
    static thread_local std::uint8_t packet[16 * (1 + 3 * 3 * 8 * 128)];
    std::uint32_t verts = 0;
    std::uint8_t *at = packet + 16;
    const auto put = [&](const Corner &k, bool adc) {
        const float q = 1.0f / k.c[3];
        const v4u stq{std::bit_cast<std::uint32_t>(k.t[0] * q), std::bit_cast<std::uint32_t>(k.t[1] * q), std::bit_cast<std::uint32_t>(k.t[2] * q), 0u};
        v4u rgba;
        for (int i = 0; i < 4; ++i) rgba[i] = static_cast<std::uint32_t>(std::lround(std::fmax(0.0f, std::fmin(255.0f, k.col[i]))));
        const v4u scr{std::bit_cast<std::uint32_t>(k.c[0] * q * sx + ox), std::bit_cast<std::uint32_t>(k.c[1] * q * sy + oy),
                      std::bit_cast<std::uint32_t>(k.c[2] * q * fl(s30[2]) + fl(s31[2])), 0u};
        v4u xyz = vuops::vftoi(scr, 4);
        xyz[3] = adc ? 0x8000u : 0u;
        st(at, stq), st(at + 16, rgba), st(at + 32, xyz);
        at += 48;
        ++verts;
    };
    for (std::uint32_t e = 0; e < nends; ++e) {
        Corner poly[16], next[16];
        int n = 3;
        for (int i = 0; i < 3; ++i) {
            const std::uint32_t v = ends[e] - 2u + static_cast<std::uint32_t>(i);
            Corner &k = poly[i];
            for (int j = 0; j < 4; ++j) k.c[j] = fl(clipv[v][j]);
            const v4u tf = vuops::vitof(ld(qw(mem, E7 + 1u + N + v)), 12);
            for (int j = 0; j < 3; ++j) k.t[j] = fl(tf[j]);
            const v4u col = ld(qw(mem, E7 + 1u + 2u * N + v));
            // the GS takes the low byte of each word (HP2 keeps colours as floats 98304 + c, whose low mantissa bits are c)
            for (int j = 0; j < 4; ++j) k.col[j] = static_cast<float>(col[j] & 0xFFu);
        }
        // the planes as distances that must stay >= 0: w - eps, z + w, w - z, gx w -+ x, gy w -+ y
        const auto dist = [&](const Corner &k, int p) {
            switch (p) {
            case 0: return k.c[3] - 1e-5f;
            case 1: return k.c[2] + k.c[3];
            case 2: return k.c[3] - k.c[2];
            case 3: return gx * k.c[3] - k.c[0];
            case 4: return gx * k.c[3] + k.c[0];
            case 5: return gy * k.c[3] - k.c[1];
            default: return gy * k.c[3] + k.c[1];
            }
        };
        for (int p = 0; p < 7 && n >= 3; ++p) {
            int m = 0;
            for (int i = 0; i < n; ++i) {
                const Corner &a = poly[i], &b = poly[(i + 1) % n];
                const float da = dist(a, p), db = dist(b, p);
                if (da >= 0.0f && m < 16) next[m++] = a;
                if ((da >= 0.0f) != (db >= 0.0f) && m < 16) next[m++] = lerp(a, b, da / (da - db));
            }
            n = m;
            for (int i = 0; i < n; ++i) poly[i] = next[i];
        }
        for (int i = 1; i + 1 < n; ++i) put(poly[0], true), put(poly[i], true), put(poly[i + 1], false);
    }
    if (!verts) return true;
    // the batch's GIF tag with NLOOP = the corners sent, EOP set
    std::uint64_t lo = (static_cast<std::uint64_t>(tag[1]) << 32) | tag[0];
    lo = (lo & ~0x7FFFull) | verts | 0x8000ull;
    tag[0] = static_cast<std::uint32_t>(lo), tag[1] = static_cast<std::uint32_t>(lo >> 32);
    st(packet, tag);
    static const bool no_pieces = std::getenv("PS2_HP2_NO_PIECES") != nullptr;  // debugging: the cut triangles are left out
    if (!no_pieces) vu::hp2_send_packet(packet, 1u + 3u * verts);
    return true;
}

}  // namespace

// Stage 0x16D8, the plain transform (program key C4C61B5987BD7B14). Registers at the entry (set by the dispatcher and the stages before):
// vi7 = TOP (the GIF tag; positions at TOP+1.., texture coordinates at TOP+1+N.., colours at TOP+1+2N..), vi15 = N (vertices), vi13 =
// the matrix block (rows at +104 .. +107), vi12 = the viewport block (scale +104, offset +105), vi14 = the output buffer, vi2 = this
// stage's table word (vi11 = vi2 + 49 is its clip handler, 0x1860 here). Writes the GIF tag and ST, RGBA, XYZ per vertex at vi14, and
// leaves every register as the microcode does at its JR to 0x0140 (the kick); returns the next program counter, or 0 when the
// compiled program has to run (a triangle the clipper has to cut, a count it does not expect).
std::uint32_t stage_16d8(Vu0Regs &r, std::uint8_t *mem, bool guard) {
    const std::uint32_t N = r.vi[15];
    if (N < 2 || N > 127) return 0;
    if ((r.status & 0xC0u) != 0xC0u) return 0;  // the sticky zero/sign bits not yet set: the compiled program works them out
    const std::uint32_t E7 = r.vi[7], E12 = r.vi[12], E13 = r.vi[13], E14 = r.vi[14], E2 = r.vi[2];
    const v4u m4 = clampv(ld(qw(mem, E13 + 104u))), m5 = clampv(ld(qw(mem, E13 + 105u))), m6 = clampv(ld(qw(mem, E13 + 106u))),
              m7 = clampv(ld(qw(mem, E13 + 107u)));
    const v4u raw4 = ld(qw(mem, E13 + 104u)), raw5 = ld(qw(mem, E13 + 105u)), raw6 = ld(qw(mem, E13 + 106u)), raw7 = ld(qw(mem, E13 + 107u));
    const v4u raw30 = ld(qw(mem, E12 + 104u)), raw31 = ld(qw(mem, E12 + 105u));
    const v4u s30 = clampv(raw30), s31 = clampv(raw31);
    const std::uint32_t one = 0x3F800000u;
    // clip-space position of vertex v: ((m7 * 1 + m4 * x) + m5 * y) + m6 * z, each step rounded (no fused multiply-add, as the programs)
    const auto pos = [&](std::uint32_t v) { return clampv(ld(qw(mem, E7 + 1u + v))); };
    const auto acc_xy = [&](v4u p) { return add(add(mul(m7, splat(one)), mul(m4, splat(p[0]))), mul(m5, splat(p[1]))); };
    const auto transform = [&](v4u p) { return add(acc_xy(p), mul(m6, splat(p[2]))); };

    // One pass over the corners: the clip-space position and the clip flags of every corner the loop clips (0 .. N+1: it transforms two
    // ahead), and for 0 .. N: Q = 1 / w (the fast DIV, vu::fdiv with vf0.w = 1), the viewport position, its FTOI4 and the ST.
    std::uint32_t flags[130];
    v4u clipv[130], q[130], ndc[130], xyz[130], st_v[130];
    const v4u off = mul(s31, splat(one));
    v4u f16w = rv(r.vf[16]);  // ST: x, y, z from the texture coordinates times Q; w keeps being multiplied by each Q
    std::uint32_t w0 = 0;
    for (std::uint32_t v = 0; v <= N + 1; ++v) {
        const v4u c = transform(pos(v));
        clipv[v] = c;
        flags[v] = clip_flags(c, c[3]);
        if (v == 0) w0 = c[3];
        if (v > N) break;
        const std::uint32_t t = vu::fast_operand(c[3]);
        const std::uint32_t qv = (t & fp::kAbs) ? vu::fast_result(std::bit_cast<std::uint32_t>(1.0f / std::bit_cast<float>(t))) : ((one ^ t) & fp::kSign) | fp::kAbs;
        const v4u qs = clampv(splat(qv));
        q[v] = splat(qv);
        // vf15 holds the corner's x, y, z (ADD.xyz vf15 = vf11 + 0: -0 becomes +0) and the first corner's w (ADD.xyz keeps w)
        const v4u z15 = add(c, splat(0u));
        const v4u f15 = v == 0 ? c : v4u{z15[0], z15[1], z15[2], w0};
        const v4u n = mul(f15, qs);
        ndc[v] = n;
        xyz[v] = vuops::vftoi(sel(n, add(mul(n, s30), off), 14u), 4);
        const v4u f = vuops::vitof(ld(qw(mem, E7 + 1u + N + v)), 12);
        f16w = mul(clampv(v4u{f[0], f[1], f[2], f16w[3]}), qs);
        st_v[v] = f16w;
    }
    // the triangle test of iteration k (FCAND 0x3FFFF over the last three CLIPs: corners k-1, k, k+1; before the first, zero)
    const auto tri = [&](std::uint32_t k) { return (k ? flags[k - 1] : 0u) | flags[k] | flags[k + 1]; };
    // which triangles the microcode clips (k < N-1: the last test only marks corner N, which is not drawn). Exact: such a batch is the
    // compiled program's. Guard band: a triangle whose corners can all be drawn is drawn as it is; any other is cut here (clip_triangle)
    // and its last corner is not drawn in the strip (as the microcode marks it, i13 = -1).
    bool cut[130] = {};
    std::uint32_t to_cut[128], ncut = 0;
    for (std::uint32_t k = 0; k + 1 < N; ++k) {
        if (!(tri(k) & 0x3FFFFu)) continue;
        if (!guard) return 0;
        bool safe = true;
        for (std::uint32_t v = k ? k - 1 : k; v <= k + 1 && safe; ++v)
            safe = static_cast<std::int32_t>(clipv[v][3]) > 0 && !(flags[v] & 0x30u) && in_range(xyz[v]);  // w > 0, between near and far, in range
        if (safe) continue;
        cut[k + 1] = true;
        std::uint16_t adc;
        std::memcpy(&adc, qw(mem, E7 + 1u + k + 1u) + 12, 2);
        if (k >= 1 && !(adc & 0x8000u)) to_cut[ncut++] = k + 1;  // a triangle the strip draws (corners k-1, k, k+1)
    }
    if (ncut && !clip_and_send(mem, E7, N, ld(qw(mem, E7)), clipv, s30, s31, to_cut, ncut)) return 0;

    // ---- the outputs (the microcode's order of stores)
    std::uint8_t *out = qw(mem, E14);
    const v4u tag = ld(qw(mem, E7));
    st(out, tag);
    for (std::uint32_t k = 0; k < N; ++k) {
        std::uint8_t *o = qw(mem, E14 + 1u + 3u * k);
        st(o, st_v[k]);
        st(qw(mem, E14 + 2u + 3u * k), ld(qw(mem, E7 + 1u + 2u * N + k)));
        std::uint8_t *x = qw(mem, E14 + 3u + 3u * k);
        std::uint16_t adc;
        std::memcpy(&adc, qw(mem, E7 + 1u + k) + 12, 2);
        // a triangle cut out of the strip (clip_and_send drew its pieces) marks its last corner as not drawn, as the microcode does
        if (cut[k]) adc = 0xFFFFu;
        const std::uint32_t w = adc;
        std::memcpy(x, &xyz[k], 12);
        std::memcpy(x + 12, &w, 4);
    }
    // the GIF tag's words that the prologue stores
    {
        v4u f28 = vuops::vftoi(vuops::vitof(tag, 15), 0);
        const std::uint32_t i13 = (f28[1] & 0xFFFFu) & 8u;
        std::memcpy(qw(mem, 16u), &i13, 4);
        const std::uint32_t tw = tag[3];  // vf9 after two MR32: .y is the tag's w
        std::memcpy(qw(mem, 15u) + 4, &tw, 4);
        wv(r.vf[28], f28);
    }

    // ---- the registers as the microcode leaves them at the JR to 0x0140
    const v4u pn1 = pos(N + 1);
    wv(r.vf[4], raw4), wv(r.vf[5], raw5), wv(r.vf[6], raw6), wv(r.vf[7], raw7);
    wv(r.vf[30], raw30), wv(r.vf[31], raw31);
    wv(r.vf[9], v4u{tag[2], tag[3], tag[0], tag[1]});
    wv(r.vf[10], ld(qw(mem, E7 + 1u + N + 2u)));
    wv(r.vf[11], clipv[N + 1]);
    wv(r.vf[12], ndc[N]);
    wv(r.vf[13], xyz[N - 1]);
    wv(r.vf[14], ld(qw(mem, E7 + 1u + 2u * N + (N - 1u))));
    {
        const v4u z15 = add(clipv[N + 1], splat(0u));
        wv(r.vf[15], v4u{z15[0], z15[1], z15[2], w0});
    }
    wv(r.vf[16], st_v[N]);
    {
        const v4u t = ld(qw(mem, E7 + 1u + N + N + 1u));
        const v4u old17 = rv(r.vf[17]);
        wv(r.vf[17], v4u{t[0], t[1], t[2], old17[3]});
    }
    wv(r.acc, acc_xy(pn1));
    r.q = q[N][0];
    // CLIP history: the last four CLIPs (corners N-2 .. N+1), six bits each
    {
        std::uint32_t c = r.clip;
        for (std::uint32_t v = 0; v <= N + 1; ++v) c = ((c << 6) | flags[v]) & 0xFFFFFFu;
        r.clip = c;
    }
    // MAC and the status as FlagPipeV::finish leaves them: the last result was ADD.xyz vf15 (w counts as 1.0); the last DIV cleared
    // the divide bits (fdiv: 0x20 for a division by zero)
    {
        const v4u z = add(clipv[N + 1], splat(0u));
        const v4u last{z[0], z[1], z[2], one};
        const std::uint32_t mac = mac_of(last);
        std::uint32_t status = r.status & ~0x30u;
        if (!(clipv[N][3] & fp::kAbs) || !(clipv[N][3] & 0x7F800000u)) status |= 0x20u;
        status &= 0xFF0u;
        if (mac & 0x000Fu) status |= 0x1u;
        if (mac & 0x00F0u) status |= 0x2u;
        if (mac & 0xF000u) status |= 0x8u;
        r.mac = mac;
        r.status = status;
    }
    std::uint16_t adc_n;
    std::memcpy(&adc_n, qw(mem, E7 + 1u + N) + 12, 2);
    std::uint16_t m14, m16;
    std::memcpy(&m14, qw(mem, 14u) + 12, 2);
    std::memcpy(&m16, qw(mem, 16u) + 8, 2);
    r.vi[1] = m14;
    r.vi[5] = static_cast<std::uint16_t>(E14 + 1u + 3u * N);
    r.vi[7] = static_cast<std::uint16_t>(E7 + 4u + N);
    r.vi[8] = static_cast<std::uint16_t>(E7 + 2u * N + 3u);
    r.vi[9] = static_cast<std::uint16_t>(E7 + 3u * N + 1u);
    r.vi[10] = 8u;
    r.vi[11] = static_cast<std::uint16_t>(E2 + 49u);
    r.vi[12] = m16;
    r.vi[13] = (tri(N - 1) & 0x3FFFFu) ? 0xFFFFu : adc_n;
    r.vi[15] = static_cast<std::uint16_t>(E7 + 3u * N + 1u);
    return 0x140u;
}

}  // namespace ps2::hp2
