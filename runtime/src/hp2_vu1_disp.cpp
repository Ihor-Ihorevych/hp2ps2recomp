// Harry Potter 2's VU1 microcode run natively (the native dispatcher, plan M4; see hp2_vu1.cpp for the transform stage).
//
// A batch (MSCAL at 0x0010) is the dispatcher, up to four stages it calls through a table, the kick program and the end, and inside the
// stages more programs that jump to each other (the skinning stage's per-vertex jump table, the clipper's pieces): ~8 hops between
// compiled programs per batch at the least and one per vertex in some stages, each one writing registers back and reading them again.
// Here the image's programs are one function (hp2_image_fused.inc, made by scripts/hp2_fuse.py from vurecomp's fast-mode output of the
// programs Harry Potter 2 runs: the same code, the registers kept in locals across the hops), and the transform stage (0x16D8) is the
// native one, called from the loop below. A program the function does not hold, or whose code is not the code it was made from (the
// key of that address in this code generation, checked at every jump into it), goes back to the chain loop.
#define PS2_VU_FAST 1
#define PS2_VU_NOCLAMP 1
#include "ps2/cpu.hpp"
#include "ps2/vu.hpp"
#include "ps2/vu_ops.hpp"

#include <cstdint>
#include <cstring>

namespace ps2::hp2 {

std::uint32_t stage_16d8(Vu0Regs &r, std::uint8_t *mem, bool guard);  // hp2_vu1.cpp
bool key_ok(std::uint32_t pc, std::uint64_t key);                     // vu_micro.cpp

namespace {
using namespace ps2;
using namespace ps2::vuops;
constexpr std::uint32_t kEnded = 0xFFFFFFFFu;  // the program ended (E bit; r.tpc holds where the next MSCAL starts)
#include "hp2_image_fused.inc"
}  // namespace

// `pc` is one of the fused programs, with the code it was made from
bool image_ok(std::uint32_t pc) {
    const std::uint64_t key = image_fused_key(pc);
    return key && key_ok(pc, key);
}

// From `pc` on until the batch ends (-1) or reaches a program the native set does not cover (its address: the chain loop runs it).
// `transform`: the code at 0x16D8 is the code the native transform stage was made for (key checked by the caller).
std::int64_t run_batch(Vu0Regs &r, std::uint8_t *mem, std::uint32_t pc, bool guard, bool transform, bool image) {
    if ((r.status & 0xC0u) != 0xC0u) return pc;  // the fused bodies are the kSticky = false copies (sticky zero/sign bits already set)
    for (int steps = 0; steps < 4096; ++steps) {
        if (pc == 0x16D8u && transform) {
            const std::uint32_t next = stage_16d8(r, mem, guard);
            if (!next) return pc;
            pc = next;
        } else if (image && image_ok(pc)) {
            pc = image_fused(r, mem, pc);
            if (pc == kEnded) return -1;
        } else {
            return pc;
        }
    }
    return pc;
}

}  // namespace ps2::hp2
