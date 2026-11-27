// Harry Potter 2's VU1 skinning stage (entry 0x1C28) as one function (the native dispatcher, plan M4; see hp2_vu1.cpp).
//
// The stage blends up to four bone matrices per vertex and moves the vertex and its normal with the blend. In the microcode the vertex
// loop jumps through a table per vertex (JR to 0x1D60 / 0x1E10 / 0x1E88 / 0x1EE0 by the number of bones), so every vertex is a hop
// between compiled programs: each hop writes ~28 vector registers back, the chain loop looks the next program up, and that program
// reads and clamps them again and restarts its flag pipeline: ~1 000 - 1 500 cycles a vertex on a PC for a few dozen float operations.
// Here the five programs are one function (hp2_skin_fused.inc, made by scripts/hp2_fuse_skin.py from vurecomp's fast-mode output of
// them): the same code, with the registers kept in locals across the hops between them. Exact as the compiled fast programs are (the
// same code; PS2_HP2_NATIVE=check compares the two). It is built with the fast vector unit's definitions, as the live libraries are.
#define PS2_VU_FAST 1
#define PS2_VU_NOCLAMP 1
#include "ps2/cpu.hpp"
#include "ps2/vu.hpp"
#include "ps2/vu_ops.hpp"

#include <cstdint>
#include <cstring>

namespace ps2::hp2 {

bool key_ok(std::uint32_t pc, std::uint64_t key);  // vu_micro.cpp

namespace {
using namespace ps2;
using namespace ps2::vuops;
#include "hp2_skin_fused.inc"
}  // namespace

// 0: not taken (the compiled programs run); else the program counter the stage leaves for (the dispatcher's return, or 0x01D0)
std::uint32_t stage_1c28(Vu0Regs &r, std::uint8_t *mem) {
    if ((r.status & 0xC0u) != 0xC0u) return 0;  // the fused bodies are the kSticky = false copies (sticky zero/sign bits already set)
    return skin_fused(r, mem);
}

}  // namespace ps2::hp2
