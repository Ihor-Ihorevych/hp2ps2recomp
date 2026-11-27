// Harry Potter 2's VU1 dispatcher run natively (plan M4; see hp2_vu1.cpp): a whole batch without the chain loop's hops.
//
// A batch (MSCAL at 0x0010) is the dispatcher, up to four stages it calls through a table (returning to 0x00D0, 0x00E8, 0x0108), the
// kick program (0x0140) and the end (0x0000): ~8 hops between compiled programs per batch, each one writing registers back and reading
// them again. Here the dispatcher's six programs are one function (hp2_disp_fused.inc, made by scripts/hp2_fuse.py from vurecomp's
// fast-mode output), and the stages that have native versions (the transform 0x16D8, the skinning 0x1C28) are called from the loop
// below directly. A stage without one returns the batch to the chain loop at that stage; its return to the dispatcher comes back here.
#define PS2_VU_FAST 1
#define PS2_VU_NOCLAMP 1
#include "ps2/cpu.hpp"
#include "ps2/vu.hpp"
#include "ps2/vu_ops.hpp"

#include <cstdint>
#include <cstring>

namespace ps2::hp2 {

std::uint32_t stage_16d8(Vu0Regs &r, std::uint8_t *mem, bool guard);  // hp2_vu1.cpp
std::uint32_t stage_1c28(Vu0Regs &r, std::uint8_t *mem);              // hp2_vu1_skin.cpp

namespace {
using namespace ps2;
using namespace ps2::vuops;
constexpr std::uint32_t kEnded = 0xFFFFFFFFu;  // the program ended (E bit): r.tpc holds where the next MSCAL... starts
#include "hp2_disp_fused.inc"
}  // namespace

bool is_dispatcher_entry(std::uint32_t pc) { return pc == 0x0000u || pc == 0x0010u || pc == 0x00D0u || pc == 0x00E8u || pc == 0x0108u || pc == 0x0140u; }

// From `pc` (a dispatcher entry) on until the batch ends (-1) or reaches a program the native set does not cover (its address: the chain
// loop runs it). `transform` / `skin`: those stages' code is the code the native versions were made for (keys checked by the caller).
std::int64_t run_batch(Vu0Regs &r, std::uint8_t *mem, std::uint32_t pc, bool guard, bool transform, bool skin) {
    for (int steps = 0; steps < 64; ++steps) {
        if (is_dispatcher_entry(pc)) {
            pc = disp_fused(r, mem, pc);
            if (pc == kEnded) return -1;
        } else if (pc == 0x16D8u && transform) {
            const std::uint32_t next = stage_16d8(r, mem, guard);
            if (!next) return pc;
            pc = next;
        } else if (pc == 0x1C28u && skin) {
            const std::uint32_t next = stage_1c28(r, mem);
            if (!next) return pc;
            pc = next;
        } else {
            return pc;
        }
    }
    return pc;
}

}  // namespace ps2::hp2
