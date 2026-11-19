// Stand-in for runtime/src/prov.cpp (the provenance counters), for the tests that build vu_micro.cpp alone: the VU code only reads the switch
// and reports a program start.
#include <cstdint>

namespace ps2::prov {
bool g_on = false;
bool g_writers = false;
void vu_run(std::uint32_t, std::uint64_t) {}
}  // namespace ps2::prov
