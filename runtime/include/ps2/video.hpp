#pragma once
#include <cstdint>

namespace ps2 {
// Display-only options, independent of emulated GS state and save states.
// bit 0: edge antialiasing.
// bit 2: sharp enlargement (cubic plus light contrast-adaptive sharpening), the other two bits are not used with it.
// bit 1: soft enlargement (a smooth blend everywhere) instead of the sharp one (hard pixels, only their borders blend).
void set_video_options(std::uint32_t options);
std::uint32_t video_options();
// How the picture fills the window: 0 original 4:3 (bars at the sides), 1 stretched over the whole window, 2 zoomed (4:3 cropped to fill,
// no bars). Display only, changes live.
void set_display_mode(int mode);
int display_mode();
void set_video_scale(std::uint32_t scale);
std::uint32_t video_scale();
}
