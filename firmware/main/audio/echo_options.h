#ifndef ECHO_OPTIONS_H
#define ECHO_OPTIONS_H

#include <esp_aec.h>

// The echo canceller's mode when the "echo" settings hold none. It is what
// the voice front end of a CONFIG_USE_SWITCHABLE_DEVICE_AEC build is created
// with and what self.audio.set_echo_options and get_echo_stats report, and
// it is defined once so that the two cannot drift apart.
//
// That option exists for the StackChan board only (see Kconfig.projbuild),
// and voip_high_perf, which every other build of the front end uses, costs
// more than this board has while it speaks. Measured on a robot on
// 2026-10-10 with each mode stored in the settings: in voip_high_perf the
// capture task took about 90 % of core 0, about 5 % of capture buffers were
// dropped and 15 s of speech played with about 60 holes; in voip_low_cost
// 0.2 % were dropped, there were about 10 holes, and 35 to 40 dB of echo was
// removed.
constexpr aec_mode_t kSwitchableAecDefaultMode = AEC_MODE_VOIP_LOW_COST;

#endif
