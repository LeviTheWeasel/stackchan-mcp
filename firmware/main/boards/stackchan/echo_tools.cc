#include "echo_tools.h"

#include "application.h"
#include "cores3_audio_codec.h"
#include "echo_reference.h"
#include "mcp_server.h"

#include <cJSON.h>
#include <esp_heap_caps.h>
#include <esp_log.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include <atomic>
#include <cmath>
#include <string>

#define TAG "EchoTools"

namespace {

// One third of a second of playback: long enough for a correlation peak to
// stand out, short enough to analyse without holding anything up.
constexpr size_t kProbeFrames = 8000;
// The recording starts when the reference first gets this loud, so it never
// consists of the silence before a reply.
constexpr int kProbeTriggerLevel = 300;

enum ProbeState { kProbeIdle, kProbeArmed, kProbeAnalysing, kProbeDone };

struct Probe {
    CoreS3AudioCodec* codec = nullptr;
    int sample_rate = 24000;
    int mic_delay_frames = 0;
    int16_t* mic = nullptr;
    int16_t* reference = nullptr;
    int16_t* second = nullptr;
    std::atomic<int> state{kProbeIdle};
    EchoAnalysis mic_vs_reference;
    EchoAnalysis second_vs_reference;
    EchoAnalysis second_vs_mic;
    int rms_second = 0;
};

Probe probe;
bool echo_cancellation_on = false;

double Rms(uint64_t energy, uint32_t samples) {
    return samples == 0 ? 0.0 : std::sqrt((double)energy / samples);
}

void AddDecibels(cJSON* object, const char* name, double numerator, double denominator) {
    if (numerator > 0 && denominator > 0) {
        cJSON_AddNumberToObject(object, name, std::round(200.0 * std::log10(numerator / denominator)) / 10.0);
    } else {
        cJSON_AddNullToObject(object, name);
    }
}

void AddAnalysis(cJSON* parent, const char* name, const EchoAnalysis& analysis, int lag_shift, int sample_rate) {
    cJSON* item = cJSON_AddObjectToObject(parent, name);
    const int lag = analysis.best_lag + lag_shift;
    cJSON_AddNumberToObject(item, "lag_frames", lag);
    cJSON_AddNumberToObject(item, "lag_ms", std::round(10000.0 * lag / sample_rate) / 10.0);
    cJSON_AddNumberToObject(item, "correlation", analysis.correlation_permille / 1000.0);
}

// Runs off the main task: the three correlations take a few hundred
// milliseconds, and the main task is what sends microphone audio upstream.
void AnalyseProbe(void*) {
    // The microphone was held back by mic_delay_frames before it was
    // recorded; the hardware's second channel was not. Lags found against
    // the second channel are shifted by that much when reported, so all
    // three read on the microphone's clock.
    probe.mic_vs_reference = AnalyzeEcho(probe.mic, probe.reference, kProbeFrames, -480, 960, 2);
    probe.second_vs_reference = AnalyzeEcho(probe.second, probe.reference, kProbeFrames, -1200, 480, 2);
    probe.second_vs_mic = AnalyzeEcho(probe.second, probe.mic, kProbeFrames, -960, 480, 2);
    uint64_t energy = 0;
    for (size_t i = 0; i < kProbeFrames; ++i) {
        energy += (uint64_t)((int32_t)probe.second[i] * probe.second[i]);
    }
    probe.rms_second = (int)std::lround(Rms(energy, kProbeFrames));
    probe.state.store(kProbeDone, std::memory_order_release);
    vTaskDelete(NULL);
}

bool AllocateProbe() {
    if (probe.mic != nullptr) {
        return true;
    }
    const size_t bytes = kProbeFrames * sizeof(int16_t);
    probe.mic = (int16_t*)heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM);
    probe.reference = (int16_t*)heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM);
    probe.second = (int16_t*)heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM);
    if (probe.mic == nullptr || probe.reference == nullptr || probe.second == nullptr) {
        heap_caps_free(probe.mic);
        heap_caps_free(probe.reference);
        heap_caps_free(probe.second);
        probe.mic = probe.reference = probe.second = nullptr;
        return false;
    }
    return true;
}

}  // namespace

void RegisterEchoTools(McpServer& mcp_server, CoreS3AudioCodec* codec) {
    probe.codec = codec;
    probe.sample_rate = codec->input_sample_rate();

    mcp_server.AddTool(
        "self.audio.set_echo_cancellation",
        "Switch device-side echo cancellation on or off. On, the microphone "
        "signal has the robot's own speech removed before it is sent, using "
        "the samples given to the speaker as reference, and the device "
        "accepts realtime listening (microphone live while it speaks). Off "
        "is the state after every boot. Takes effect at once; not persisted.",
        PropertyList({Property("enabled", kPropertyTypeBoolean)}),
        [codec](const PropertyList& properties) -> ReturnValue {
            const bool enable = properties["enabled"].value<bool>();
            auto& app = Application::GetInstance();
            bool ok = true;
            const char* reason = nullptr;
            if (enable) {
                // Reference before canceller. A canceller running on the
                // hardware's second channel is what silenced the microphone.
                if (!codec->SetSoftwareReference(true)) {
                    ok = false;
                    reason = "software echo reference unavailable";
                } else if (!app.SetDeviceAec(true)) {
                    codec->SetSoftwareReference(false);
                    ok = false;
                    reason = "firmware built without switchable echo cancellation";
                }
            } else {
                app.SetDeviceAec(false);
                codec->SetSoftwareReference(false);
            }
            echo_cancellation_on = enable && ok;

            cJSON* root = cJSON_CreateObject();
            cJSON_AddBoolToObject(root, "ok", ok);
            cJSON_AddBoolToObject(root, "enabled", echo_cancellation_on);
            if (reason != nullptr) {
                cJSON_AddStringToObject(root, "reason", reason);
            }
            cJSON_AddStringToObject(root, "persistence", "until reboot");
            return root;
        });

    mcp_server.AddTool(
        "self.audio.get_echo_stats",
        "Report the state of echo cancellation: whether the software "
        "reference is tied to the microphone clock, and the sound level "
        "before and after the voice front end, separately for moments when "
        "the speaker was playing and when it was quiet. With nobody talking, "
        "echo_reduction_db is how much of its own voice the robot removes. "
        "Levels accumulate while the device listens; reset=true clears them "
        "after reading.",
        PropertyList({Property("reset", kPropertyTypeBoolean, false)}),
        [codec](const PropertyList& properties) -> ReturnValue {
            const bool reset = properties["reset"].value<bool>();
            const auto stats = codec->echo_reference().GetStats();
            const auto levels = Application::GetInstance().GetAudioService().GetEchoLevels(reset);

            cJSON* root = cJSON_CreateObject();
            cJSON_AddBoolToObject(root, "echo_cancellation", echo_cancellation_on);

            cJSON* reference = cJSON_AddObjectToObject(root, "reference");
            cJSON_AddBoolToObject(reference, "enabled", stats.enabled);
            cJSON_AddBoolToObject(reference, "synced", stats.synced);
            cJSON_AddBoolToObject(reference, "located", stats.located);
            cJSON_AddNumberToObject(reference, "offset_frames", stats.offset_frames);
            cJSON_AddNumberToObject(reference, "syncs", stats.syncs);
            cJSON_AddNumberToObject(reference, "resyncs", stats.resyncs);
            cJSON_AddNumberToObject(reference, "relocations", stats.relocations);
            cJSON_AddNumberToObject(reference, "gaps", stats.gaps);
            cJSON_AddNumberToObject(reference, "tx_interrupts", stats.tx_interrupts);
            cJSON_AddNumberToObject(reference, "rx_interrupts", stats.rx_interrupts);
            cJSON_AddNumberToObject(reference, "rx_overflows", stats.rx_overflows);
            cJSON_AddNumberToObject(reference, "frames", (double)stats.frames);
            cJSON_AddNumberToObject(reference, "late_frames", (double)stats.late_frames);
            cJSON_AddNumberToObject(reference, "stale_frames", (double)stats.stale_frames);
            cJSON_AddNumberToObject(reference, "silent_frames", (double)stats.silent_frames);

            const double mic_playing = Rms(levels.mic_energy_playing, levels.input_samples_playing);
            const double reference_playing = Rms(levels.reference_energy_playing, levels.input_samples_playing);
            const double output_playing = Rms(levels.output_energy_playing, levels.output_samples_playing);
            const double mic_quiet = Rms(levels.mic_energy_quiet, levels.input_samples_quiet);
            const double output_quiet = Rms(levels.output_energy_quiet, levels.output_samples_quiet);

            cJSON* playing = cJSON_AddObjectToObject(root, "while_playing");
            cJSON_AddNumberToObject(playing, "seconds", std::round(levels.input_samples_playing / 1600.0) / 10.0);
            cJSON_AddNumberToObject(playing, "mic_rms", std::round(mic_playing));
            cJSON_AddNumberToObject(playing, "reference_rms", std::round(reference_playing));
            cJSON_AddNumberToObject(playing, "output_rms", std::round(output_playing));
            cJSON* quiet = cJSON_AddObjectToObject(root, "while_quiet");
            cJSON_AddNumberToObject(quiet, "seconds", std::round(levels.input_samples_quiet / 1600.0) / 10.0);
            cJSON_AddNumberToObject(quiet, "mic_rms", std::round(mic_quiet));
            cJSON_AddNumberToObject(quiet, "output_rms", std::round(output_quiet));

            // What the front end removed while the speaker played, and what
            // it does to a quiet room anyway (noise suppression). Only the
            // difference between the two is the canceller's doing.
            AddDecibels(root, "echo_reduction_db", mic_playing, output_playing);
            AddDecibels(root, "quiet_reduction_db", mic_quiet, output_quiet);
            cJSON_AddBoolToObject(root, "reset", reset);
            return root;
        });

    mcp_server.AddTool(
        "self.audio.echo_probe",
        "Record a third of a second of what the echo canceller is given "
        "while the robot speaks, and report how the microphone relates to "
        "the reference. action=\"arm\" starts waiting for playback (echo "
        "cancellation must be on and the device listening); "
        "action=\"result\" returns the state or the finished analysis. "
        "mic_vs_reference.lag_ms should be a few milliseconds and positive: "
        "the microphone hears a sample that long after the reference shows "
        "it. second_channel describes what the hardware puts on capture "
        "channel 1, which the software reference replaces.",
        PropertyList({Property("action", kPropertyTypeString, std::string("result"))}),
        [codec](const PropertyList& properties) -> ReturnValue {
            const std::string action = properties["action"].value<std::string>();
            cJSON* root = cJSON_CreateObject();
            int state = probe.state.load(std::memory_order_acquire);

            if (action == "arm") {
                if (state == kProbeAnalysing) {
                    cJSON_AddBoolToObject(root, "ok", false);
                    cJSON_AddStringToObject(root, "state", "analysing");
                    return root;
                }
                if (!AllocateProbe()) {
                    cJSON_AddBoolToObject(root, "ok", false);
                    cJSON_AddStringToObject(root, "reason", "no memory for the recording");
                    return root;
                }
                probe.mic_delay_frames = codec->echo_reference().config().mic_delay_frames;
                codec->echo_reference().ArmCapture(probe.mic, probe.reference, probe.second,
                    kProbeFrames, kProbeTriggerLevel);
                probe.state.store(kProbeArmed, std::memory_order_release);
                cJSON_AddBoolToObject(root, "ok", true);
                cJSON_AddStringToObject(root, "state", "armed");
                cJSON_AddBoolToObject(root, "reference_enabled", codec->echo_reference().enabled());
                return root;
            }

            if (state == kProbeArmed) {
                const size_t captured = codec->echo_reference().CapturedFrames();
                if (captured < kProbeFrames) {
                    cJSON_AddStringToObject(root, "state", captured == 0 ? "waiting for playback" : "recording");
                    cJSON_AddNumberToObject(root, "frames", (double)captured);
                    return root;
                }
                codec->echo_reference().DisarmCapture();
                probe.state.store(kProbeAnalysing, std::memory_order_release);
                if (xTaskCreate(AnalyseProbe, "echo_probe", 4096, nullptr, 1, nullptr) != pdPASS) {
                    probe.state.store(kProbeIdle, std::memory_order_release);
                    cJSON_AddStringToObject(root, "state", "failed");
                    cJSON_AddStringToObject(root, "reason", "could not start the analysis");
                    return root;
                }
                state = kProbeAnalysing;
            }

            if (state == kProbeAnalysing) {
                cJSON_AddStringToObject(root, "state", "analysing");
            } else if (state == kProbeDone) {
                cJSON_AddStringToObject(root, "state", "done");
                cJSON_AddNumberToObject(root, "frames", (double)kProbeFrames);
                cJSON_AddNumberToObject(root, "sample_rate", probe.sample_rate);
                cJSON_AddNumberToObject(root, "mic_rms", probe.mic_vs_reference.rms_mic);
                cJSON_AddNumberToObject(root, "reference_rms", probe.mic_vs_reference.rms_reference);
                cJSON_AddNumberToObject(root, "second_channel_rms", probe.rms_second);
                AddAnalysis(root, "mic_vs_reference", probe.mic_vs_reference, 0, probe.sample_rate);
                AddAnalysis(root, "second_channel_vs_reference", probe.second_vs_reference,
                    probe.mic_delay_frames, probe.sample_rate);
                AddAnalysis(root, "second_channel_vs_mic", probe.second_vs_mic,
                    probe.mic_delay_frames, probe.sample_rate);
            } else {
                cJSON_AddStringToObject(root, "state", "idle");
            }
            return root;
        });

    ESP_LOGI(TAG, "Echo tools registered");
}
