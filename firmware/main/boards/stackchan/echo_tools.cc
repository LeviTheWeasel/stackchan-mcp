#include "echo_tools.h"

#include "application.h"
#include "cores3_audio_codec.h"
#include "echo_reference.h"
#include "mcp_server.h"
#include "settings.h"

#include <cJSON.h>
#include <esp_aec.h>
#include <esp_heap_caps.h>
#include <esp_log.h>
#include <esp_system.h>
#include <esp_timer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <string>
#include <vector>

#define TAG "EchoTools"

namespace {

// A sixth of a second of playback: long enough for a correlation peak to
// stand out. The first version recorded twice as much and ran three
// fine-grained correlations over it, which took ten seconds of a core and
// made the capture loop drop audio for as long as it ran.
constexpr size_t kProbeFrames = 4000;
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

// Runs off the main task, which is what sends microphone audio upstream.
// Each correlation is a coarse pass to find the neighbourhood and a fine
// pass around it, and the task yields between them.
EchoAnalysis CoarseToFine(const int16_t* a, const int16_t* b, int min_lag, int max_lag) {
    EchoAnalysis coarse = AnalyzeEcho(a, b, kProbeFrames, min_lag, max_lag, 6);
    vTaskDelay(1);
    return AnalyzeEcho(a, b, kProbeFrames, coarse.best_lag - 6, coarse.best_lag + 6, 1);
}

void AnalyseProbe(void*) {
    // The microphone was held back by mic_delay_frames before it was
    // recorded; the hardware's second channel was not. The lag found
    // against the second channel is shifted by that much when reported, so
    // both read on the microphone's clock.
    probe.mic_vs_reference = CoarseToFine(probe.mic, probe.reference, -480, 960);
    vTaskDelay(1);
    probe.second_vs_mic = CoarseToFine(probe.second, probe.mic, -600, 120);
    uint64_t energy = 0;
    for (size_t i = 0; i < kProbeFrames; ++i) {
        energy += (uint64_t)((int32_t)probe.second[i] * probe.second[i]);
    }
    probe.rms_second = (int)std::lround(Rms(energy, kProbeFrames));
    probe.state.store(kProbeDone, std::memory_order_release);
    vTaskDelete(NULL);
}

// --- options that are fixed when the front end is created ------------------

struct AecModeName {
    const char* name;
    int value;
};
const AecModeName kAecModes[] = {
    {"sr_low_cost", AEC_MODE_SR_LOW_COST},
    {"sr_high_perf", AEC_MODE_SR_HIGH_PERF},
    {"voip_low_cost", AEC_MODE_VOIP_LOW_COST},
    {"voip_high_perf", AEC_MODE_VOIP_HIGH_PERF},
};

const char* AecModeToName(int value) {
    for (const auto& mode : kAecModes) {
        if (mode.value == value) return mode.name;
    }
    return "unknown";
}

// What is stored, which is what the next boot will use. Not necessarily what
// this boot is running: see "applied_at_boot" in the tool's reply.
void AddStoredOptions(cJSON* parent) {
    Settings settings("echo", false);
    cJSON* options = cJSON_AddObjectToObject(parent, "options");
    cJSON_AddStringToObject(options, "aec_mode",
        AecModeToName(settings.GetInt("aec_mode", AEC_MODE_VOIP_HIGH_PERF)));
    cJSON_AddNumberToObject(options, "filter_length", settings.GetInt("filter_len", 0));
    cJSON_AddBoolToObject(options, "noise_suppression", settings.GetBool("ns", true));
    cJSON_AddNumberToObject(options, "input_core", settings.GetInt("in_core", 0));
    cJSON_AddNumberToObject(options, "afe_core", settings.GetInt("afe_core", -1));
}

void RebootSoon(void*) {
    // Long enough for the tool's reply to reach the gateway.
    vTaskDelay(pdMS_TO_TICKS(1500));
    // On the main task, like every other use of the protocol it tears down.
    Application::GetInstance().Schedule([]() { Application::GetInstance().Reboot(); });
    vTaskDelete(NULL);
}

// --- where the processor time goes -----------------------------------------

struct TaskSample {
    UBaseType_t number;
    uint32_t run_time;
};
std::vector<TaskSample> previous_tasks;
int64_t previous_tasks_us = 0;

// Share of one core each task used since the previous call, busiest first.
// The two idle tasks are the headroom: a core whose idle task is near zero
// has nothing left to give.
void AddCpuTable(cJSON* parent) {
    const UBaseType_t capacity = uxTaskGetNumberOfTasks() + 8;
    std::vector<TaskStatus_t> tasks(capacity);
    uint32_t total = 0;
    const UBaseType_t count = uxTaskGetSystemState(tasks.data(), capacity, &total);
    const int64_t now_us = esp_timer_get_time();
    const int64_t elapsed_us = now_us - previous_tasks_us;

    cJSON* cpu = cJSON_AddObjectToObject(parent, "cpu");
    cJSON_AddNumberToObject(cpu, "seconds", std::round(elapsed_us / 100000.0) / 10.0);

    struct Row {
        const TaskStatus_t* task;
        double percent;
    };
    std::vector<Row> rows;
    std::vector<TaskSample> current;
    current.reserve(count);
    for (UBaseType_t i = 0; i < count; ++i) {
        current.push_back({tasks[i].xTaskNumber, tasks[i].ulRunTimeCounter});
        if (previous_tasks_us == 0 || elapsed_us <= 0) continue;
        for (const auto& before : previous_tasks) {
            if (before.number != tasks[i].xTaskNumber) continue;
            const uint32_t used = tasks[i].ulRunTimeCounter - before.run_time;
            rows.push_back({&tasks[i], 100.0 * used / (double)elapsed_us});
            break;
        }
    }
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.percent > b.percent; });

    cJSON* list = cJSON_AddArrayToObject(cpu, "tasks");
    for (const auto& row : rows) {
        const bool idle = std::string(row.task->pcTaskName).rfind("IDLE", 0) == 0;
        if (row.percent < 1.0 && !idle) continue;
        cJSON* item = cJSON_CreateObject();
        cJSON_AddStringToObject(item, "name", row.task->pcTaskName);
        cJSON_AddNumberToObject(item, "percent", std::round(row.percent * 10.0) / 10.0);
        cJSON_AddNumberToObject(item, "priority", row.task->uxCurrentPriority);
        const BaseType_t core = xTaskGetCoreID(row.task->xHandle);
        cJSON_AddNumberToObject(item, "core", core == tskNO_AFFINITY ? -1 : core);
        cJSON_AddItemToArray(list, item);
    }
    previous_tasks = std::move(current);
    previous_tasks_us = now_us;
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

            // Whether the capture loop keeps up: it has 10 ms per loop, and
            // the I2S driver buffers 50 ms before it starts discarding.
            const auto load = Application::GetInstance().GetAudioService().GetInputLoad(reset);
            cJSON* input = cJSON_AddObjectToObject(root, "capture_loop");
            cJSON_AddNumberToObject(input, "loops", load.loops);
            cJSON_AddNumberToObject(input, "mean_wait_ms",
                load.loops ? std::round(load.wait_us / (double)load.loops / 10.0) / 100.0 : 0);
            cJSON_AddNumberToObject(input, "mean_work_ms",
                load.loops ? std::round(load.work_us / (double)load.loops / 10.0) / 100.0 : 0);
            cJSON_AddNumberToObject(input, "max_work_ms", std::round(load.max_work_us / 100.0) / 10.0);
            cJSON_AddNumberToObject(input, "slow_loops", load.slow_loops);
            cJSON_AddNumberToObject(input, "stalled_loops", load.stalled_loops);

            // Whether the speaker was kept fed. The dropouts are stretches
            // of silence that ended with sound again; the short ones fall
            // inside a reply. Counted since boot, in manual mode as well.
            const auto feed = Application::GetInstance().GetAudioService().GetOutputFeed(reset);
            cJSON* playback = cJSON_AddObjectToObject(root, "playback");
            cJSON_AddNumberToObject(playback, "dropouts_upto_30ms", stats.dropouts_30ms);
            cJSON_AddNumberToObject(playback, "dropouts_upto_100ms", stats.dropouts_100ms);
            cJSON_AddNumberToObject(playback, "dropouts_upto_500ms", stats.dropouts_500ms);
            cJSON_AddNumberToObject(playback, "dropout_ms",
                (double)stats.dropout_buffers * codec->echo_reference().config().frames_per_interrupt * 1000 /
                    codec->echo_reference().config().sample_rate);
            cJSON_AddNumberToObject(playback, "idle_runs", stats.idle_runs);
            cJSON_AddNumberToObject(playback, "packets", feed.packets);
            cJSON_AddNumberToObject(playback, "late_packets", feed.late_packets);
            cJSON_AddNumberToObject(playback, "max_packet_gap_ms", feed.max_gap_ms);
            cJSON_AddNumberToObject(playback, "max_packets_waiting", feed.max_waiting);

            AddCpuTable(root);
            AddStoredOptions(root);
            cJSON_AddBoolToObject(root, "reset", reset);
            return root;
        });

    mcp_server.AddTool(
        "self.audio.set_echo_options",
        "Store how the voice front end is built: the echo canceller's mode "
        "(aec_mode: sr_low_cost, sr_high_perf, voip_low_cost, "
        "voip_high_perf), its filter length in 16 ms blocks (1-8, 0 for the "
        "library default), noise suppression on or off, the core the capture "
        "task runs on (0, 1, or -1 for either) and the core the front end "
        "prefers (0, 1, or -1 for the library default). Only what is given "
        "changes; reset=true clears everything back to the defaults. Stored "
        "in NVS and applied at the next boot; reboot=true restarts the "
        "device once the reply is sent (avatar sets must be loaded again).",
        PropertyList({
            Property("aec_mode", kPropertyTypeString, std::string()),
            Property("filter_length", kPropertyTypeInteger, -1, -1, 8),
            Property("noise_suppression", kPropertyTypeString, std::string()),
            Property("input_core", kPropertyTypeInteger, -2, -2, 1),
            Property("afe_core", kPropertyTypeInteger, -2, -2, 1),
            Property("reset", kPropertyTypeBoolean, false),
            Property("reboot", kPropertyTypeBoolean, false),
        }),
        [](const PropertyList& properties) -> ReturnValue {
            cJSON* root = cJSON_CreateObject();
            const char* problem = nullptr;
            {
                Settings settings("echo", true);
                if (properties["reset"].value<bool>()) {
                    settings.EraseAll();
                }
                const std::string aec_mode = properties["aec_mode"].value<std::string>();
                if (!aec_mode.empty()) {
                    bool known = false;
                    for (const auto& mode : kAecModes) {
                        if (aec_mode == mode.name) {
                            settings.SetInt("aec_mode", mode.value);
                            known = true;
                        }
                    }
                    if (!known) problem = "unknown aec_mode";
                }
                const int filter_length = properties["filter_length"].value<int>();
                if (filter_length >= 0) {
                    settings.SetInt("filter_len", filter_length);
                }
                // A string because "not given" has to be told apart from false.
                const std::string noise_suppression = properties["noise_suppression"].value<std::string>();
                if (noise_suppression == "on" || noise_suppression == "true") {
                    settings.SetBool("ns", true);
                } else if (noise_suppression == "off" || noise_suppression == "false") {
                    settings.SetBool("ns", false);
                } else if (!noise_suppression.empty()) {
                    problem = "noise_suppression must be on or off";
                }
                const int input_core = properties["input_core"].value<int>();
                if (input_core >= -1) {
                    settings.SetInt("in_core", input_core);
                }
                const int afe_core = properties["afe_core"].value<int>();
                if (afe_core >= -1) {
                    settings.SetInt("afe_core", afe_core);
                }
            }
            cJSON_AddBoolToObject(root, "ok", problem == nullptr);
            if (problem != nullptr) {
                cJSON_AddStringToObject(root, "reason", problem);
            }
            AddStoredOptions(root);
            cJSON_AddStringToObject(root, "takes_effect", "next boot");
            const bool reboot = properties["reboot"].value<bool>() && problem == nullptr;
            cJSON_AddBoolToObject(root, "rebooting", reboot);
            if (reboot) {
                xTaskCreate(RebootSoon, "echo_reboot", 4096, nullptr, 1, nullptr);
            }
            return root;
        });

    mcp_server.AddTool(
        "self.audio.echo_probe",
        "Record a sixth of a second of what the echo canceller is given "
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
                AddAnalysis(root, "second_channel_vs_mic", probe.second_vs_mic,
                    probe.mic_delay_frames, probe.sample_rate);
            } else {
                cJSON_AddStringToObject(root, "state", "idle");
            }
            return root;
        });

    ESP_LOGI(TAG, "Echo tools registered");
}
