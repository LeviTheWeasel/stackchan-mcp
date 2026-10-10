#include "echo_reference.h"

#include <cmath>
#include <cstdlib>

namespace {

// A direction whose interrupt has not fired for this long has stopped.
constexpr int64_t kStoppedAfterUs = 50 * 1000;
// A read that takes this long waited for audio that had not been captured
// yet, which means the driver had nothing queued when it was asked.
constexpr int64_t kWaitedReadUs = 1000;
// Disagreement, in frames, between the offset in use and a fresh measurement
// that is put down to interrupt jitter and left alone.
constexpr int64_t kOffsetToleranceFrames = 60;
// A correction is made only after this many measurements in a row call for
// it, so one late interrupt does not move the reference.
constexpr int kConfirmations = 3;

int64_t RoundedDivide(int64_t numerator, int64_t denominator) {
    return (numerator >= 0 ? numerator + denominator / 2 : numerator - denominator / 2) / denominator;
}

}  // namespace

#ifdef ESP_PLATFORM
void EchoReference::Lock::Enter() { portENTER_CRITICAL_SAFE(&mux_); }
void EchoReference::Lock::Exit() { portEXIT_CRITICAL_SAFE(&mux_); }
#else
void EchoReference::Lock::Enter() { mutex_.lock(); }
void EchoReference::Lock::Exit() { mutex_.unlock(); }
#endif

EchoReference::EchoReference(const Config& config)
    : config_(config),
      ring_mask_(config.ring_frames - 1),
      ring_(config.ring_frames, 0),
      delay_(config.mic_delay_frames, 0) {
}

void EchoReference::OnSent(const int16_t* samples, size_t stride, int64_t now_us) {
    // This interrupt is the only writer of tx_frames_, so it may read it
    // unlocked. The copy is done before the new count is published: readers
    // never look at frames at or past tx_frames_, and Process() keeps clear
    // of the oldest slots, which are the ones this copy lands on.
    const int64_t base = tx_frames_;
    // A starved run ends with the first buffer the driver does not report as
    // overflowed. That is known one interrupt late, here.
    uint32_t ended_run = 0;
    if (tx_overflowed_) {
        tx_overflowed_ = false;
        ++tx_starved_run_;
    } else {
        ended_run = tx_starved_run_;
        tx_starved_run_ = 0;
    }
    if (enabled_) {
        for (int i = 0; i < config_.frames_per_interrupt; ++i) {
            ring_[(base + i) & ring_mask_] = samples != nullptr ? samples[i * stride] : 0;
        }
    }
    Guard guard(lock_);
    tx_frames_ = base + config_.frames_per_interrupt;
    tx_time_us_ = now_us;
    ++stats_.tx_interrupts;
    if (ended_run > 0) {
        const int64_t run_ms = (int64_t)ended_run * config_.frames_per_interrupt * 1000 / config_.sample_rate;
        if (run_ms > 500) {
            ++stats_.idle_runs;
        } else {
            stats_.dropout_buffers += ended_run;
            if (run_ms <= 30) {
                ++stats_.dropouts_30ms;
            } else if (run_ms <= 100) {
                ++stats_.dropouts_100ms;
            } else {
                ++stats_.dropouts_500ms;
            }
        }
    }
}

void EchoReference::OnSendOverflow() {
    tx_overflowed_ = true;
}

void EchoReference::OnReceived(int64_t now_us) {
    Guard guard(lock_);
    rx_frames_ += config_.frames_per_interrupt;
    rx_time_us_ = now_us;
    ++stats_.rx_interrupts;
}

void EchoReference::OnReceiveOverflow() {
    Guard guard(lock_);
    rx_dropped_frames_ += config_.frames_per_interrupt;
    ++stats_.rx_overflows;
}

void EchoReference::Restart() {
    located_ = false;
    mismatch_count_ = 0;
    synced_ = false;
    drift_count_ = 0;
    ClearDelayLine();
    Guard guard(lock_);
    seen_dropped_frames_ = rx_dropped_frames_;
    stats_.located = false;
    stats_.synced = false;
}

void EchoReference::BeforeRead(int64_t now_us) {
    read_started_us_ = now_us;
    Guard guard(lock_);
    rx_frames_before_read_ = rx_frames_;
}

void EchoReference::Process(int16_t* data, size_t frames, size_t channels, int64_t now_us) {
    bool enabled;
    int64_t tx_frames, tx_time_us, rx_frames, rx_time_us, rx_dropped_frames, valid_from;
    int16_t* capture_mic;
    int16_t* capture_reference;
    int16_t* capture_second;
    size_t capture_capacity;
    bool capture_armed;
    bool capture_running;
    size_t capture_count;
    int capture_trigger;
    uint32_t capture_generation;
    {
        Guard guard(lock_);
        enabled = enabled_;
        tx_frames = tx_frames_;
        tx_time_us = tx_time_us_;
        rx_frames = rx_frames_;
        rx_time_us = rx_time_us_;
        rx_dropped_frames = rx_dropped_frames_;
        valid_from = valid_from_;
        capture_mic = capture_mic_;
        capture_reference = capture_reference_;
        capture_second = capture_second_;
        capture_capacity = capture_capacity_;
        capture_armed = capture_armed_;
        capture_running = capture_running_;
        capture_count = capture_count_;
        capture_trigger = capture_trigger_;
        capture_generation = capture_generation_;
    }

    // Where this chunk sits in the microphone stream, counted in the same
    // frames the capture interrupt counts. Kept up to date whether or not
    // the reference is in use, so that it can be switched on mid-stream.
    const int64_t dropped = rx_dropped_frames - seen_dropped_frames_;
    seen_dropped_frames_ = rx_dropped_frames;
    bool gap = dropped > 0;
    reader_end_ += dropped + (int64_t)frames;
    if (now_us - read_started_us_ >= kWaitedReadUs) {
        // The read had to wait, so every buffer captured before it had
        // already been read: the chunk starts where the interrupt count
        // stood when the read was made.
        const int64_t expected = rx_frames_before_read_ + (int64_t)frames;
        if (!located_) {
            reader_end_ = expected;
            located_ = true;
            mismatch_count_ = 0;
        } else if (reader_end_ != expected) {
            if (++mismatch_count_ >= kConfirmations) {
                reader_end_ = expected;
                mismatch_count_ = 0;
                gap = true;
                Guard guard(lock_);
                ++stats_.relocations;
            }
        } else {
            mismatch_count_ = 0;
        }
    }

    if (!enabled || channels < 2) {
        was_enabled_ = false;
        return;
    }
    if (!was_enabled_) {
        was_enabled_ = true;
        synced_ = false;
        drift_count_ = 0;
        ClearDelayLine();
    }
    if (gap) {
        // Audio was lost between what the delay line holds and what follows.
        // Emptying it costs 12 ms of silence and keeps every frame that comes
        // out paired with the reference of its own moment.
        ClearDelayLine();
    }

    // Tie the two clocks together. At the time of its latest interrupt each
    // direction had moved exactly the counted number of frames, and both
    // move at the same rate, so
    //   speaker position - microphone position
    //     = (tx_frames - rx_frames) + (rx_time - tx_time) * sample_rate
    // at any moment while both are running.
    const bool tx_running = tx_frames > 0 && now_us - tx_time_us < kStoppedAfterUs;
    const bool rx_running = rx_frames > 0 && now_us - rx_time_us < kStoppedAfterUs;
    uint32_t new_syncs = 0;
    uint32_t new_resyncs = 0;
    if (tx_running && rx_running) {
        const int64_t measured = (tx_frames - rx_frames) +
            RoundedDivide((rx_time_us - tx_time_us) * config_.sample_rate, 1000000);
        if (!synced_) {
            offset_ = measured;
            synced_ = true;
            drift_count_ = 0;
            ++new_syncs;
        } else if (std::llabs(measured - offset_) > kOffsetToleranceFrames) {
            if (++drift_count_ >= kConfirmations) {
                offset_ = measured;
                drift_count_ = 0;
                ++new_resyncs;
            }
        } else {
            drift_count_ = 0;
        }
    } else {
        // With the speaker clock stopped nothing is being played, and the
        // old offset says nothing about where it will restart.
        synced_ = false;
    }

    // The microphone frame leaving the delay line was captured
    // mic_delay_frames before the one going in. Its reference is the speaker
    // frame of that same moment, taken lead_frames early.
    //
    // That frame has always been reported by the time it is wanted: the
    // newest frame read is at most the capture count, the speaker count
    // trails the true speaker position by less than one interrupt period,
    // and mic_delay_frames covers that period plus the lead plus jitter.
    const bool usable = synced_ && located_;
    const int64_t first_wanted = reader_end_ - (int64_t)frames - config_.mic_delay_frames +
        offset_ + config_.lead_frames;
    const int64_t oldest_kept = tx_frames - (config_.ring_frames - 2 * config_.frames_per_interrupt);
    uint64_t late = 0;
    uint64_t stale = 0;
    uint64_t silent = 0;
    for (size_t i = 0; i < frames; ++i) {
        int16_t* frame = data + i * channels;
        const int16_t second = frame[1];
        const int16_t mic = delay_[delay_pos_];
        delay_[delay_pos_] = frame[0];
        delay_pos_ = (delay_pos_ + 1) % delay_.size();

        int16_t reference = 0;
        if (!usable) {
            ++silent;
        } else {
            const int64_t wanted = first_wanted + (int64_t)i;
            if (wanted >= tx_frames) {
                ++late;
            } else if (wanted < oldest_kept || wanted < valid_from) {
                ++stale;
            } else {
                reference = ring_[wanted & ring_mask_];
            }
        }
        frame[0] = mic;
        frame[1] = reference;

        if (capture_armed && capture_count < capture_capacity) {
            if (!capture_running && std::abs((int)reference) >= capture_trigger) {
                capture_running = true;
            }
            if (capture_running) {
                capture_mic[capture_count] = mic;
                capture_reference[capture_count] = reference;
                capture_second[capture_count] = second;
                ++capture_count;
            }
        }
    }

    Guard guard(lock_);
    stats_.synced = synced_;
    stats_.located = located_;
    stats_.offset_frames = (int)offset_;
    stats_.syncs += new_syncs;
    stats_.resyncs += new_resyncs;
    stats_.gaps += gap ? 1 : 0;
    stats_.frames += frames;
    stats_.late_frames += late;
    stats_.stale_frames += stale;
    stats_.silent_frames += silent;
    if (capture_armed_ && capture_generation_ == capture_generation) {
        // Still the recording this chunk was started for.
        capture_running_ = capture_running;
        capture_count_ = capture_count;
    }
}

void EchoReference::SetEnabled(bool enabled) {
    Guard guard(lock_);
    if (enabled && !enabled_) {
        valid_from_ = tx_frames_;
    }
    enabled_ = enabled;
    stats_.enabled = enabled;
    if (!enabled) {
        stats_.synced = false;
    }
}

bool EchoReference::enabled() {
    Guard guard(lock_);
    return enabled_;
}

EchoReference::Stats EchoReference::GetStats() {
    Guard guard(lock_);
    return stats_;
}

void EchoReference::ArmCapture(int16_t* mic, int16_t* reference, int16_t* second_channel,
                               size_t capacity, int trigger_level) {
    Guard guard(lock_);
    capture_mic_ = mic;
    capture_reference_ = reference;
    capture_second_ = second_channel;
    capture_capacity_ = capacity;
    capture_count_ = 0;
    capture_trigger_ = trigger_level;
    capture_running_ = false;
    ++capture_generation_;
    capture_armed_ = mic != nullptr && reference != nullptr && second_channel != nullptr && capacity > 0;
}

void EchoReference::DisarmCapture() {
    Guard guard(lock_);
    capture_armed_ = false;
    capture_running_ = false;
}

size_t EchoReference::CapturedFrames() {
    Guard guard(lock_);
    return capture_count_;
}

bool EchoReference::CaptureArmed() {
    Guard guard(lock_);
    return capture_armed_;
}

void EchoReference::ClearDelayLine() {
    for (auto& sample : delay_) {
        sample = 0;
    }
    delay_pos_ = 0;
}

EchoAnalysis AnalyzeEcho(const int16_t* mic, const int16_t* reference, size_t frames,
                         int min_lag, int max_lag, int lag_step) {
    EchoAnalysis result;
    if (mic == nullptr || reference == nullptr || frames == 0 || lag_step <= 0) {
        return result;
    }
    int64_t mic_energy = 0;
    int64_t reference_energy = 0;
    for (size_t i = 0; i < frames; ++i) {
        mic_energy += (int64_t)mic[i] * mic[i];
        reference_energy += (int64_t)reference[i] * reference[i];
    }
    result.rms_mic = (int)std::lround(std::sqrt((double)mic_energy / frames));
    result.rms_reference = (int)std::lround(std::sqrt((double)reference_energy / frames));
    if (mic_energy == 0 || reference_energy == 0) {
        return result;
    }

    int64_t best = 0;
    for (int lag = min_lag; lag <= max_lag; lag += lag_step) {
        // mic[i] against reference[i - lag], over the frames both have.
        const int64_t begin = lag > 0 ? lag : 0;
        const int64_t end = lag < 0 ? (int64_t)frames + lag : (int64_t)frames;
        int64_t sum = 0;
        for (int64_t i = begin; i < end; ++i) {
            sum += (int64_t)mic[i] * reference[i - lag];
        }
        if (std::llabs(sum) > std::llabs(best)) {
            best = sum;
            result.best_lag = lag;
        }
    }
    const double scale = std::sqrt((double)mic_energy) * std::sqrt((double)reference_energy);
    result.correlation_permille = (int)std::lround(1000.0 * (double)best / scale);
    return result;
}
