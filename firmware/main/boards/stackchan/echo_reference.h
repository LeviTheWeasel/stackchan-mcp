#ifndef _STACKCHAN_ECHO_REFERENCE_H
#define _STACKCHAN_ECHO_REFERENCE_H

#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef ESP_PLATFORM
#include <freertos/FreeRTOS.h>
#else
#include <mutex>
#endif

/**
 * An echo reference made from the samples the speaker was actually given.
 *
 * Echo cancellation subtracts what the speaker played from what the
 * microphone heard, so it needs a second input carrying the speaker signal.
 * Boards built for it loop the amplifier output back into a spare ADC input.
 * This board's second capture channel is not such a loop, and the one
 * firmware that fed it to the canceller as if it were (2026-10-10) came back
 * with a microphone that heard nothing.
 *
 * The firmware does know the speaker signal, though: it is the PCM it hands
 * to the I2S driver. What it does not know by itself is *when* a given sample
 * leaves, because the driver queues up to 60 ms of audio. So the samples are
 * not taken where they are written. They are taken from the driver's "this
 * DMA buffer has just been sent" interrupt, which is the speaker's own clock.
 *
 * Playback and capture share one I2S port and therefore one bit clock, so the
 * two streams advance frame for frame. Each direction counts its DMA buffers
 * in its interrupt and stamps the time of the latest one. From the two counts
 * and the two timestamps, the frame the speaker was sending while a given
 * microphone frame was being captured follows by arithmetic, to within the
 * jitter of an interrupt (see Process()).
 *
 * The reference has to be known before the echo it explains arrives. The
 * buffer being sent right now is only reported once it has finished, so the
 * microphone is held back by a little over one buffer (12 ms). That is the
 * whole cost of the scheme.
 *
 * Nothing here depends on ESP-IDF except the lock; host_test/ runs it against
 * simulated interrupts.
 */
class EchoReference {
public:
    struct Config {
        int sample_rate = 24000;
        // Frames in one DMA buffer. The same for both directions: it is what
        // makes the two interrupts tick at the same rate.
        int frames_per_interrupt = 240;
        // Speaker history kept, in frames. Power of two.
        int ring_frames = 4096;
        // How long the microphone is held back. Must cover one interrupt
        // period plus `lead_frames` plus interrupt jitter; see Process().
        int mic_delay_frames = 288;
        // How far ahead of the matching speaker frame the reference is taken.
        // The canceller copes with a reference that is early and not with one
        // that is late, so the error is pushed to the safe side.
        int lead_frames = 24;
    };

    struct Stats {
        bool enabled = false;
        bool synced = false;         // speaker and microphone clocks are tied together
        int offset_frames = 0;       // speaker frame index minus microphone frame index
        uint32_t syncs = 0;          // times the tie was (re)established from scratch
        uint32_t resyncs = 0;        // times it was corrected while running
        bool located = false;        // position of the reader in the microphone stream is known
        uint32_t relocations = 0;    // times that position turned out wrong and was corrected
        uint32_t gaps = 0;           // times captured audio was lost before it was read
        uint32_t tx_interrupts = 0;
        uint32_t rx_interrupts = 0;
        uint32_t rx_overflows = 0;
        uint64_t frames = 0;         // frames given a reference
        uint64_t late_frames = 0;    // wanted speaker frame had not been sent yet
        uint64_t stale_frames = 0;   // wanted speaker frame was already overwritten
        uint64_t silent_frames = 0;  // clocks not tied yet or speaker stopped; reference is silence

        // Playback, counted whether or not the reference is enabled. A
        // buffer sent before the player had written anything new into the
        // queue goes out as silence. A run of those that ends with sound
        // again is counted by its length: the short ones fall inside a
        // reply and are heard as stutter, the long ones are the speaker
        // simply not being used.
        uint32_t dropouts_30ms = 0;      // runs of up to 30 ms
        uint32_t dropouts_100ms = 0;     // over 30 ms, up to 100 ms
        uint32_t dropouts_500ms = 0;     // over 100 ms, up to 500 ms
        uint32_t dropout_buffers = 0;    // buffers in all of those runs together
        uint32_t idle_runs = 0;          // longer than 500 ms
    };

    explicit EchoReference(const Config& config);
    const Config& config() const { return config_; }

    // --- interrupt side -------------------------------------------------
    // One DMA buffer has just been sent. `samples` holds
    // frames_per_interrupt frames, the wanted sample every `stride` int16.
    void OnSent(const int16_t* samples, size_t stride, int64_t now_us);
    // The buffer just reported by OnSent() went out with the player behind:
    // everything it had written has been sent. The driver reports this after
    // OnSent() for the same buffer.
    void OnSendOverflow();
    // One DMA buffer has just been filled by the microphones.
    void OnReceived(int64_t now_us);
    // The oldest unread capture buffer was thrown away to make room.
    void OnReceiveOverflow();

    // --- capture side (one thread) --------------------------------------
    // The capture channel was stopped and started: audio it had queued is
    // gone, and where the next read sits in the stream has to be found again.
    void Restart();
    // Call immediately before asking the driver for audio.
    void BeforeRead(int64_t now_us);
    // `data` is `frames` frames of `channels` interleaved samples just read
    // from the driver, channel 0 the microphone. When enabled, channel 0 is
    // replaced by the held-back microphone and channel 1 by the reference.
    void Process(int16_t* data, size_t frames, size_t channels, int64_t now_us);

    // --- any thread -----------------------------------------------------
    void SetEnabled(bool enabled);
    bool enabled();
    Stats GetStats();

    // Record what Process() sees, for AnalyzeEcho(). The three buffers each
    // hold `capacity` samples and stay owned by the caller. Recording starts
    // with the first chunk whose reference reaches `trigger_level`.
    void ArmCapture(int16_t* mic, int16_t* reference, int16_t* second_channel,
                    size_t capacity, int trigger_level);
    void DisarmCapture();
    // Frames recorded so far; equals `capacity` when the recording is done.
    size_t CapturedFrames();
    bool CaptureArmed();

private:
    class Lock {
    public:
        void Enter();
        void Exit();
    private:
#ifdef ESP_PLATFORM
        portMUX_TYPE mux_ = portMUX_INITIALIZER_UNLOCKED;
#else
        std::mutex mutex_;
#endif
    };
    struct Guard {
        explicit Guard(Lock& lock) : lock_(lock) { lock_.Enter(); }
        ~Guard() { lock_.Exit(); }
        Lock& lock_;
    };

    void ClearDelayLine();

    const Config config_;
    const int64_t ring_mask_;
    std::vector<int16_t> ring_;
    std::vector<int16_t> delay_;
    size_t delay_pos_ = 0;

    Lock lock_;
    // Guarded by lock_: written in interrupts, read by the capture thread.
    bool enabled_ = false;
    int64_t tx_frames_ = 0;
    int64_t tx_time_us_ = 0;
    int64_t rx_frames_ = 0;
    int64_t rx_time_us_ = 0;
    int64_t rx_dropped_frames_ = 0;
    int64_t valid_from_ = 0;  // speaker frames before this were not recorded
    Stats stats_;

    // Send interrupt only.
    bool tx_overflowed_ = false;   // OnSendOverflow() came for the previous buffer
    uint32_t tx_starved_run_ = 0;  // consecutive buffers it came for

    // Capture thread only.
    bool was_enabled_ = false;
    bool synced_ = false;      // offset_ is valid
    int64_t offset_ = 0;       // speaker frame index minus microphone frame index
    int drift_count_ = 0;
    bool located_ = false;     // reader_end_ is valid
    int64_t reader_end_ = 0;   // microphone frame index just past the last frame read
    int mismatch_count_ = 0;
    int64_t seen_dropped_frames_ = 0;
    int64_t read_started_us_ = 0;
    int64_t rx_frames_before_read_ = 0;

    // Guarded by lock_ for the pointers; filled by the capture thread.
    int16_t* capture_mic_ = nullptr;
    int16_t* capture_reference_ = nullptr;
    int16_t* capture_second_ = nullptr;
    size_t capture_capacity_ = 0;
    size_t capture_count_ = 0;
    int capture_trigger_ = 0;
    bool capture_armed_ = false;
    bool capture_running_ = false;
    uint32_t capture_generation_ = 0;  // bumped by every ArmCapture()
};

/**
 * How much of `mic` is explained by `reference`, and at what delay.
 *
 * Slides `reference` against `mic` over lags [min_lag, max_lag] in steps of
 * `lag_step` and reports the lag with the strongest correlation. A positive
 * lag means the microphone hears a sample that many frames after the
 * reference shows it, which is what a usable reference looks like.
 */
struct EchoAnalysis {
    int rms_mic = 0;
    int rms_reference = 0;
    int best_lag = 0;
    // Normalised correlation at best_lag, in thousandths: 1000 is identical
    // up to gain, 0 is unrelated, negative is inverted.
    int correlation_permille = 0;
};

EchoAnalysis AnalyzeEcho(const int16_t* mic, const int16_t* reference, size_t frames,
                         int min_lag, int max_lag, int lag_step);

#endif  // _STACKCHAN_ECHO_REFERENCE_H
