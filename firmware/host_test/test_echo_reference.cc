// EchoReference against simulated I2S interrupts.
//
// The simulated speaker plays a ramp, so every sample names the frame it
// belongs to, and the simulated microphone hears that ramp a fixed number of
// frames later. After Process() each output frame therefore says two things:
// which speaker frame the microphone was hearing, and which speaker frame it
// was given as reference. The reference is right when it is `lead_frames`
// ahead of the speaker frame of the same moment -- and it must never be
// behind what the microphone hears, because a canceller cannot use that.

#include "echo_reference.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <deque>
#include <random>
#include <vector>

namespace {

constexpr int kRate = 24000;
constexpr int kBuffer = 240;                // frames per DMA buffer
constexpr int64_t kPeriodUs = 10000;        // one buffer at 24 kHz
constexpr int kQueue = 5;                   // dma_desc_num - 1, as in the driver
constexpr int kRamp = 20011;                // ramp length; a prime, well above any offset tested
constexpr int kEchoDelay = 30;              // frames from speaker to microphone
constexpr int16_t kSecondChannel = 7777;    // what the hardware puts on channel 1

int16_t RampAt(int64_t frame) {
    return (int16_t)(((frame % kRamp) + kRamp) % kRamp + 1);  // never 0: 0 means "nothing"
}

// Difference between two ramp values, as a number of frames near zero.
int RampDistance(int16_t a, int16_t b) {
    int d = ((int)a - (int)b) % kRamp;
    if (d > kRamp / 2) d -= kRamp;
    if (d < -kRamp / 2) d += kRamp;
    return d;
}

class Rig {
public:
    explicit Rig(int64_t rx_phase_us, int stride = 1)
        : reference_(EchoReference::Config{}), stride_(stride) {
        next_tx_us_ = kPeriodUs;
        next_rx_us_ = kPeriodUs + rx_phase_us;
        rx_origin_us_ = rx_phase_us;
    }

    EchoReference& reference() { return reference_; }

    // Timestamps handed to the interrupt handlers are off by up to this much.
    void SetJitterUs(int jitter_us) { jitter_us_ = jitter_us; }

    void StopSpeaker() {
        speaker_running_ = false;
        stopped_since_us_ = now_us_;
    }
    void StartSpeaker(int64_t at_us) {
        speaker_running_ = true;
        quiet_.push_back({stopped_since_us_, at_us});
        stopped_since_us_ = -1;
        // The frame count carries on; the moment it carries on from is new.
        tx_origin_us_ = at_us - tx_frames_ * 1000000 / kRate;
        next_tx_us_ = at_us + kPeriodUs;
    }

    // The capture channel is closed and reopened: queued audio is lost.
    void RestartCapture(int64_t resume_us) {
        queue_.clear();
        rx_origin_us_ = resume_us - rx_frames_ * 1000000 / kRate;
        next_rx_us_ = resume_us + kPeriodUs;
        reference_.Restart();
    }

    // The next capture interrupt never arrives and its buffer is never queued.
    void LoseNextCaptureInterrupt() { lose_rx_ = true; }

    // The reader does something else until `until_us`.
    void StallReader(int64_t until_us) {
        Advance(until_us);
        now_us_ = until_us;
    }

    struct Frame {
        int16_t mic;
        int16_t reference;
    };

    // Run the capture loop until simulated time `until_us`, returning every
    // frame that came out of Process().
    std::vector<Frame> Run(int64_t until_us) {
        std::vector<Frame> out;
        while (now_us_ < until_us) {
            Advance(now_us_);
            reference_.BeforeRead(now_us_);
            if (queue_.empty()) {
                // Sleep until an interrupt queues a buffer. One that is lost
                // queues nothing, and the read sleeps on through it.
                while (queue_.empty()) {
                    now_us_ = next_rx_us_ + 100;
                    Advance(now_us_);
                }
            } else {
                now_us_ += 30;
            }
            const int64_t first = queue_.front();
            queue_.pop_front();
            std::vector<int16_t> data(kBuffer * 2);
            for (int i = 0; i < kBuffer; ++i) {
                data[2 * i] = Heard(first + i);
                data[2 * i + 1] = kSecondChannel;
            }
            reference_.Process(data.data(), kBuffer, 2, now_us_);
            for (int i = 0; i < kBuffer; ++i) {
                out.push_back({data[2 * i], data[2 * i + 1]});
            }
            now_us_ += 1500;  // the rest of the capture loop
        }
        return out;
    }

    int64_t now_us() const { return now_us_; }

private:
    // What the microphone records in capture frame `frame`: the speaker
    // frame that left kEchoDelay frames earlier, or silence.
    int16_t Heard(int64_t frame) const {
        const int64_t at_us = rx_origin_us_ + frame * 1000000 / kRate;
        const int64_t left_us = at_us - (int64_t)kEchoDelay * 1000000 / kRate;
        if (stopped_since_us_ >= 0 && left_us >= stopped_since_us_) {
            return 0;
        }
        for (const auto& quiet : quiet_) {
            if (left_us >= quiet.first && left_us < quiet.second) return 0;
        }
        const int64_t speaker = (at_us - tx_origin_us_) * kRate / 1000000 - kEchoDelay;
        return speaker < 0 ? 0 : RampAt(speaker);
    }

    int64_t Jitter() {
        if (jitter_us_ == 0) return 0;
        return std::uniform_int_distribution<int>(0, jitter_us_)(random_);
    }

    void Advance(int64_t until_us) {
        while (true) {
            const bool tx_due = speaker_running_ && next_tx_us_ <= until_us;
            const bool rx_due = next_rx_us_ <= until_us;
            if (!tx_due && !rx_due) return;
            if (tx_due && (!rx_due || next_tx_us_ <= next_rx_us_)) {
                std::vector<int16_t> buffer(kBuffer * stride_, 0);
                for (int i = 0; i < kBuffer; ++i) {
                    buffer[i * stride_] = RampAt(tx_frames_ + i);
                }
                reference_.OnSent(buffer.data(), stride_, next_tx_us_ + Jitter());
                tx_frames_ += kBuffer;
                next_tx_us_ += kPeriodUs;
            } else {
                if (lose_rx_) {
                    lose_rx_ = false;
                } else {
                    reference_.OnReceived(next_rx_us_ + Jitter());
                    if ((int)queue_.size() == kQueue) {
                        queue_.pop_front();
                        reference_.OnReceiveOverflow();
                    }
                    queue_.push_back(rx_frames_);
                }
                rx_frames_ += kBuffer;
                next_rx_us_ += kPeriodUs;
            }
        }
    }

    EchoReference reference_;
    int stride_;
    int jitter_us_ = 0;
    std::mt19937 random_{12345};
    bool speaker_running_ = true;
    bool lose_rx_ = false;
    int64_t now_us_ = 0;
    int64_t tx_origin_us_ = 0;
    int64_t rx_origin_us_ = 0;
    int64_t next_tx_us_ = 0;
    int64_t next_rx_us_ = 0;
    int64_t tx_frames_ = 0;  // frames reported sent
    int64_t rx_frames_ = 0;  // frames captured, reported or not
    std::deque<int64_t> queue_;
    int64_t stopped_since_us_ = -1;                  // speaker stopped at this time, or -1
    std::vector<std::pair<int64_t, int64_t>> quiet_;  // earlier stops, as [from, until) times
};

struct Verdict {
    int checked = 0;        // frames with both an echo and a reference
    int unreferenced = 0;   // frames with an echo and no reference
    int min_lead = 1 << 30;
    int max_lead = -(1 << 30);
};

// How far the reference runs ahead of what the microphone hears, beyond the
// physical echo delay. EchoReference aims for Config::lead_frames.
Verdict Judge(const std::vector<Rig::Frame>& frames, size_t from = 0) {
    Verdict verdict;
    for (size_t i = from; i < frames.size(); ++i) {
        if (frames[i].mic == 0) continue;
        if (frames[i].reference == 0) {
            ++verdict.unreferenced;
            continue;
        }
        const int lead = RampDistance(frames[i].reference, frames[i].mic) - kEchoDelay;
        ++verdict.checked;
        verdict.min_lead = std::min(verdict.min_lead, lead);
        verdict.max_lead = std::max(verdict.max_lead, lead);
    }
    return verdict;
}

constexpr int kLead = 24;  // EchoReference::Config::lead_frames
constexpr size_t kSettle = 20 * kBuffer;  // frames allowed for finding the alignment

TEST(EchoReferenceTest, LeavesAudioAloneUntilEnabled) {
    Rig rig(3000);
    auto frames = rig.Run(300000);
    ASSERT_FALSE(frames.empty());
    int heard = 0;
    for (const auto& frame : frames) {
        EXPECT_EQ(frame.reference, kSecondChannel);
        heard += frame.mic != 0;
    }
    EXPECT_GT(heard, 0);
    EXPECT_EQ(rig.reference().GetStats().frames, 0u);
}

TEST(EchoReferenceTest, ReferenceLeadsTheEchoWhateverThePhaseOfTheTwoInterrupts) {
    for (int64_t phase = 0; phase < kPeriodUs; phase += 617) {
        Rig rig(phase);
        rig.reference().SetEnabled(true);
        auto frames = rig.Run(1000000);
        auto verdict = Judge(frames, kSettle);
        ASSERT_GT(verdict.checked, 10000) << "phase " << phase;
        EXPECT_EQ(verdict.unreferenced, 0) << "phase " << phase;
        EXPECT_GE(verdict.min_lead, kLead - 2) << "phase " << phase;
        EXPECT_LE(verdict.max_lead, kLead + 2) << "phase " << phase;
        auto stats = rig.reference().GetStats();
        EXPECT_TRUE(stats.synced);
        EXPECT_TRUE(stats.located);
        EXPECT_EQ(stats.resyncs, 0u);
        EXPECT_EQ(stats.relocations, 0u);
        EXPECT_EQ(stats.late_frames, 0u) << "phase " << phase;
    }
}

TEST(EchoReferenceTest, TakesTheWantedSampleFromAStereoBuffer) {
    Rig rig(4200, 2);
    rig.reference().SetEnabled(true);
    auto verdict = Judge(rig.Run(600000), kSettle);
    ASSERT_GT(verdict.checked, 5000);
    EXPECT_GE(verdict.min_lead, kLead - 2);
    EXPECT_LE(verdict.max_lead, kLead + 2);
}

TEST(EchoReferenceTest, CanBeSwitchedOnWhileCaptureIsRunning) {
    Rig rig(7300);
    rig.Run(400000);
    rig.reference().SetEnabled(true);
    auto verdict = Judge(rig.Run(1000000), kSettle);
    ASSERT_GT(verdict.checked, 5000);
    EXPECT_EQ(verdict.unreferenced, 0);
    EXPECT_GE(verdict.min_lead, kLead - 2);
    EXPECT_LE(verdict.max_lead, kLead + 2);
}

TEST(EchoReferenceTest, ToleratesInterruptsThatAreStampedLate) {
    Rig rig(2500);
    rig.SetJitterUs(300);
    rig.reference().SetEnabled(true);
    auto verdict = Judge(rig.Run(3000000), kSettle);
    ASSERT_GT(verdict.checked, 50000);
    EXPECT_EQ(verdict.unreferenced, 0);
    // 300 us is 7 frames; the reference must stay ahead of the echo.
    EXPECT_GE(verdict.min_lead, kLead - 10);
    EXPECT_LE(verdict.max_lead, kLead + 10);
    EXPECT_GE(verdict.min_lead + kEchoDelay, 0);
    EXPECT_EQ(rig.reference().GetStats().resyncs, 0u);
}

TEST(EchoReferenceTest, RealignsAfterTheReaderFallsBehindAndAudioIsDropped) {
    Rig rig(1200);
    rig.reference().SetEnabled(true);
    rig.Run(500000);
    rig.StallReader(rig.now_us() + 180000);  // 18 buffers pile up on a queue of 5
    auto frames = rig.Run(rig.now_us() + 1000000);
    auto stats = rig.reference().GetStats();
    EXPECT_GT(stats.rx_overflows, 0u);
    EXPECT_GT(stats.gaps, 0u);
    // Right after the stall some reference has scrolled out of the ring;
    // those frames get silence, never a wrong sample.
    auto all = Judge(frames);
    EXPECT_GE(all.min_lead, kLead - 2);
    EXPECT_LE(all.max_lead, kLead + 2);
    auto settled = Judge(frames, kSettle);
    ASSERT_GT(settled.checked, 10000);
    EXPECT_EQ(settled.unreferenced, 0);
}

TEST(EchoReferenceTest, RealignsAfterCaptureIsRestarted) {
    Rig rig(8800);
    rig.reference().SetEnabled(true);
    rig.Run(500000);
    rig.RestartCapture(rig.now_us() + 37000);
    auto frames = rig.Run(rig.now_us() + 1000000);
    auto all = Judge(frames);
    EXPECT_GE(all.min_lead, kLead - 2);
    EXPECT_LE(all.max_lead, kLead + 2);
    auto settled = Judge(frames, kSettle);
    ASSERT_GT(settled.checked, 10000);
    EXPECT_EQ(settled.unreferenced, 0);
}

TEST(EchoReferenceTest, GivesSilenceWhileTheSpeakerClockIsStoppedAndRealignsAfter) {
    Rig rig(5100);
    rig.reference().SetEnabled(true);
    rig.Run(500000);
    rig.StopSpeaker();
    auto during = rig.Run(rig.now_us() + 300000);
    int referenced = 0;
    for (size_t i = 10 * kBuffer; i < during.size(); ++i) {
        referenced += during[i].reference != 0;
    }
    EXPECT_EQ(referenced, 0);
    EXPECT_FALSE(rig.reference().GetStats().synced);

    rig.StartSpeaker(rig.now_us() + 4321);
    auto frames = rig.Run(rig.now_us() + 1000000);
    auto settled = Judge(frames, kSettle);
    ASSERT_GT(settled.checked, 10000);
    EXPECT_EQ(settled.unreferenced, 0);
    EXPECT_GE(settled.min_lead, kLead - 2);
    EXPECT_LE(settled.max_lead, kLead + 2);
}

TEST(EchoReferenceTest, RecoversWhenACaptureInterruptIsLost) {
    Rig rig(6400);
    rig.reference().SetEnabled(true);
    rig.Run(500000);
    rig.LoseNextCaptureInterrupt();
    auto frames = rig.Run(rig.now_us() + 1000000);
    // The capture count is now one buffer short of the truth. That shows up
    // as the two clocks having moved apart by a buffer, and is corrected as
    // such after a few chunks.
    EXPECT_EQ(rig.reference().GetStats().resyncs, 1u);
    auto settled = Judge(frames, kSettle);
    ASSERT_GT(settled.checked, 10000);
    EXPECT_EQ(settled.unreferenced, 0);
    EXPECT_GE(settled.min_lead, kLead - 2);
    EXPECT_LE(settled.max_lead, kLead + 2);
}

TEST(EchoReferenceTest, RecordsWhatTheCancellerIsGivenOnceTheSpeakerIsHeard) {
    Rig rig(900);
    rig.reference().SetEnabled(true);
    std::vector<int16_t> mic(4000), reference(4000), second(4000);
    rig.reference().ArmCapture(mic.data(), reference.data(), second.data(), mic.size(), 1);
    rig.Run(600000);
    ASSERT_EQ(rig.reference().CapturedFrames(), mic.size());
    for (size_t i = 0; i < mic.size(); ++i) {
        ASSERT_NE(reference[i], 0);
        EXPECT_EQ(second[i], kSecondChannel);
    }
    // The ramp is no test signal for correlation; what matters here is that
    // the recording is what Process() produced.
    EXPECT_NEAR(RampDistance(reference[2000], mic[2000]) - kEchoDelay, kLead, 2);
}

TEST(AnalyzeEchoTest, FindsTheDelayAndStrengthOfAnEcho) {
    std::mt19937 random(7);
    std::uniform_int_distribution<int> speech(-8000, 8000);
    std::uniform_int_distribution<int> room(-400, 400);
    const int delay = 37;
    std::vector<int16_t> reference(6000), mic(6000), inverted(6000), unrelated(6000);
    for (auto& sample : reference) sample = (int16_t)speech(random);
    for (size_t i = 0; i < mic.size(); ++i) {
        const int echo = i >= (size_t)delay ? reference[i - delay] / 2 : 0;
        mic[i] = (int16_t)(echo + room(random));
        inverted[i] = (int16_t)(-echo + room(random));
        unrelated[i] = (int16_t)speech(random);
    }

    auto heard = AnalyzeEcho(mic.data(), reference.data(), mic.size(), -200, 400, 1);
    EXPECT_EQ(heard.best_lag, delay);
    EXPECT_GT(heard.correlation_permille, 900);
    EXPECT_NEAR(heard.rms_reference, 4619, 200);  // uniform +-8000

    auto flipped = AnalyzeEcho(inverted.data(), reference.data(), mic.size(), -200, 400, 1);
    EXPECT_EQ(flipped.best_lag, delay);
    EXPECT_LT(flipped.correlation_permille, -900);

    auto nothing = AnalyzeEcho(unrelated.data(), reference.data(), mic.size(), -200, 400, 1);
    EXPECT_LT(std::abs(nothing.correlation_permille), 100);

    std::vector<int16_t> silence(6000, 0);
    auto dead = AnalyzeEcho(silence.data(), reference.data(), mic.size(), -200, 400, 1);
    EXPECT_EQ(dead.rms_mic, 0);
    EXPECT_EQ(dead.correlation_permille, 0);
}

}  // namespace
