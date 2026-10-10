#ifndef _BOX_AUDIO_CODEC_H
#define _BOX_AUDIO_CODEC_H

#include "audio_codec.h"
#include "echo_reference.h"

#include <esp_codec_dev.h>
#include <esp_codec_dev_defaults.h>

class CoreS3AudioCodec : public AudioCodec {
private:
    const audio_codec_data_if_t* data_if_ = nullptr;
    const audio_codec_ctrl_if_t* out_ctrl_if_ = nullptr;
    const audio_codec_if_t* out_codec_if_ = nullptr;
    const audio_codec_ctrl_if_t* in_ctrl_if_ = nullptr;
    const audio_codec_if_t* in_codec_if_ = nullptr;
    const audio_codec_gpio_if_t* gpio_if_ = nullptr;

    esp_codec_dev_handle_t output_dev_ = nullptr;
    esp_codec_dev_handle_t input_dev_ = nullptr;

    // Stands in for the speaker loopback this board does not have. Off until
    // SetSoftwareReference(true); see echo_reference.h.
    EchoReference echo_reference_;
    bool echo_reference_hooked_ = false;

    static bool OnI2sSent(i2s_chan_handle_t handle, i2s_event_data_t* event, void* user_ctx);
    static bool OnI2sSendOverflow(i2s_chan_handle_t handle, i2s_event_data_t* event, void* user_ctx);
    static bool OnI2sReceived(i2s_chan_handle_t handle, i2s_event_data_t* event, void* user_ctx);
    static bool OnI2sReceiveOverflow(i2s_chan_handle_t handle, i2s_event_data_t* event, void* user_ctx);

    void CreateDuplexChannels(gpio_num_t mclk, gpio_num_t bclk, gpio_num_t ws, gpio_num_t dout, gpio_num_t din);

    virtual int Read(int16_t* dest, int samples) override;
    virtual int Write(const int16_t* data, int samples) override;

public:
    CoreS3AudioCodec(void* i2c_master_handle, int input_sample_rate, int output_sample_rate,
        gpio_num_t mclk, gpio_num_t bclk, gpio_num_t ws, gpio_num_t dout, gpio_num_t din,
        uint8_t aw88298_addr, uint8_t es7210_addr, bool input_reference);
    virtual ~CoreS3AudioCodec();

    virtual void SetOutputVolume(int volume) override;
    virtual void EnableInput(bool enable) override;
    virtual void EnableOutput(bool enable) override;

    // Replace capture channel 1 with a reference built from the samples sent
    // to the speaker. Returns false, and changes nothing, when the I2S
    // interrupts could not be hooked at start-up.
    bool SetSoftwareReference(bool enable);
    EchoReference& echo_reference() { return echo_reference_; }
};

#endif // _BOX_AUDIO_CODEC_H
