# Interrupting the robot mid-reply (barge-in)

In an ElevenLabs Agents conversation the far side already knows how to be
interrupted: it listens while it speaks, and when the guest talks over a reply
it stops and sends an `interruption` event. Until this change that never
happened with the robot, because the robot was not sending it anything to hear.
The firmware switched voice capture off the moment it started speaking and the
gateway re-armed the microphone after every reply. On a real call that looked
like this: `interruptions: 0`, `listen_rearms` equal to the number of turns, and
no user speech transcribed during any reply.

This branch makes the microphone stay live through playback.

## What changed

**Firmware**

- `listen.start` from the gateway now honours `"mode": "realtime"`. In a
  realtime session the device keeps voice processing running while it speaks
  and returns to Listening, not Idle, when a reply ends. The session ends on
  `listen.stop`, or whenever the device reaches Idle for any other reason.
- Realtime is refused, with a log line and a fallback to manual, unless echo
  cancellation is on. A live microphone beside a playing speaker with nothing
  removing the speaker from the signal feeds every reply back to the agent as
  if the guest had said it.
- New `{"type":"tts","state":"abort"}`: drop the audio already buffered on the
  device. `tts.stop` ends an utterance but lets buffered frames play out, which
  is up to a couple of seconds of a sentence that was supposed to stop.
- Device-side AEC is **not** enabled for the StackChan build. The first
  version of this branch did enable it, and it broke the microphone; see
  "First flash" below. Until the board has a real echo reference, realtime
  requests are refused by the firmware and the robot stays half-duplex.

**Gateway**

- `STACKCHAN_AGENTS_LISTEN_MODE=realtime` (or `listen_mode: "realtime"` on
  `agents_converse_start`) asks the device for a realtime session. The default
  is still `manual`, so deploying the gateway changes nothing by itself.
- On `interruption` in realtime mode the gateway sends `tts abort` before it
  cancels playback.
- The per-turn re-arm is still sent in realtime mode. New firmware ignores it
  while listening; old firmware needs it. A gateway set to realtime therefore
  cannot leave an un-flashed device deaf.
- `agents_converse_status` reports `listen_mode`.

## First flash (2026-10-10): AEC silenced the microphone

Commit 702d754 built the StackChan firmware with `CONFIG_USE_DEVICE_AEC=y`.
On the robot the microphone still produced frames (`listen()` returned 65
frames for 3.9 s) but nothing in them could be transcribed: three
conversations, in manual and in realtime mode, were recorded by the agent as
silence while the guest was speaking.

The likely cause is in `boards/stackchan/cores3_audio_codec.cc`. The ES7210 is
set up with MIC1, MIC2 and MIC3 on a four-slot TDM bus, and capture opens
channels 0 and 1. `AUDIO_INPUT_REFERENCE true` then labels channel 1 as the
echo reference, but channel 1 is the second microphone, not the speaker
loopback. Without AEC the audio front end ignores the reference, which is why
the stock build works. With AEC it subtracts one microphone from the other, and
a voice that reaches both is exactly what gets removed. This reading comes from
the code and the symptoms; the captured audio level was not measured.

So AEC is off again in this branch, and the realtime plumbing stays in as dead
weight until someone establishes which ES7210 input, if any, carries the
speaker signal on the CoreS3 (MIC3 is the candidate) and feeds that to the AFE
as the reference.

## What has not been verified

The gateway side is covered by unit tests. **Nothing here has been heard on a
robot.** Three things can only be settled on the hardware:

1. **Whether the echo cancellation is good enough.** The speaker and the
   microphones share a small enclosure, and Espressif's own note on this option
   asks for "physical acoustic isolation between the microphone and speaker".
   If it is not good enough the robot will hear itself and cut its own replies
   short, or the agent will transcribe the robot's speech as the guest's.
2. **How it behaves with the neck servos and at the current speaker volume.**
3. **Whether enabling AEC changes anything for `listen()` and beat mode.** With
   AEC compiled in, the audio front end is created without its voice-activity
   detector (`vad_init = false`), as on every other AEC board upstream.

So treat the first flash as an experiment, and keep the previous firmware
binary at hand.

## Build and flash

```bash
git submodule update --init --recursive
cd firmware
docker run --rm --cpus=4 --ulimit nofile=65536:65536 \
  -v $PWD:/project -w /project espressif/idf:v5.5.2 \
  python ./scripts/release.py stackchan

# App-only update over USB; keeps Wi-Fi and gateway settings in NVS.
esptool.py --chip esp32s3 --port /dev/cu.usbmodem1101 -b 460800 \
  write_flash 0x20000 build/xiaozhi.bin
```

## Turn it on

On the gateway host, add to the gateway's environment and restart it:

```bash
STACKCHAN_AGENTS_LISTEN_MODE=realtime
```

For a single call without touching the environment, pass
`{"listen_mode": "realtime"}` to `agents_converse_start`.

## Check that it works

Start a conversation, ask for something long, and talk over the answer. Then
read `agents_converse_status`:

| field | half-duplex (before) | realtime working |
|---|---|---|
| `listen_mode` | `manual` | `realtime` |
| `frames_uplinked` | stops growing while the robot speaks | keeps growing while the robot speaks |
| `interruptions` | `0` | goes up when you talk over a reply |

`listen_mode: realtime` with `frames_uplinked` frozen during replies means the
firmware did not accept the mode: either it is the old build, or it logged
`Realtime listening needs echo cancellation` on the serial console.

Signs that the echo cancellation is not holding:

- `interruptions` rises while nobody is speaking;
- replies stop a second or two in, by themselves;
- the conversation transcript shows the robot's own sentences as user turns.

If that happens, lower the speaker volume first (`set_volume`), then consider
the agent's turn settings on the ElevenLabs side before giving up on the mode.

## Go back

Set `STACKCHAN_AGENTS_LISTEN_MODE=manual` (or remove it) and restart the
gateway. That alone restores the old behaviour, whatever firmware is on the
robot. To undo the firmware as well, flash the previous `xiaozhi.bin`.
