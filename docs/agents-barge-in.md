# Interrupting the robot mid-reply (barge-in)

In an ElevenLabs Agents conversation the far side already knows how to be
interrupted: it listens while it speaks, and when the guest talks over a reply
it stops and sends an `interruption` event. With the robot that never
happened, because the robot was not sending it anything to hear. The firmware
switched voice capture off the moment it started speaking and the gateway
re-armed the microphone after every reply. On a real call that looked like
this: `interruptions: 0`, `listen_rearms` equal to the number of turns, and no
user speech transcribed during any reply.

Keeping the microphone open through playback is the easy half. The other half
is that the microphone then hears the robot, and the agent takes the robot's
own voice for the guest's. This branch does both: a realtime listening mode,
and echo cancellation on the device to make it usable.

**State: runs on a robot; the echo is cancelled and the agent can be
interrupted by a loud enough sound, but a person talking over a reply has not
yet been heard.** See "How loud the agent needs it" for where that stands.

## How the echo is cancelled

Echo cancellation subtracts what the speaker played from what the microphone
heard, so it needs the speaker signal as a second input. Boards designed for
it loop the amplifier output back into a spare ADC channel. The StackChan
build reads two capture channels and labels the second one as that reference,
but when echo cancellation was first switched on over it the microphone went
silent (see "First flash").

So the reference is made in software, from the PCM the firmware hands to the
I2S driver. The difficulty is timing: the driver queues up to 60 ms of audio,
so the moment a sample is written says little about the moment it is played.
`boards/stackchan/echo_reference.{h,cc}` therefore does not take the samples
where they are written. It takes them from the driver's "this DMA buffer has
just been sent" interrupt, and it counts buffers in the capture interrupt as
well. Playback and capture share one I2S port and one bit clock, so the two
counts advance frame for frame, and with a timestamp on each interrupt the
speaker frame that was playing while a given microphone frame was captured
follows by arithmetic.

Two consequences:

- The microphone signal is held back by 12 ms. A buffer is only reported once
  it has been sent, and the reference has to be known before the echo it
  explains arrives.
- The reference is deliberately taken 1 ms early. A canceller copes with a
  reference that leads the echo and not with one that trails it.

The alignment re-establishes itself after anything that breaks the count:
capture restarting, the reader falling behind and the driver dropping audio,
the speaker clock stopping, a lost interrupt. `firmware/host_test/
test_echo_reference.cc` runs each of those against simulated interrupts.

## What changed

**Firmware**

- `listen.start` from the gateway honours `"mode": "realtime"`. In a realtime
  session the device keeps voice processing running while it speaks and
  returns to Listening, not Idle, when a reply ends. The session ends on
  `listen.stop`, or whenever the device reaches Idle for any other reason.
- Realtime is refused, with a log line and a fallback to manual, unless echo
  cancellation is on.
- Echo cancellation is compiled in but **off at boot**
  (`CONFIG_USE_SWITCHABLE_DEVICE_AEC`). Until something switches it on, the
  audio path is the one the stock firmware has: the software reference is not
  applied and the canceller is not in the path.
- `self.audio.set_echo_cancellation {enabled}` switches the software
  reference and the canceller together. Not persisted: a reboot is always a
  way back to the stock audio path.
- `self.audio.get_echo_stats` and `self.audio.echo_probe` measure the result
  (below).
- `{"type":"tts","state":"abort"}` drops the audio already buffered on the
  device. `tts.stop` ends an utterance but lets buffered frames play out,
  which is up to a couple of seconds of a sentence that was supposed to stop.

**Gateway**

- `STACKCHAN_AGENTS_LISTEN_MODE=realtime` (or `listen_mode: "realtime"` on
  `agents_converse_start`) asks for a realtime conversation. The default is
  still `manual`, so deploying the gateway changes nothing by itself.
- A realtime conversation first asks the device to switch echo cancellation
  on. If the device cannot (older firmware, or the I2S hooks failed), the
  conversation runs in manual mode and `agents_converse_status` says so in
  `listen_mode_note`. Echo cancellation is switched off again when the
  conversation ends.
- On `interruption` in realtime mode the gateway sends `tts abort` before it
  cancels playback.
- The per-turn re-arm is still sent in realtime mode. New firmware ignores it
  while listening; old firmware needs it.
- New tools `set_echo_cancellation`, `get_echo_stats`, `echo_probe` relay to
  the firmware tools of the same names.

## First flash (2026-10-10): AEC silenced the microphone

Commit 702d754 built the StackChan firmware with `CONFIG_USE_DEVICE_AEC=y`,
which runs the canceller from boot over the hardware's second capture
channel. On the robot the microphone still produced frames (`listen()`
returned 65 frames for 3.9 s) but nothing in them could be transcribed: three
conversations, in manual and in realtime mode, were recorded by the agent as
silence while the guest was speaking.

What that second channel carries was never measured. The ES7210 is set up
with MIC1, MIC2 and MIC3 and capture opens two slots, so it is either the
second microphone or MIC3, and nothing in the code says MIC3 is wired to the
amplifier on this board. Either would explain the result: a canceller given a
second microphone removes the voice both microphones hear, and one given a
floating input can decide the far end never stops talking and suppress
everything. `echo_probe` reports the level of that channel and how it relates
to the microphone and to the speaker signal, which settles the question the
next time the robot is on a bench.

## What has not been verified

Host tests cover the alignment arithmetic and the gateway tests cover the
conversation logic. **Nothing here has run on a robot.** Open, in the order
they should be checked:

1. **That the stock path is untouched.** With echo cancellation off, the
   microphone must behave as before. The I2S callbacks are registered either
   way, and the front end now contains a canceller it is told not to use.
2. **That the reference lines up on real hardware.** `get_echo_stats` must
   show `reference.synced` and `reference.located` true with `resyncs`,
   `relocations`, `late_frames` and `stale_frames` not growing, and
   `echo_probe` must show the microphone following the reference by a few
   milliseconds.
3. **How much echo is removed.** The speaker and the microphones share a
   small enclosure and the speaker distorts at volume, which a linear
   canceller cannot follow. `echo_reduction_db` with nobody talking is the
   number.
4. **Whether what is left still trips the agent.** If `interruptions` rises
   while nobody speaks, or replies stop by themselves, the residue is too
   loud: lower the speaker volume first.
5. **Whether the guest is still heard while the robot talks.** That is the
   point of the exercise and the one thing only a person can check.

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

Keep the previous `xiaozhi.bin`, or a full `read_flash 0x0 0x1000000` backup,
at hand. Avatar sets live in PSRAM and have to be loaded again after any
reboot.

## Measure before trusting it

All of this can be done without anybody speaking to the robot.

```text
set_echo_cancellation {enabled: true}
get_echo_stats {reset: true}
echo_probe {action: "arm"}
agents_converse_start {listen_mode: "realtime"}     # let the greeting play
echo_probe {action: "result"}
get_echo_stats {}
agents_converse_status {}
```

| reading | good | means trouble |
|---|---|---|
| `reference.synced`, `reference.located` | both true | reference is silence; nothing is cancelled |
| `reference.resyncs`, `relocations` | 0, or 1 after a hiccup | climbing: the two clocks keep losing each other |
| `reference.late_frames`, `stale_frames` | 0 | the 12 ms hold-back is too short, or the reader stalls |
| `echo_probe.mic_vs_reference.lag_ms` | +1 to +10 | negative: reference trails the echo and is useless |
| `echo_probe.mic_vs_reference.correlation` | well away from 0 | near 0: microphone does not hear the speaker, or misaligned |
| `echo_reduction_db` (room quiet) | well above `quiet_reduction_db` | equal: the canceller removes nothing |
| `agents_converse_status.interruptions` | 0 while nobody speaks | the agent hears the robot |

`echo_probe.second_channel_*` describes the hardware channel the software
reference replaces. A correlation with the microphone near 1 at 0 ms means it
is a second microphone; a correlation with the reference near 1 means the
board has a loopback after all and could use it directly.

## How loud the agent needs it

Dialled directly, with synthetic speech in place of the microphone, the agent
has a level below which speech is silence to it. Levels are the RMS of 16-bit
samples while speaking.

| what it was sent | what it did |
|---|---|
| a question at 470, then an utterance at 470 or 250 over its reply | answered, and stopped within half a second of being talked over |
| the same utterance at 150 over its reply | carried on; nothing transcribed |
| a question at 150 in a quiet line | nothing; treated the guest as absent |
| its own reply coming back at 600 or 2500, nobody speaking | interrupted itself within about a second, every time |

On the robot, a person at the desk reaches the microphone at roughly 470
between replies. While the robot speaks, the canceller in `voip_low_cost`
takes its own voice from about 5000 down to 20..100, and lets a sound that is
not in the speaker signal through 2 to 10 dB weaker than it would in a quiet
room (measured with the head servos as the sound: 650 between replies, 230 to
440 during one). A voice thinned out like that sits at the agent's threshold,
which matches what was seen: with clean capture and 34 dB of echo removed, a
guest talking over three replies was not heard once.

That the path itself works was shown by switching the canceller off in the
middle of a reply: the robot's own voice went up the line and the agent
interrupted within three seconds.

Two things in the gateway address this:

- `agents_converse_status` reports `uplink_levels`: for the stretches when a
  reply was playing and for the stretches between replies, the level of the
  microphone audio over the whole of it and its loudest half second. With
  nobody talking, `while_reply_plays` is the echo that was left. When the
  guest talks over a reply, the loudest half second is their voice as the
  agent got it.
- `uplink_gain_db` on `agents_converse_start` (or
  `STACKCHAN_AGENTS_UPLINK_GAIN_DB`) amplifies the microphone audio before
  the agent gets it. It lifts the leftover echo by the same amount, so the
  room for it is what lies between the echo and the voice: lower the speaker
  volume to widen it. Raise the gain until `interruptions` rises with nobody
  talking, then go back by 6 dB.

## Turn it on

On the gateway host, add to the gateway's environment and restart it:

```bash
STACKCHAN_AGENTS_LISTEN_MODE=realtime
```

For a single call without touching the environment, pass
`{"listen_mode": "realtime"}` to `agents_converse_start`.

Then read `agents_converse_status` during a conversation:

| field | half-duplex | realtime working |
|---|---|---|
| `listen_mode` | `manual` | `realtime` |
| `listen_mode_note` | set if realtime was asked for and refused | `null` |
| `echo_cancellation` | `null` or `false` | `true` |
| `frames_uplinked` | stops growing while the robot speaks | keeps growing while the robot speaks |
| `uplink_levels.while_reply_plays` | `seconds` stays near 0 | `loudest_half_second_rms` jumps when you talk over a reply |
| `interruptions` | `0` | goes up when you talk over a reply |

## Go back

Set `STACKCHAN_AGENTS_LISTEN_MODE=manual` (or remove it) and restart the
gateway. That alone restores the old behaviour, whatever firmware is on the
robot. On the device, a reboot switches echo cancellation off. To undo the
firmware as well, flash the previous `xiaozhi.bin`.
