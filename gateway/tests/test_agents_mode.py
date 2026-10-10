from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from types import SimpleNamespace
from typing import Any

import pytest

from stackchan_mcp import agents_mode
from stackchan_mcp.agents_mode import AgentsConversation, is_agents_mode_owner


class _FakeESP32:
    def __init__(self) -> None:
        self.device_connected = True
        self.connection = SimpleNamespace(protocol_version=1, session_id="conv-session")
        self.listen_lock = asyncio.Lock()
        self.listen_calls: list[tuple[str, str, str]] = []
        self.tts_calls: list[str] = []
        self.tool_calls: list[tuple[str, dict[str, Any]]] = []
        # What the device answers to a tool call: a (result, error) pair, or
        # None for firmware that knows the echo cancellation tool.
        self.tool_reply: tuple[Any, Any] | None = None

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> tuple[Any, Any]:
        self.tool_calls.append((name, arguments))
        if self.tool_reply is not None:
            return self.tool_reply
        payload = {"ok": True, "enabled": arguments["enabled"], "persistence": "until reboot"}
        return {"content": [{"type": "text", "text": json.dumps(payload)}]}, None

    async def send_listen_state(
        self,
        state: str,
        mode: str = "manual",
        profile: str = "voice",
    ) -> None:
        self.listen_calls.append((state, mode, profile))

    async def send_tts_state(self, state: str) -> None:
        self.tts_calls.append(state)


class _FakeGateway:
    def __init__(self) -> None:
        self.esp32 = _FakeESP32()


class _FakeWS:
    """Minimal stand-in for the ElevenLabs conversation socket."""

    def __init__(self, inbound: list[dict[str, Any]] | None = None) -> None:
        self.sent: list[str] = []
        self._inbound = list(inbound or [])
        self.closed = False

    async def send(self, payload: str) -> None:
        self.sent.append(payload)

    async def recv(self) -> str:
        if not self._inbound:
            raise ConnectionError("closed")
        return json.dumps(self._inbound.pop(0))

    async def close(self) -> None:
        self.closed = True


def _conversation(gateway: Any = None, *, listen_mode: str = "manual") -> AgentsConversation:
    return AgentsConversation(
        gateway or _FakeGateway(),
        agent_id="agent_test",
        api_key="key_test",
        listen_mode=listen_mode,
    )


def test_owner_token_is_recognised_and_scoped() -> None:
    assert is_agents_mode_owner("agents_mode")
    assert is_agents_mode_owner("agents_mode:7")
    assert not is_agents_mode_owner("beat_mode:7")
    assert not is_agents_mode_owner(None)


def test_resolve_agent_id_prefers_explicit_then_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STACKCHAN_ELEVEN_AGENT_ID", "from_env")
    assert agents_mode.resolve_agent_id("explicit") == "explicit"
    assert agents_mode.resolve_agent_id(None) == "from_env"
    monkeypatch.delenv("STACKCHAN_ELEVEN_AGENT_ID")
    with pytest.raises(ValueError):
        agents_mode.resolve_agent_id(None)


def test_resolve_api_key_requires_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("STACKCHAN_ELEVENLABS_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API key is not configured"):
        agents_mode.resolve_api_key()

    monkeypatch.setenv("ELEVENLABS_API_KEY", "generic")
    assert agents_mode.resolve_api_key() == "generic"
    # The StackChan-specific name wins so a machine-wide key can be
    # overridden for this gateway.
    monkeypatch.setenv("STACKCHAN_ELEVENLABS_KEY", "specific")
    assert agents_mode.resolve_api_key() == "specific"


@pytest.mark.asyncio
async def test_audio_event_is_queued_and_is_final_closes_the_turn() -> None:
    conv = _conversation()
    pcm = b"\x01\x02" * 8

    await conv._handle_event(
        {
            "type": "audio",
            "audio_event": {"audio_base_64": base64.b64encode(pcm).decode()},
        }
    )
    await conv._handle_event(
        {
            "type": "audio",
            "audio_event": {
                "audio_base_64": base64.b64encode(pcm).decode(),
                "is_final": True,
            },
        }
    )

    assert conv._playback_queue.get_nowait() == pcm
    assert conv._playback_queue.get_nowait() == pcm
    # The sentinel is what ends one send_pcm_stream call, letting the
    # device drop out of the speaking state between replies.
    assert conv._playback_queue.get_nowait() is None
    assert conv.status()["audio_chunks_received"] == 2


@pytest.mark.asyncio
async def test_agent_response_complete_closes_the_turn() -> None:
    """``is_final`` is optional on the wire; this may be the only marker."""
    conv = _conversation()
    pcm = b"\x03\x04" * 8

    await conv._handle_event(
        {
            "type": "audio",
            "audio_event": {"audio_base_64": base64.b64encode(pcm).decode()},
        }
    )
    await conv._handle_event(
        {
            "type": "agent_response_complete",
            "agent_response_complete_event": {"event_id": 9},
        }
    )

    assert conv._playback_queue.get_nowait() == pcm
    assert conv._playback_queue.get_nowait() is None
    assert conv.status()["last_turn_end_reason"] == "agent_response_complete"


@pytest.mark.asyncio
async def test_turn_closes_itself_when_no_end_marker_ever_arrives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A turn nobody closes must not keep the device speaking forever."""
    monkeypatch.setattr(agents_mode, "TURN_IDLE_TIMEOUT_S", 0.05)

    played: list[bytes] = []

    async def _fake_send_pcm_stream(_gateway: Any, chunks: Any, **_kwargs: Any) -> None:
        async for chunk in chunks:
            played.append(chunk)

    monkeypatch.setattr(agents_mode, "send_pcm_stream", _fake_send_pcm_stream)

    conv = _conversation()
    await conv._play_turn(b"only-chunk")

    assert played == [b"only-chunk"]
    assert conv.status()["turns_played"] == 1
    assert conv.status()["last_turn_end_reason"] == "idle_timeout"


async def _wait_until(predicate: Any, timeout_s: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition was not reached in time")
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_playback_loop_rearms_the_mic_after_every_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The device drops out of listening to speak and returns to idle.

    So the single ``listen.start`` that ``start`` issues covers only the
    window before the agent's first reply. Miss the re-arm and the user
    is never heard again for the rest of the conversation.
    """

    async def _fake_send_pcm_stream(_gateway: Any, chunks: Any, **_kwargs: Any) -> None:
        async for _chunk in chunks:
            pass

    monkeypatch.setattr(agents_mode, "send_pcm_stream", _fake_send_pcm_stream)

    gateway = _FakeGateway()
    conv = _conversation(gateway)
    conv._active = True

    loop_task = asyncio.create_task(conv._playback_loop())
    try:
        for _ in range(2):
            await conv._enqueue_playback(b"chunk")
            await conv._close_turn("agent_response_complete")
        await _wait_until(lambda: conv.status()["listen_rearms"] == 2)
    finally:
        loop_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await loop_task

    assert conv.status()["turns_played"] == 2
    # Re-armed with the same profile the conversation opened on: "raw"
    # here would hand the agent un-cancelled echo of its own voice.
    assert gateway.esp32.listen_calls == [
        ("start", "manual", "voice"),
        ("start", "manual", "voice"),
    ]


class _ConnectionClosedOK(Exception):
    """Stand-in for the websockets clean-close exception.

    ``_note_far_side_hangup`` matches on the class name so the module
    keeps its lazy websockets import, so the name here is load-bearing.
    """


_ConnectionClosedOK.__name__ = "ConnectionClosedOK"


@pytest.mark.asyncio
async def test_agent_hangup_releases_the_device_and_clears_the_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``end_call`` closes the socket; nothing else would reap the mode.

    Leaving it armed strands the recording slot, and ``listen()`` and beat
    mode then refuse to run until the gateway is restarted.
    """
    monkeypatch.setattr(agents_mode, "FAREWELL_GRACE_S", 0.2)
    monkeypatch.setattr(agents_mode, "_mode", None)

    gateway = _FakeGateway()
    conv = _conversation(gateway)
    conv._ws = _FakeWS()
    conv._active = True
    monkeypatch.setattr(agents_mode, "_mode", conv)

    conv._note_far_side_hangup(_ConnectionClosedOK("1000 (OK)"))
    assert conv._teardown_task is not None
    await asyncio.wait_for(conv._teardown_task, timeout=5.0)

    status = conv.status()
    assert status["ended_reason"] == "agent_hangup"
    assert status["active"] is False
    # A clean goodbye is not a failure, and must not be filed as one.
    assert status["last_error"] is None
    assert ("stop", "manual", "voice") in gateway.esp32.listen_calls
    assert agents_mode._mode is None


@pytest.mark.asyncio
async def test_downlink_loop_treats_a_closed_socket_as_a_hangup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The read loop is where the hangup is observed, so it must act on it.

    Returning quietly -- as it used to -- leaves ``active`` set and the
    microphone held by a conversation that no longer has a far side.
    """
    monkeypatch.setattr(agents_mode, "FAREWELL_GRACE_S", 0.2)
    monkeypatch.setattr(agents_mode, "_mode", None)

    class _ClosingWS(_FakeWS):
        async def recv(self) -> str:
            raise _ConnectionClosedOK("1000 (OK)")

    conv = _conversation()
    conv._ws = _ClosingWS()
    conv._active = True
    monkeypatch.setattr(agents_mode, "_mode", conv)

    await conv._downlink_loop()

    assert conv._teardown_task is not None, "closed socket did not trigger teardown"
    await asyncio.wait_for(conv._teardown_task, timeout=5.0)
    assert conv.status()["ended_reason"] == "agent_hangup"
    assert conv.status()["active"] is False


@pytest.mark.asyncio
async def test_a_dropped_socket_is_reported_as_a_loss_not_a_hangup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agents_mode, "FAREWELL_GRACE_S", 0.2)
    monkeypatch.setattr(agents_mode, "_mode", None)

    conv = _conversation()
    conv._ws = _FakeWS()
    conv._active = True
    monkeypatch.setattr(agents_mode, "_mode", conv)

    conv._note_far_side_hangup(ConnectionError("no close frame"))
    await asyncio.wait_for(conv._teardown_task, timeout=5.0)

    status = conv.status()
    assert status["ended_reason"] == "far_side_lost"
    assert "no close frame" in (status["last_error"] or "")


@pytest.mark.asyncio
async def test_our_own_stop_closing_the_socket_is_not_a_hangup() -> None:
    """``stop`` clears ``active`` before it closes the socket."""
    conv = _conversation()
    conv._active = False

    conv._note_far_side_hangup(_ConnectionClosedOK("1000 (OK)"))

    assert conv._teardown_task is None
    assert conv.status()["ended_reason"] is None


@pytest.mark.asyncio
async def test_teardown_waits_for_the_farewell_to_finish_playing() -> None:
    """The goodbye is still queued when the close lands: let it play."""
    conv = _conversation()
    conv._playback_queue.put_nowait(b"farewell")

    draining = asyncio.create_task(conv._drain_playback(3.0))
    await asyncio.sleep(0.15)
    assert not draining.done(), "released the device while the goodbye was queued"

    conv._drain_playback_queue()
    await asyncio.wait_for(draining, timeout=3.0)


@pytest.mark.asyncio
async def test_rearm_is_skipped_once_the_conversation_is_stopping() -> None:
    """``stop`` clears ``active`` before it cancels the playback task.

    A re-arm racing that teardown would leave the mic open on a
    conversation that no longer has anything reading from it.
    """
    gateway = _FakeGateway()
    conv = _conversation(gateway)
    conv._active = False

    await conv._rearm_listening()

    assert gateway.esp32.listen_calls == []
    assert conv.status()["listen_rearms"] == 0


@pytest.mark.asyncio
async def test_rearm_failure_is_recorded_rather_than_raised() -> None:
    """A dropped device must not take the playback loop down with it."""
    gateway = _FakeGateway()

    async def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise ConnectionError("device went away")

    gateway.esp32.send_listen_state = _boom  # type: ignore[method-assign]

    conv = _conversation(gateway)
    conv._active = True

    await conv._rearm_listening()

    assert conv.status()["listen_rearms"] == 0
    assert "listen re-arm failed" in (conv.status()["last_error"] or "")


@pytest.mark.asyncio
async def test_stop_gives_up_on_playback_that_ignores_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``stop`` runs under the module lock, so it must never wait forever."""
    monkeypatch.setattr(agents_mode, "STOP_GRACE_S", 0.05)

    conv = _conversation()
    conv._ws = _FakeWS()
    conv._active = True

    started = asyncio.Event()
    release = asyncio.Event()

    async def _uncancellable() -> None:
        started.set()
        while not release.is_set():
            try:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                continue

    stubborn = asyncio.create_task(_uncancellable())
    conv._playback_task = stubborn
    conv._tasks = [stubborn]
    await started.wait()

    try:
        await asyncio.wait_for(conv.stop(), timeout=2.0)
    finally:
        release.set()
        await stubborn

    assert conv.status()["active"] is False
    assert not stubborn.cancelled()


@pytest.mark.asyncio
async def test_interruption_drops_queued_audio_and_cancels_playback() -> None:
    conv = _conversation()
    for _ in range(3):
        conv._playback_queue.put_nowait(b"stale")

    started = asyncio.Event()

    async def _never_ending() -> None:
        started.set()
        await asyncio.sleep(3600)

    conv._playback_task = asyncio.create_task(_never_ending())
    await started.wait()

    await conv._handle_event({"type": "interruption", "interruption_event": {"event_id": 1}})

    assert conv._playback_queue.empty()
    assert conv._playback_task is None
    assert conv.status()["interruptions"] == 1


@pytest.mark.asyncio
async def test_ping_is_answered_with_matching_event_id() -> None:
    conv = _conversation()
    conv._ws = _FakeWS()

    await conv._handle_event({"type": "ping", "ping_event": {"event_id": 4242}})

    assert json.loads(conv._ws.sent[-1]) == {"type": "pong", "event_id": 4242}


@pytest.mark.asyncio
async def test_transcript_and_response_events_surface_in_status() -> None:
    conv = _conversation()

    await conv._handle_event(
        {"type": "user_transcript", "user_transcription_event": {"user_transcript": "hello"}}
    )
    await conv._handle_event(
        {"type": "agent_response", "agent_response_event": {"agent_response": "welcome"}}
    )
    await conv._handle_event(
        {
            "type": "agent_response_correction",
            "agent_response_correction_event": {
                "original_agent_response": "welcome to the whole tour",
                "corrected_agent_response": "welcome to",
            },
        }
    )

    status = conv.status()
    assert status["last_user_transcript"] == "hello"
    # After an interruption the agent retracts the tail it never spoke;
    # status should show what was actually said, not the original plan.
    assert status["last_agent_response"] == "welcome to"


@pytest.mark.asyncio
async def test_uplink_queue_overflow_is_counted_not_raised() -> None:
    conv = _conversation()
    limit = conv._uplink_queue.maxsize

    for _ in range(limit + 5):
        conv._on_opus_frame(b"frame")

    assert conv._uplink_queue.qsize() == limit
    assert conv.status()["frames_dropped"] == 5


@pytest.mark.asyncio
async def test_start_rejects_an_agent_with_a_mismatched_audio_format() -> None:
    conv = _conversation()
    conv._ws = _FakeWS(
        [
            {
                "type": "conversation_initiation_metadata",
                "conversation_initiation_metadata_event": {
                    "conversation_id": "conv_1",
                    "user_input_audio_format": "pcm_16000",
                    "agent_output_audio_format": "pcm_44100",
                },
            }
        ]
    )

    with pytest.raises(RuntimeError, match="pcm_16000"):
        await conv._await_initiation_metadata()


@pytest.mark.asyncio
async def test_initiation_metadata_records_the_conversation_id() -> None:
    conv = _conversation()
    conv._ws = _FakeWS(
        [
            {
                "type": "conversation_initiation_metadata",
                "conversation_initiation_metadata_event": {
                    "conversation_id": "conv_abc",
                    "user_input_audio_format": "pcm_16000",
                    "agent_output_audio_format": "pcm_16000",
                },
            }
        ]
    )

    await conv._await_initiation_metadata()

    assert conv.status()["conversation_id"] == "conv_abc"


@pytest.mark.asyncio
async def test_start_refuses_when_another_owner_holds_the_microphone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conv = _conversation()
    monkeypatch.setattr(agents_mode, "is_recording", lambda: True)
    monkeypatch.setattr(agents_mode, "recording_owner", lambda: "beat_mode:3")

    with pytest.raises(RuntimeError, match="already active"):
        await conv.start()


@pytest.mark.asyncio
async def test_start_refuses_a_disconnected_device() -> None:
    gateway = _FakeGateway()
    gateway.esp32.device_connected = False
    conv = _conversation(gateway)

    with pytest.raises(RuntimeError, match="No ESP32 device connected"):
        await conv.start()


@pytest.mark.asyncio
async def test_stop_is_idempotent_on_an_inactive_conversation() -> None:
    conv = _conversation()
    status = await conv.stop()
    assert status["active"] is False


# --- listen mode: can the guest be heard while the device speaks? ----


def test_listen_mode_is_manual_unless_asked_otherwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("STACKCHAN_AGENTS_LISTEN_MODE", raising=False)
    assert agents_mode.resolve_listen_mode() == "manual"

    monkeypatch.setenv("STACKCHAN_AGENTS_LISTEN_MODE", " Realtime ")
    assert agents_mode.resolve_listen_mode() == "realtime"
    # An explicit argument beats the environment, in both directions.
    assert agents_mode.resolve_listen_mode("manual") == "manual"

    monkeypatch.delenv("STACKCHAN_AGENTS_LISTEN_MODE")
    assert agents_mode.resolve_listen_mode("realtime") == "realtime"


def test_an_unknown_listen_mode_is_refused_not_guessed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STACKCHAN_AGENTS_LISTEN_MODE", "duplex")
    with pytest.raises(ValueError, match="duplex"):
        agents_mode.resolve_listen_mode()


@pytest.mark.asyncio
async def test_rearm_asks_for_the_mode_the_conversation_was_started_in() -> None:
    gateway = _FakeGateway()
    conv = _conversation(gateway, listen_mode="realtime")
    conv._active = True

    await conv._rearm_listening()

    # Sent in realtime mode too: firmware that supports it ignores the
    # message while listening, firmware that does not needs it to hear
    # anything after the first reply.
    assert gateway.esp32.listen_calls == [("start", "realtime", "voice")]
    assert conv.status()["listen_mode"] == "realtime"


@pytest.mark.asyncio
async def test_interruption_in_realtime_mode_empties_the_device_buffer_first() -> None:
    gateway = _FakeGateway()
    conv = _conversation(gateway, listen_mode="realtime")
    conv._playback_queue.put_nowait(b"stale")

    started = asyncio.Event()

    async def _never_ending() -> None:
        started.set()
        await asyncio.sleep(3600)

    task = asyncio.create_task(_never_ending())
    conv._playback_task = task
    await started.wait()

    still_playing_when_abort_was_sent: list[bool] = []

    async def _record(state: str) -> None:
        gateway.esp32.tts_calls.append(state)
        still_playing_when_abort_was_sent.append(not task.done())

    gateway.esp32.send_tts_state = _record  # type: ignore[method-assign]

    await conv._handle_event({"type": "interruption", "interruption_event": {"event_id": 1}})

    # ``abort`` goes out while the stream is still open, so the speaker is
    # silenced before the cancelled stream's own tts.stop arrives.
    assert gateway.esp32.tts_calls == ["abort"]
    assert still_playing_when_abort_was_sent == [True]
    assert task.cancelled()
    assert conv._playback_queue.empty()
    assert conv.status()["interruptions"] == 1


@pytest.mark.asyncio
async def test_interruption_in_manual_mode_sends_no_abort() -> None:
    gateway = _FakeGateway()
    conv = _conversation(gateway)

    await conv._handle_event({"type": "interruption", "interruption_event": {"event_id": 1}})

    assert gateway.esp32.tts_calls == []
    assert conv.status()["interruptions"] == 1


@pytest.mark.asyncio
async def test_a_failed_abort_does_not_keep_the_reply_playing() -> None:
    gateway = _FakeGateway()
    conv = _conversation(gateway, listen_mode="realtime")

    async def _never_ending() -> None:
        await asyncio.sleep(3600)

    task = asyncio.create_task(_never_ending())
    conv._playback_task = task
    await asyncio.sleep(0)

    async def _boom(state: str) -> None:
        raise ConnectionError("device went away")

    gateway.esp32.send_tts_state = _boom  # type: ignore[method-assign]

    await conv._handle_event({"type": "interruption", "interruption_event": {"event_id": 1}})

    assert task.cancelled()
    assert conv._playback_task is None


@pytest.mark.asyncio
async def test_realtime_switches_echo_cancellation_on_and_off_again() -> None:
    gateway = _FakeGateway()
    conv = _conversation(gateway, listen_mode="realtime")

    await conv._prepare_realtime()

    assert gateway.esp32.tool_calls == [("self.audio.set_echo_cancellation", {"enabled": True})]
    assert conv.status()["listen_mode"] == "realtime"
    assert conv.status()["echo_cancellation"] is True
    assert conv.status()["listen_mode_note"] is None

    conv._active = True
    await conv.stop()

    assert gateway.esp32.tool_calls[-1] == ("self.audio.set_echo_cancellation", {"enabled": False})
    assert conv.status()["echo_cancellation"] is False


@pytest.mark.asyncio
async def test_realtime_becomes_manual_on_firmware_without_echo_cancellation() -> None:
    gateway = _FakeGateway()
    gateway.esp32.tool_reply = (None, {"code": -32601, "message": "Unknown tool"})
    conv = _conversation(gateway, listen_mode="realtime")

    await conv._prepare_realtime()

    status = conv.status()
    assert status["listen_mode"] == "manual"
    assert status["echo_cancellation"] is False
    assert "Unknown tool" in status["listen_mode_note"]

    # Nothing was switched on, so stopping must not try to switch it off.
    conv._active = True
    await conv.stop()
    assert len(gateway.esp32.tool_calls) == 1


@pytest.mark.asyncio
async def test_realtime_becomes_manual_when_the_device_cannot_switch_it_on() -> None:
    gateway = _FakeGateway()
    refusal = {"ok": False, "enabled": False, "reason": "software echo reference unavailable"}
    gateway.esp32.tool_reply = ({"content": [{"type": "text", "text": json.dumps(refusal)}]}, None)
    conv = _conversation(gateway, listen_mode="realtime")

    await conv._prepare_realtime()

    status = conv.status()
    assert status["listen_mode"] == "manual"
    assert "software echo reference unavailable" in status["listen_mode_note"]


@pytest.mark.asyncio
async def test_a_manual_conversation_never_touches_echo_cancellation() -> None:
    gateway = _FakeGateway()
    conv = _conversation(gateway)
    conv._active = True

    await conv.stop()

    assert gateway.esp32.tool_calls == []
    assert conv.status()["echo_cancellation"] is None


# --- one conversation ending, the next beginning ----------------------
#
# Each of these is a way for a conversation that is over to go on acting:
# on the device, or on the conversation that follows it.


async def _speaking_stream(gateway: Any, chunks: Any, **_kwargs: Any) -> None:
    """``send_pcm_stream`` as the device sees it: tts.start, paced audio, tts.stop."""
    await gateway.esp32.send_tts_state("start")
    try:
        async for _chunk in chunks:
            await asyncio.sleep(0.02)
    finally:
        await gateway.esp32.send_tts_state("stop")


@pytest.mark.asyncio
async def test_stop_mid_reply_ends_the_playback_loop_for_good(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reply cut short by ``stop`` must not start again behind its back.

    The loop used to take its own cancellation for the turn's and carry on
    with what was left in the queue: ``stop`` waited out its grace period,
    released the device, and the rest of the reply was then played to it
    anyway -- holding the TTS lock against whichever conversation came next.
    """
    monkeypatch.setattr(agents_mode, "send_pcm_stream", _speaking_stream)
    monkeypatch.setattr(agents_mode, "STOP_GRACE_S", 1.0)
    monkeypatch.setattr(agents_mode, "TURN_IDLE_TIMEOUT_S", 0.05)

    gateway = _FakeGateway()
    conv = _conversation(gateway)
    conv._ws = _FakeWS()
    conv._active = True
    for _ in range(50):
        conv._playback_queue.put_nowait(b"reply")
    loop_task = asyncio.create_task(conv._playback_loop())
    conv._tasks = [loop_task]
    await _wait_until(lambda: gateway.esp32.tts_calls == ["start"])

    loop = asyncio.get_running_loop()
    began = loop.time()
    await conv.stop()
    took = loop.time() - began

    assert loop_task.done(), "the playback loop outlived stop"
    assert took < 0.5, "stop sat out its grace period waiting for the loop"
    # Long enough for a loop that survived to start on the rest of the reply.
    await asyncio.sleep(0.2)
    assert gateway.esp32.tts_calls == ["start", "stop"]
    assert conv._playback_queue.empty()


@pytest.mark.asyncio
async def test_an_interruption_ends_the_turn_but_not_the_playback_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other cancellation the loop sees, which it does have to survive."""
    monkeypatch.setattr(agents_mode, "send_pcm_stream", _speaking_stream)
    monkeypatch.setattr(agents_mode, "TURN_IDLE_TIMEOUT_S", 0.05)

    gateway = _FakeGateway()
    conv = _conversation(gateway)
    conv._active = True
    for _ in range(50):
        conv._playback_queue.put_nowait(b"reply")
    loop_task = asyncio.create_task(conv._playback_loop())
    try:
        await _wait_until(lambda: gateway.esp32.tts_calls == ["start"])
        await conv._handle_event({"type": "interruption", "interruption_event": {"event_id": 1}})
        await _wait_until(lambda: conv.status()["listen_rearms"] == 1)

        assert gateway.esp32.tts_calls == ["start", "stop"]
        assert not loop_task.done(), "an interruption took the playback loop down"

        await conv._enqueue_playback(b"next reply")
        await _wait_until(lambda: conv.status()["turns_played"] == 1)
    finally:
        loop_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await loop_task

    assert gateway.esp32.tts_calls == ["start", "stop", "start", "stop"]


@pytest.mark.asyncio
async def test_the_farewell_does_not_reopen_the_microphone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a hangup the turn that ends is the last: nothing is listening.

    Re-arming there puts ``listen.start`` on the wire milliseconds ahead of
    the teardown's ``listen.stop``. The firmware can apply those two in the
    opposite order, which leaves the device listening after it was released.
    """
    monkeypatch.setattr(agents_mode, "send_pcm_stream", _speaking_stream)
    monkeypatch.setattr(agents_mode, "TURN_IDLE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(agents_mode, "FAREWELL_GRACE_S", 5.0)

    gateway = _FakeGateway()
    conv = _conversation(gateway, listen_mode="realtime")
    conv._ws = _FakeWS()
    conv._active = True
    monkeypatch.setattr(agents_mode, "_mode", conv)
    conv._playback_queue.put_nowait(b"goodbye")
    loop_task = asyncio.create_task(conv._playback_loop())
    conv._tasks = [loop_task]
    await _wait_until(lambda: gateway.esp32.tts_calls == ["start"])

    conv._note_far_side_hangup(_ConnectionClosedOK("1000 (OK)"))
    assert conv._teardown_task is not None
    await asyncio.wait_for(conv._teardown_task, timeout=5.0)

    # The goodbye was played to its end, and then only the release followed.
    assert gateway.esp32.tts_calls == ["start", "stop"]
    assert conv.status()["turns_played"] == 1
    assert gateway.esp32.listen_calls == [("stop", "manual", "voice")]
    assert conv.status()["listen_rearms"] == 0
    assert agents_mode._mode is None


@pytest.mark.asyncio
async def test_a_late_hangup_teardown_leaves_the_next_conversation_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The farewell grace can outlast the conversation it belongs to.

    Stopped by hand while its goodbye was still playing, a conversation used
    to come back when the grace ran out and stop whatever was current.
    """
    monkeypatch.setattr(agents_mode, "FAREWELL_GRACE_S", 0.05)

    finished = _conversation()
    gateway = _FakeGateway()
    current = _conversation(gateway)
    current._ws = _FakeWS()
    current._active = True
    monkeypatch.setattr(agents_mode, "_mode", current)

    await finished._teardown_after_hangup()

    assert current.active
    assert agents_mode._mode is current
    assert gateway.esp32.listen_calls == []


@pytest.mark.asyncio
async def test_an_explicit_stop_retires_the_pending_hangup_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agents_mode, "FAREWELL_GRACE_S", 30.0)

    conv = _conversation()
    conv._ws = _FakeWS()
    conv._active = True
    monkeypatch.setattr(agents_mode, "_mode", conv)
    # A goodbye nobody is playing keeps the teardown waiting out its grace.
    conv._playback_queue.put_nowait(b"farewell")

    conv._note_far_side_hangup(_ConnectionClosedOK("1000 (OK)"))
    teardown = conv._teardown_task
    assert teardown is not None
    await asyncio.sleep(0.1)
    assert not teardown.done()

    status = await agents_mode.stop_agents_conversation()

    # Stopped by hand, but it was the far side that ended it.
    assert status["ended_reason"] == "agent_hangup"
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(teardown, timeout=1.0)
    assert teardown.cancelled()
    assert agents_mode._mode is None


@pytest.mark.asyncio
async def test_start_ends_a_leftover_listening_session_before_opening_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The device ignores ``listen.start`` while it is already listening.

    Seen 2026-10-10: a device still listening from the conversation before
    took the new one's ``listen.start`` as a no-op, and the board's 30 s
    listening timeout, counted from the old session, cut the microphone
    4.5 s in. 76 frames went up and then none; the agent, hearing nothing,
    never spoke, so no turn ended and nothing re-armed the microphone.
    """
    monkeypatch.setattr(agents_mode, "LISTEN_RESET_SETTLE_S", 0.05)
    monkeypatch.setattr(agents_mode, "_mode", None)

    class _QuietWS(_FakeWS):
        async def recv(self) -> str:
            if self._inbound:
                return json.dumps(self._inbound.pop(0))
            await asyncio.sleep(3600)
            raise AssertionError("unreachable")

    ws = _QuietWS(
        [
            {
                "type": "conversation_initiation_metadata",
                "conversation_initiation_metadata_event": {
                    "conversation_id": "conv_next",
                    "user_input_audio_format": "pcm_16000",
                    "agent_output_audio_format": "pcm_16000",
                },
            }
        ]
    )

    async def _signed_url(*_args: Any, **_kwargs: Any) -> str:
        return "wss://example.invalid/conversation"

    async def _connect(*_args: Any, **_kwargs: Any) -> _FakeWS:
        return ws

    monkeypatch.setattr(agents_mode, "fetch_signed_url", _signed_url)
    monkeypatch.setattr("websockets.connect", _connect)
    monkeypatch.setattr(agents_mode, "StreamingOpusDecoder", lambda: None)
    monkeypatch.setattr(agents_mode, "start_recording", lambda *_a, **_k: None)

    gateway = _FakeGateway()
    sent_at: list[float] = []
    record = gateway.esp32.send_listen_state

    async def _timed(state: str, mode: str = "manual", profile: str = "voice") -> None:
        sent_at.append(asyncio.get_running_loop().time())
        await record(state, mode=mode, profile=profile)

    gateway.esp32.send_listen_state = _timed  # type: ignore[method-assign]

    conv = _conversation(gateway)
    try:
        status = await conv.start()
        assert status["active"] is True
        assert gateway.esp32.listen_calls == [
            ("stop", "manual", "voice"),
            ("start", "manual", "voice"),
        ]
        # Far enough apart for the board to have seen the device idle.
        assert sent_at[1] - sent_at[0] >= 0.04
    finally:
        await conv.stop()


# --- what the agent is sent: level and gain --------------------------


def _tone(level: int, samples: int = 960) -> bytes:
    """A frame of 16-bit PCM whose RMS is exactly ``level``."""
    return b"".join(
        (level if index % 2 else -level).to_bytes(2, "little", signed=True)
        for index in range(samples)
    )


class _PassThroughDecoder:
    """The frames in these tests are already PCM."""

    def decode_frame(self, frame: bytes) -> bytes:
        return frame


async def _uplink(conv: AgentsConversation, frames: list[bytes]) -> list[bytes]:
    """Run frames through the uplink loop; returns the PCM the agent was sent."""
    conv._ws = _FakeWS()
    conv._decoder = _PassThroughDecoder()
    task = asyncio.create_task(conv._uplink_loop())
    try:
        before = conv.status()["frames_uplinked"]
        for frame in frames:
            conv._on_opus_frame(frame)
        await _wait_until(lambda: conv.status()["frames_uplinked"] == before + len(frames))
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    return [base64.b64decode(json.loads(sent)["user_audio_chunk"]) for sent in conv._ws.sent]


def test_uplink_gain_is_zero_unless_asked_otherwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("STACKCHAN_AGENTS_UPLINK_GAIN_DB", raising=False)
    assert agents_mode.resolve_uplink_gain_db() == 0.0

    monkeypatch.setenv("STACKCHAN_AGENTS_UPLINK_GAIN_DB", " 6 ")
    assert agents_mode.resolve_uplink_gain_db() == 6.0
    # An explicit request beats the environment, and so does an explicit 0.
    assert agents_mode.resolve_uplink_gain_db(9.5) == 9.5
    assert agents_mode.resolve_uplink_gain_db(0) == 0.0
    assert agents_mode.resolve_uplink_gain_db("") == 6.0


@pytest.mark.parametrize("bad", ["loud", "nan", 31, -13, True])
def test_an_uplink_gain_that_makes_no_sense_is_refused(
    monkeypatch: pytest.MonkeyPatch, bad: Any
) -> None:
    monkeypatch.delenv("STACKCHAN_AGENTS_UPLINK_GAIN_DB", raising=False)
    with pytest.raises(ValueError):
        agents_mode.resolve_uplink_gain_db(bad)


@pytest.mark.asyncio
async def test_without_gain_the_agent_gets_the_audio_as_the_device_sent_it() -> None:
    conv = _conversation()
    frame = _tone(300)

    sent = await _uplink(conv, [frame])

    assert sent == [frame]
    assert conv.status()["uplink_gain_db"] == 0.0
    assert conv.status()["uplink_clipped_samples"] == 0


@pytest.mark.asyncio
async def test_gain_amplifies_what_the_agent_gets_but_not_what_is_reported() -> None:
    conv = AgentsConversation(
        _FakeGateway(), agent_id="agent_test", api_key="key_test", uplink_gain_db=20.0
    )
    # Nine frames: half a second and a little, so the loudest-half-second
    # figure has a whole block to work with.
    sent = await _uplink(conv, [_tone(150)] * 9)

    assert sent[0] == _tone(1500)
    levels = conv.status()["uplink_levels"]["between_replies"]
    # Levels stay comparable with the device's own figures.
    assert levels["rms"] == 150
    assert levels["loudest_half_second_rms"] == 150
    assert levels["seconds"] == 0.5


@pytest.mark.asyncio
async def test_gain_that_overshoots_is_clipped_and_counted() -> None:
    conv = AgentsConversation(
        _FakeGateway(), agent_id="agent_test", api_key="key_test", uplink_gain_db=20.0
    )

    sent = await _uplink(conv, [_tone(5000, samples=10)])

    full_scale = b"".join(
        value.to_bytes(2, "little", signed=True) for value in (-32768, 32767) * 5
    )
    assert sent == [full_scale]
    assert conv.status()["uplink_clipped_samples"] == 10


@pytest.mark.asyncio
async def test_uplink_levels_are_kept_apart_for_replies_and_the_gaps_between(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conv = _conversation()

    await _uplink(conv, [_tone(40)] * 9)

    # A reply is playing: the echo that got through, then the guest
    # talking over it for half a second, then echo again.
    conv._reply_playing = True
    await _uplink(conv, [_tone(60)] * 9 + [_tone(400)] * 9 + [_tone(60)] * 9)
    conv._reply_playing = False
    conv._reply_stopped_at = agents_mode.time.monotonic()

    # The device is still emptying its buffer; this belongs to neither.
    await _uplink(conv, [_tone(3000)] * 9)

    monkeypatch.setattr(agents_mode, "UPLINK_LEVEL_TAIL_S", 0.0)
    await _uplink(conv, [_tone(40)] * 9)

    levels = conv.status()["uplink_levels"]
    assert levels["between_replies"] == {
        "seconds": 1.1,
        "rms": 40,
        "loudest_half_second_rms": 40,
    }
    assert levels["while_reply_plays"]["seconds"] == 1.6
    # The whole reply averages the guest away; the loudest half second
    # is where they show.
    assert levels["while_reply_plays"]["rms"] < 250
    assert levels["while_reply_plays"]["loudest_half_second_rms"] > 350


@pytest.mark.asyncio
async def test_a_reply_marks_its_own_start_and_end_for_the_levels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conv = _conversation()
    seen: list[str | None] = []

    async def _stream(_gateway: Any, chunks: Any, **_kwargs: Any) -> None:
        async for _chunk in chunks:
            seen.append(conv._uplink_phase())

    monkeypatch.setattr(agents_mode, "send_pcm_stream", _stream)
    assert conv._uplink_phase() == "between_replies"

    await conv._playback_queue.put(None)
    await conv._play_turn(b"\x00\x00" * 480)

    assert seen == ["while_reply_plays"]
    assert conv._uplink_phase() is None
    monkeypatch.setattr(agents_mode, "UPLINK_LEVEL_TAIL_S", 0.0)
    assert conv._uplink_phase() == "between_replies"
