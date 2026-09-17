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

    async def send_listen_state(
        self,
        state: str,
        mode: str = "manual",
        profile: str = "voice",
    ) -> None:
        self.listen_calls.append((state, mode, profile))


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


def _conversation(gateway: Any = None) -> AgentsConversation:
    return AgentsConversation(
        gateway or _FakeGateway(),
        agent_id="agent_test",
        api_key="key_test",
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
