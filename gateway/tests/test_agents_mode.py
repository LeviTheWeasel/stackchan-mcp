from __future__ import annotations

import asyncio
import base64
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
