"""Full-duplex conversation mode backed by the ElevenLabs Agents platform.

``listen()`` captures a fixed window and transcribes it; the caller decides
when the user stopped talking. That is fine for one-shot commands and wrong
for conversation — a fixed window either truncates a sentence or waits
through silence, and there is no way to interrupt the device mid-reply.

This mode hands those three jobs (ASR, end-of-turn detection, TTS) to an
ElevenLabs agent, which already does VAD-based turn detection and
interruption handling server-side, and keeps the device as what it actually
is: a microphone and a speaker. The agent's own LLM is configured
separately (ours points at an OpenClaw custom-LLM webhook), so nothing in
this module decides *what* to say.

Wire shape, per the ElevenLabs Agents WebSocket protocol:

    gateway -> agent   {"user_audio_chunk": "<base64 pcm16>"}
    agent -> gateway   audio / interruption / ping / user_transcript /
                       agent_response / agent_response_complete /
                       conversation_initiation_metadata

End-of-turn is the one part of that protocol we cannot rely on: the audio
event's ``is_final`` is optional and defaults to false, and
``agent_response_complete`` is not guaranteed either. So a turn is closed
by whichever of the two arrives first, and failing both, by
:data:`TURN_IDLE_TIMEOUT_S`. A turn that never closes is not a cosmetic
bug -- the device stays in its speaking state, which mutes its own
microphone, so the conversation dies silently with counters at zero.

The device speaks Opus at :data:`DEVICE_SAMPLE_RATE`; the agent is
configured for ``pcm_16000`` in both directions, so the only conversion
needed is Opus<->PCM — no resampling. :func:`start` refuses to run if the
agent negotiates a different rate rather than silently sending audio the
far side will interpret at the wrong speed.

Structure mirrors :mod:`stackchan_mcp.beat.mode`: a module-level singleton
guarded by a lock, an owner token on the shared recording slot, and
background tasks that are cancelled on stop.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
from itertools import count
from typing import Any

from .audio_stream import is_recording, recording_owner, start_recording, stop_recording_if_owner
from .stt.audio_utils import DEVICE_SAMPLE_RATE, StreamingOpusDecoder
from .tts.orchestrator import send_pcm_stream

logger = logging.getLogger(__name__)

AGENTS_MODE_OWNER = "agents_mode"
AGENTS_MODE_OWNER_PREFIX = f"{AGENTS_MODE_OWNER}:"
_OWNER_GENERATIONS = count(1)

SIGNED_URL_ENDPOINT = "https://api.elevenlabs.io/v1/convai/conversation/get-signed-url"

#: The agent must be configured for this format in both directions. It is
#: what the device already produces and consumes, so a match means zero
#: resampling on the hot path.
REQUIRED_AUDIO_FORMAT = f"pcm_{DEVICE_SAMPLE_RATE}"

#: Uplink frames are queued rather than sent inline from the audio
#: callback: a slow socket must not block the device's receive loop. At
#: 60 ms per frame this is ~12 s of speech, far more than any turn, so
#: hitting the bound means the uplink is broken, not merely slow.
UPLINK_QUEUE_MAX_FRAMES = 200

#: Decoded agent audio waiting for the speaker. The sentinel ``None``
#: marks end-of-turn, which closes one ``send_pcm_stream`` call so the
#: device leaves the speaking state between turns.
PLAYBACK_QUEUE_MAX_CHUNKS = 400

#: How long a turn may go without new audio before we close it ourselves.
#: End-of-turn is advisory on the wire: ``is_final`` is optional and
#: defaults to false, and ``agent_response_complete`` may be the only
#: signal we get -- or neither may arrive. Without a bound here a turn
#: that is never closed wedges the whole mode: ``send_pcm_stream`` never
#: returns, so the device stays in the speaking state (mouth animating,
#: mic muted) and ``stop`` has a task it cannot join.
#:
#: Treat the value as dead air, not as a rare safety net. No agent
#: observed so far ends a turn with a marker, so every reply pays this in
#: full, and the device is deaf for all of it -- the mic comes back only
#: once the turn closes. The floor is the far side's inter-chunk gap
#: (~0.7 s observed, four chunks across a 1.9 s reply): drop below that
#: and one reply gets chopped into several spoken turns, each with its
#: own tts handshake.
TURN_IDLE_TIMEOUT_S = 1.5

#: Upper bound on tearing one conversation down. Cancelling a wedged
#: playback task must never be able to hang ``stop`` itself, because
#: ``stop`` holds the module lock and everything else queues behind it.
STOP_GRACE_S = 5.0

#: How long the farewell may keep playing after the far side hangs up.
#: ``end_call`` speaks its goodbye and *then* closes the socket, so the
#: last turn is still in the playback queue when the close lands: tear
#: down immediately and the agent is cut off mid-sentence. Must clear a
#: farewell plus the ``TURN_IDLE_TIMEOUT_S`` that closes it, and it only
#: ever delays releasing a device nobody else is waiting on.
FAREWELL_GRACE_S = 10.0

_mode: "AgentsConversation | None" = None
_mode_lock: asyncio.Lock | None = None


def _new_owner_token() -> str:
    return f"{AGENTS_MODE_OWNER}:{next(_OWNER_GENERATIONS)}"


def is_agents_mode_owner(owner: str | None) -> bool:
    return owner == AGENTS_MODE_OWNER or (
        owner is not None and owner.startswith(AGENTS_MODE_OWNER_PREFIX)
    )


def resolve_api_key() -> str:
    key = (
        os.getenv("STACKCHAN_ELEVENLABS_KEY") or os.getenv("ELEVENLABS_API_KEY") or ""
    ).strip()
    if not key:
        raise RuntimeError(
            "ElevenLabs API key is not configured. Set ELEVENLABS_API_KEY "
            "(or STACKCHAN_ELEVENLABS_KEY) in the gateway environment."
        )
    return key


def resolve_agent_id(explicit: Any = None) -> str:
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    env_id = (os.getenv("STACKCHAN_ELEVEN_AGENT_ID") or "").strip()
    if env_id:
        return env_id
    raise ValueError(
        "'agent_id' is required: pass it explicitly or set "
        "STACKCHAN_ELEVEN_AGENT_ID in the gateway environment."
    )


async def fetch_signed_url(agent_id: str, api_key: str, *, timeout_s: float = 20.0) -> str:
    """Get a short-lived signed WebSocket URL for a private agent.

    Public agents can be dialled with a bare ``agent_id``, but ours is
    private, and an unsigned connect to a private agent fails at the
    handshake with no useful diagnostic.
    """
    import aiohttp

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout_s)
    ) as session:
        async with session.get(
            SIGNED_URL_ENDPOINT,
            params={"agent_id": agent_id},
            headers={"xi-api-key": api_key},
        ) as resp:
            if resp.status != 200:
                body = (await resp.text())[:200]
                raise RuntimeError(
                    f"Could not get a signed ElevenLabs conversation URL: "
                    f"HTTP {resp.status} {body!r}"
                )
            payload = await resp.json()
    url = payload.get("signed_url")
    if not isinstance(url, str) or not url:
        raise RuntimeError("ElevenLabs returned no signed_url for this agent.")
    return url


class AgentsConversation:
    """One live conversation between the device and an ElevenLabs agent."""

    def __init__(self, gateway: Any, *, agent_id: str, api_key: str) -> None:
        self._gateway = gateway
        self._agent_id = agent_id
        self._api_key = api_key

        self._recording_owner = _new_owner_token()
        self._decoder: StreamingOpusDecoder | None = None
        self._uplink_queue: asyncio.Queue[bytes] = asyncio.Queue(
            maxsize=UPLINK_QUEUE_MAX_FRAMES
        )
        self._playback_queue: asyncio.Queue[bytes | None] = asyncio.Queue(
            maxsize=PLAYBACK_QUEUE_MAX_CHUNKS
        )

        self._ws: Any = None
        self._tasks: list[asyncio.Task[None]] = []
        self._playback_task: asyncio.Task[None] | None = None
        # Deliberately not in ``_tasks``: this one outlives the tasks it
        # is tearing down, so ``stop`` cancelling ``_tasks`` must not
        # reach it -- it is the thing calling ``stop``.
        self._teardown_task: asyncio.Task[None] | None = None
        self._stop_lock = asyncio.Lock()

        self._active = False
        self._session_id: str | None = None
        self._conversation_id: str | None = None
        self._started_at: float | None = None
        self._last_error: str | None = None
        self._last_user_transcript: str | None = None
        self._last_agent_response: str | None = None
        self._frames_uplinked = 0
        self._frames_dropped = 0
        self._audio_chunks_received = 0
        self._interruptions = 0
        self._turns_played = 0
        self._listen_rearms = 0
        self._last_turn_end_reason: str | None = None
        self._ended_reason: str | None = None
        self._unlogged_event_kinds: set[str] = set()

    @property
    def active(self) -> bool:
        return self._active

    def status(self) -> dict[str, Any]:
        return {
            "active": self._active,
            "agent_id": self._agent_id,
            "conversation_id": self._conversation_id,
            "started_at": self._started_at,
            "frames_uplinked": self._frames_uplinked,
            "frames_dropped": self._frames_dropped,
            "audio_chunks_received": self._audio_chunks_received,
            "interruptions": self._interruptions,
            "turns_played": self._turns_played,
            # Should track turns_played once the first reply lands. A
            # conversation whose uplink dies after the greeting shows
            # turns climbing while this stays put.
            "listen_rearms": self._listen_rearms,
            "last_turn_end_reason": self._last_turn_end_reason,
            # Why the conversation finished: "agent_hangup" when the far
            # side closed cleanly (the agent's ``end_call``), "far_side_
            # lost" when the socket died on us, "local_stop" for an
            # explicit agents_converse_stop. None while still running.
            "ended_reason": self._ended_reason,
            "last_user_transcript": self._last_user_transcript,
            "last_agent_response": self._last_agent_response,
            "last_error": self._last_error,
        }

    # --- lifecycle --------------------------------------------------

    def _ensure_device_ready(self) -> None:
        if not self._gateway.esp32.device_connected:
            raise RuntimeError("No ESP32 device connected; cannot start a conversation.")
        connection = getattr(self._gateway.esp32, "connection", None)
        proto_version = getattr(connection, "protocol_version", 1)
        if proto_version != 1:
            raise RuntimeError(
                f"Conversation mode requires WebSocket protocol v1, but the "
                f"connected device negotiated v{proto_version}."
            )

    async def start(self) -> dict[str, Any]:
        import websockets

        self._ensure_device_ready()

        owner = recording_owner()
        if is_recording() and owner != self._recording_owner:
            raise RuntimeError(
                f"audio capture is already active (owner={owner}); stop it "
                "before starting conversation mode"
            )

        signed_url = await fetch_signed_url(self._agent_id, self._api_key)
        self._ws = await websockets.connect(signed_url, max_size=None)

        try:
            await self._ws.send(
                json.dumps({"type": "conversation_initiation_client_data"})
            )
            await self._await_initiation_metadata()
            self._arm_capture()
            # profile="voice" selects the firmware's speech AFE path, so
            # the agent hears echo-cancelled speech instead of its own
            # voice; beat mode uses profile="raw" for the opposite reason.
            #
            # It does *not* keep the mic open across playback. Measured
            # against firmware 2.2.6: the device stops sending frames the
            # moment it enters the speaking state, and comes back to idle
            # rather than to listening when the turn ends. So this
            # listen.start only covers the window before the agent's first
            # reply -- ``_rearm_listening`` owns every window after it.
            await self._gateway.esp32.send_listen_state(
                "start", mode="manual", profile="voice"
            )
        except BaseException:
            self._release_recording_slot()
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None
            raise

        self._active = True
        self._started_at = asyncio.get_running_loop().time()
        self._tasks = [
            asyncio.create_task(self._downlink_loop(), name="agents-downlink"),
            asyncio.create_task(self._uplink_loop(), name="agents-uplink"),
            asyncio.create_task(self._playback_loop(), name="agents-playback"),
        ]
        logger.info(
            "agents conversation started: agent=%s conversation=%s",
            self._agent_id,
            self._conversation_id,
        )
        return self.status()

    async def _await_initiation_metadata(self, *, timeout_s: float = 20.0) -> None:
        """Read the opening metadata frame and hard-check the audio format.

        A mismatch here is silent corruption downstream: the far side
        would interpret our 16 kHz bytes at its own rate, so the user
        hears a chipmunk and the agent hears mud. Fail loudly instead.
        """
        raw = await asyncio.wait_for(self._ws.recv(), timeout=timeout_s)
        event = json.loads(raw)
        if event.get("type") != "conversation_initiation_metadata":
            raise RuntimeError(
                f"Expected conversation_initiation_metadata first, got "
                f"{event.get('type')!r}"
            )
        meta = event.get("conversation_initiation_metadata_event", {})
        self._conversation_id = meta.get("conversation_id")
        user_fmt = meta.get("user_input_audio_format")
        agent_fmt = meta.get("agent_output_audio_format")
        if user_fmt != REQUIRED_AUDIO_FORMAT or agent_fmt != REQUIRED_AUDIO_FORMAT:
            raise RuntimeError(
                f"Agent {self._agent_id} negotiated user_input={user_fmt!r} / "
                f"agent_output={agent_fmt!r}, but this device needs "
                f"{REQUIRED_AUDIO_FORMAT!r} in both directions. Change the "
                "agent's audio format in its ElevenLabs configuration."
            )

    def _arm_capture(self) -> None:
        connection = self._gateway.esp32.connection
        session_id = getattr(connection, "session_id", "") if connection else ""
        self._session_id = session_id
        self._decoder = StreamingOpusDecoder()
        start_recording(
            session_id,
            owner=self._recording_owner,
            frame_hook=self._on_opus_frame,
            buffer_frames=False,
        )

    def _release_recording_slot(self) -> None:
        stop_recording_if_owner(self._recording_owner)

    async def stop(self) -> dict[str, Any]:
        async with self._stop_lock:
            if not self._active:
                return self.status()
            self._active = False
            # Set only when the far side has not already explained itself,
            # so a hangup keeps its reason instead of every conversation
            # reporting the teardown call that finished it.
            if self._ended_reason is None:
                self._ended_reason = "local_stop"

            for task in self._tasks:
                task.cancel()
            if self._tasks:
                # Bounded join, and a task that refuses to die is abandoned
                # rather than awaited: ``stop`` runs under the module lock,
                # so hanging here freezes every later call -- which is the
                # exact failure this replaces.
                await asyncio.wait(self._tasks, timeout=STOP_GRACE_S)
            self._tasks = []
            await self._cancel_playback()

            self._release_recording_slot()
            with contextlib.suppress(Exception):
                await self._gateway.esp32.send_listen_state("stop")
            if self._ws is not None:
                with contextlib.suppress(Exception):
                    await self._ws.close()
                self._ws = None

            logger.info(
                "agents conversation stopped: uplinked=%d dropped=%d turns=%d",
                self._frames_uplinked,
                self._frames_dropped,
                self._turns_played,
            )
            return self.status()

    # --- uplink (device mic -> agent) -------------------------------

    def _on_opus_frame(self, frame: bytes) -> None:
        """Audio-callback side: enqueue only, never block the read loop."""
        try:
            self._uplink_queue.put_nowait(frame)
        except asyncio.QueueFull:
            self._frames_dropped += 1
            if self._frames_dropped % 50 == 1:
                logger.warning(
                    "agents uplink queue full; dropped %d frames so far",
                    self._frames_dropped,
                )

    async def _uplink_loop(self) -> None:
        while True:
            frame = await self._uplink_queue.get()
            try:
                pcm = self._decoder.decode_frame(frame) if self._decoder else b""
            except Exception as exc:
                self._last_error = f"opus decode failed: {exc}"
                logger.warning("agents uplink decode failed: %s", exc)
                continue
            if not pcm:
                continue
            payload = json.dumps(
                {"user_audio_chunk": base64.b64encode(pcm).decode("ascii")}
            )
            try:
                await self._ws.send(payload)
            except Exception as exc:
                self._last_error = f"uplink send failed: {exc}"
                logger.warning("agents uplink send failed: %s", exc)
                return
            self._frames_uplinked += 1

    # --- downlink (agent -> device speaker) -------------------------

    async def _downlink_loop(self) -> None:
        while True:
            try:
                raw = await self._ws.recv()
            except Exception as exc:
                logger.info("agents downlink closed: %s", exc)
                self._note_far_side_hangup(exc)
                return
            try:
                event = json.loads(raw)
            except Exception:
                continue
            await self._handle_event(event)

    async def _handle_event(self, event: dict[str, Any]) -> None:
        """Dispatch one server event. Kept side-effect-light for testing."""
        kind = event.get("type")

        if kind == "audio":
            audio = event.get("audio_event", {})
            b64 = audio.get("audio_base_64")
            if isinstance(b64, str) and b64:
                self._audio_chunks_received += 1
                await self._enqueue_playback(base64.b64decode(b64))
            if audio.get("is_final"):
                await self._close_turn("is_final")
            return

        if kind == "interruption":
            # The user talked over the agent. Whatever is queued is now
            # stale: drop it and cut playback, otherwise the device keeps
            # reciting a reply the far side has already retracted.
            self._interruptions += 1
            self._drain_playback_queue()
            await self._cancel_playback()
            return

        if kind == "ping":
            event_id = event.get("ping_event", {}).get("event_id")
            with contextlib.suppress(Exception):
                await self._ws.send(json.dumps({"type": "pong", "event_id": event_id}))
            return

        if kind == "user_transcript":
            self._last_user_transcript = event.get("user_transcription_event", {}).get(
                "user_transcript"
            )
            return

        if kind == "agent_response":
            self._last_agent_response = event.get("agent_response_event", {}).get(
                "agent_response"
            )
            return

        if kind == "agent_response_correction":
            self._last_agent_response = event.get(
                "agent_response_correction_event", {}
            ).get("corrected_agent_response")
            return

        if kind == "agent_response_complete":
            # The far side's explicit "that reply is finished" signal, and
            # the only end-of-turn marker we get when the audio events
            # never carry ``is_final`` (it is optional, default false).
            await self._close_turn("agent_response_complete")
            return

        if kind == "client_error" or kind == "error":
            self._last_error = json.dumps(event)[:300]
            logger.warning("agents error event: %s", self._last_error)
            return

        # Anything we do not handle is logged once per kind. End-of-turn
        # on this protocol is advisory, so when a turn misbehaves the
        # first question is always "what did the far side actually send?"
        if isinstance(kind, str) and kind not in self._unlogged_event_kinds:
            self._unlogged_event_kinds.add(kind)
            logger.info("agents event kind not handled: %s", kind)

    async def _close_turn(self, reason: str) -> None:
        """Queue the sentinel that ends one ``send_pcm_stream`` call."""
        self._last_turn_end_reason = reason
        await self._enqueue_playback(None)

    async def _enqueue_playback(self, chunk: bytes | None) -> None:
        try:
            self._playback_queue.put_nowait(chunk)
        except asyncio.QueueFull:
            logger.warning("agents playback queue full; dropping agent audio")

    def _drain_playback_queue(self) -> None:
        while True:
            try:
                self._playback_queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    async def _cancel_playback(self) -> None:
        task = self._playback_task
        self._playback_task = None
        if task is None or task.done():
            return
        task.cancel()
        # Bounded join: a playback task blocked on device I/O must not be
        # able to hold teardown open. ``asyncio.wait`` returns on timeout
        # instead of waiting for the cancellation itself to land.
        await asyncio.wait({task}, timeout=STOP_GRACE_S)

    def _note_far_side_hangup(self, exc: BaseException) -> None:
        """React to the conversation socket closing under us.

        There is no "conversation over" event on this protocol: the agent
        speaks its farewell and the server then closes the socket, so this
        close *is* the hangup. Nothing else reaps the mode, and without a
        teardown here the mic stays armed forever -- ``listen()`` and beat
        mode would refuse to run until the gateway is restarted.

        A close during ``stop`` is our own doing and is ignored: ``stop``
        clears ``active`` before it touches the socket.
        """
        if not self._active or self._teardown_task is not None:
            return
        # websockets raises ConnectionClosedOK only for a clean close
        # handshake, which is what ``end_call`` produces. Anything else
        # is a drop, and the distinction is the difference between "he
        # said goodbye" and "the network ate him" when reading status.
        graceful = type(exc).__name__ == "ConnectionClosedOK"
        self._ended_reason = "agent_hangup" if graceful else "far_side_lost"
        if not graceful:
            self._last_error = f"connection closed: {exc}"
        logger.info(
            "agents conversation ended by far side: reason=%s", self._ended_reason
        )
        self._teardown_task = asyncio.create_task(
            self._teardown_after_hangup(), name="agents-hangup-teardown"
        )

    async def _teardown_after_hangup(self) -> None:
        """Let the farewell finish, then release the device.

        Routed through the module-level ``stop_agents_conversation`` rather
        than ``self.stop`` so the global ``_mode`` slot is cleared too: a
        stale entry would keep reporting a conversation that is long gone.
        """
        with contextlib.suppress(Exception):
            await self._drain_playback(FAREWELL_GRACE_S)
        with contextlib.suppress(Exception):
            await stop_agents_conversation()

    async def _drain_playback(self, timeout_s: float) -> None:
        """Wait until nothing is queued for, or playing on, the speaker."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while loop.time() < deadline:
            task = self._playback_task
            if self._playback_queue.empty() and (task is None or task.done()):
                return
            await asyncio.sleep(0.05)
        logger.warning(
            "agents farewell still playing after %.1fs; releasing the device anyway",
            timeout_s,
        )

    async def _rearm_listening(self) -> None:
        """Put the device back into listening mode after a turn.

        The firmware leaves the listening state when it starts speaking
        and returns to *idle*, not to listening, once the turn ends. The
        single ``listen.start`` that :meth:`start` issues therefore only
        covers the window before the agent's first reply -- about half a
        second when the agent opens with a greeting. Without re-arming
        here the uplink goes silent for the rest of the session: the
        agent talks, the user answers, and nothing is ever heard.

        A failure is logged rather than raised. The outbound direction
        still works, and letting this kill the playback loop would turn a
        deaf conversation into a mute one as well.
        """
        if not self._active:
            # ``stop`` clears the flag before it cancels these tasks, so
            # this keeps teardown from re-opening the mic on its way out.
            return
        try:
            await self._gateway.esp32.send_listen_state(
                "start", mode="manual", profile="voice"
            )
        except Exception as exc:
            self._last_error = f"listen re-arm failed: {exc}"
            logger.warning("agents listen re-arm failed: %s", exc)
            return
        self._listen_rearms += 1

    async def _playback_loop(self) -> None:
        """Play one agent turn per ``send_pcm_stream`` call.

        Bracketing each turn separately (rather than holding one endless
        stream open) is what lets the device return to idle between
        replies, and gives interruption something to cancel that unwinds
        the tts start/stop handshake cleanly.

        Re-arming the mic is bracketed to the same boundary: this is the
        one place every turn passes through on its way back to idle,
        whether it ended on a marker, on the idle timeout, or on an
        interruption cancelling playback mid-sentence.
        """
        while True:
            first = await self._playback_queue.get()
            if first is None:
                continue
            self._playback_task = asyncio.create_task(
                self._play_turn(first), name="agents-play-turn"
            )
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._playback_task
            self._playback_task = None
            await self._rearm_listening()

    async def _play_turn(self, first_chunk: bytes) -> None:
        async def chunks():
            yield first_chunk
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        self._playback_queue.get(), TURN_IDLE_TIMEOUT_S
                    )
                except asyncio.TimeoutError:
                    # No more audio and no end-of-turn marker ever came.
                    # Close the turn regardless: leaving it open keeps the
                    # device speaking forever, which mutes its own mic.
                    self._last_turn_end_reason = "idle_timeout"
                    logger.warning(
                        "agents turn closed by idle timeout after %.1fs "
                        "with no end-of-turn marker",
                        TURN_IDLE_TIMEOUT_S,
                    )
                    return
                if chunk is None:
                    return
                yield chunk

        try:
            await send_pcm_stream(
                self._gateway,
                chunks(),
                source_rate=DEVICE_SAMPLE_RATE,
                source_label=f"elevenlabs-agent:{self._conversation_id or '?'}",
            )
            self._turns_played += 1
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._last_error = f"playback failed: {exc}"
            logger.warning("agents playback failed: %s", exc)


# --- module-level lifecycle ------------------------------------------


def _get_mode_lock() -> asyncio.Lock:
    global _mode_lock
    if _mode_lock is None:
        _mode_lock = asyncio.Lock()
    return _mode_lock


async def start_agents_conversation(
    gateway: Any,
    *,
    agent_id: Any = None,
) -> dict[str, Any]:
    global _mode
    async with _get_mode_lock():
        if _mode is not None and _mode.active:
            return _mode.status()
        mode = AgentsConversation(
            gateway,
            agent_id=resolve_agent_id(agent_id),
            api_key=resolve_api_key(),
        )
        _mode = mode
        try:
            return await mode.start()
        except BaseException:
            if _mode is mode:
                _mode = None
            raise


async def stop_agents_conversation() -> dict[str, Any]:
    global _mode
    async with _get_mode_lock():
        if _mode is None:
            return {"active": False}
        status = await _mode.stop()
        _mode = None
        return status


def get_agents_conversation_snapshot() -> dict[str, Any]:
    if _mode is None:
        return {"active": False}
    return _mode.status()
