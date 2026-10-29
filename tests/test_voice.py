"""Muse Voice Transcribe push-to-talk relay (src/voice.py) against a fake Meta ASR socket."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi.testclient import TestClient

from src.voice import FRAME_BYTES, VOICE_MODEL, relay_transcription


class FakeUpstream:
    """Meta realtime ASR: acks the handshake, emits partial/final transcripts, closes after endStream."""

    def __init__(self) -> None:
        self.sent: list[Any] = []
        self.events: asyncio.Queue[str | None] = asyncio.Queue()

    async def __aenter__(self) -> "FakeUpstream":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def send(self, data: Any) -> None:
        self.sent.append(data)
        if isinstance(data, bytes) and sum(isinstance(d, bytes) for d in self.sent) == 1:
            await self.events.put(json.dumps({"type": "transcript", "transcript": "Get an", "final": False}))
        if isinstance(data, str) and json.loads(data).get("type") == "endStream":
            await self.events.put(json.dumps({"type": "transcript", "transcript": "Get an ambulance to the worst hit block.", "final": True}))
            await self.events.put(None)

    async def recv(self) -> str:
        return json.dumps({"sessionId": "test-session"})

    def __aiter__(self) -> "FakeUpstream":
        return self

    async def __anext__(self) -> str:
        event = await self.events.get()
        if event is None:
            raise StopAsyncIteration
        return event


class FakeBrowser:
    def __init__(self, frames: list[dict[str, Any]]) -> None:
        self.incoming = list(frames)
        self.outgoing: list[dict[str, Any]] = []

    async def receive(self) -> dict[str, Any]:
        await asyncio.sleep(0)
        return self.incoming.pop(0)

    async def send_json(self, data: dict[str, Any]) -> None:
        self.outgoing.append(data)


async def test_relay_streams_microphone_frames_and_returns_final_transcript() -> None:
    upstream = FakeUpstream()
    audio = bytes(range(256)) * 40  # 10,240 bytes: two full 80 ms frames plus a remainder
    browser = FakeBrowser(
        [
            {"type": "websocket.receive", "bytes": audio[:5000]},
            {"type": "websocket.receive", "bytes": audio[5000:]},
            {"type": "websocket.receive", "text": json.dumps({"type": "end"})},
        ]
    )

    summary = await relay_transcription(browser, "test-key", connect=lambda *a, **k: upstream)

    handshake = json.loads(upstream.sent[0])
    assert handshake == {
        "authorization": {"accessToken": "Bearer test-key"},
        "audioEncoding": "PCM_24KHZ",
        "model": VOICE_MODEL,
        "mode": "PUSH_TO_TALK",
    }
    frames = [d for d in upstream.sent if isinstance(d, bytes)]
    assert [len(f) for f in frames] == [FRAME_BYTES, FRAME_BYTES, len(audio) - 2 * FRAME_BYTES]
    assert b"".join(frames) == audio
    assert json.loads(upstream.sent[-1]) == {"type": "endStream"}

    assert browser.outgoing[0] == {"type": "ready"}
    assert {"type": "transcript", "text": "Get an", "final": False} in browser.outgoing
    assert browser.outgoing[-1] == {"type": "transcript", "text": "Get an ambulance to the worst hit block.", "final": True}
    assert summary["transcript"] == "Get an ambulance to the worst hit block."
    assert summary["audio_ms"] == round(len(audio) / 48)
    assert summary["model"] == VOICE_MODEL


def test_voice_endpoint_says_to_type_when_no_api_key(monkeypatch) -> None:
    from src.app import app, state

    monkeypatch.setattr(state.router, "api_key", "")
    with TestClient(app).websocket_connect("/api/voice/stream") as ws:
        msg = ws.receive_json()
    assert msg["type"] == "error"
    assert "Type the question" in msg["message"]
