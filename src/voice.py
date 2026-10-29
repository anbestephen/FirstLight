"""
Push-to-talk commander questions via Muse Voice Transcribe (`muse-voice-transcribe-1.0`).

The browser streams microphone audio (16-bit little-endian mono PCM at 24 kHz) over a local
WebSocket; this module relays it to Meta's realtime ASR endpoint in 80 ms frames and streams
the transcript back word by word. The API key stays on the laptop. The final transcript is
then asked to the commander agent like a typed question.

Browser -> server: binary PCM frames, then the text frame `{"type": "end"}` on release.
Server -> browser: `{"type": "ready"}`, `{"type": "transcript", "text", "final"}`, and from the
caller `{"type": "done", ...}` or `{"type": "error", "message"}`.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable

import websockets

META_ASR_URL = "wss://api.meta.ai/v1/asr/realtime"
VOICE_MODEL = "muse-voice-transcribe-1.0"
SAMPLE_RATE_HZ = 24_000
FRAME_BYTES = SAMPLE_RATE_HZ * 2 * 80 // 1000  # 80 ms of PCM16 = 3,840 bytes
HANDSHAKE_TIMEOUT_S = 10.0


async def relay_transcription(
    client: Any,  # fastapi.WebSocket, already accepted
    api_key: str,
    *,
    connect: Callable[..., Any] = websockets.connect,
    url: str = META_ASR_URL,
) -> dict[str, Any]:
    """Relay one push-to-talk utterance; returns the final transcript and audio timings."""
    async with connect(url, open_timeout=5) as upstream:
        await upstream.send(
            json.dumps(
                {
                    "authorization": {"accessToken": f"Bearer {api_key}"},
                    "audioEncoding": "PCM_24KHZ",
                    "model": VOICE_MODEL,
                    "mode": "PUSH_TO_TALK",
                }
            )
        )
        ack = json.loads(await asyncio.wait_for(upstream.recv(), HANDSHAKE_TIMEOUT_S))
        if "sessionId" not in ack:
            raise RuntimeError(f"Muse Voice handshake rejected: {ack}")
        await client.send_json({"type": "ready"})

        finals: list[str] = []
        audio_bytes = 0
        released_at: float | None = None

        async def pump_microphone() -> None:
            nonlocal audio_bytes, released_at
            pending = bytearray()
            while True:
                msg = await client.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    pending.extend(msg["bytes"])
                    while len(pending) >= FRAME_BYTES:
                        await upstream.send(bytes(pending[:FRAME_BYTES]))
                        del pending[:FRAME_BYTES]
                        audio_bytes += FRAME_BYTES
                elif msg.get("text") and json.loads(msg["text"]).get("type") == "end":
                    break
            if pending:
                await upstream.send(bytes(pending))
                audio_bytes += len(pending)
            released_at = time.perf_counter()
            # Half-close: the server flushes the last words, then closes with code 1000.
            await upstream.send(json.dumps({"type": "endStream"}))

        microphone = asyncio.create_task(pump_microphone())
        try:
            async for raw in upstream:
                if isinstance(raw, bytes):
                    continue
                event = json.loads(raw)
                if event.get("type") == "transcript":
                    text = str(event.get("transcript", "")).strip()
                    if event.get("final"):
                        finals.append(text)
                        text = ""
                    await client.send_json(
                        {
                            "type": "transcript",
                            "text": " ".join([*finals, text]).strip(),
                            "final": bool(event.get("final")),
                        }
                    )
                elif event.get("type") == "error":
                    raise RuntimeError(f"Muse Voice error: {event.get('message')}")
        finally:
            if not microphone.done():
                microphone.cancel()
        await asyncio.gather(microphone, return_exceptions=True)

    return {
        "transcript": " ".join(finals).strip(),
        "audio_ms": round(audio_bytes / (SAMPLE_RATE_HZ * 2) * 1000),
        "finalize_ms": round((time.perf_counter() - released_at) * 1000) if released_at else None,
        "model": VOICE_MODEL,
    }
