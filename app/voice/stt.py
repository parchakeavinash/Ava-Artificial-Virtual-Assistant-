import asyncio
import json
import logging
from queue import Empty, Queue
import threading
from typing import Any

import websockets

from app.config.settings import settings

logger = logging.getLogger(__name__)

# Sarvam Realtime WebSocket URL
_WS_URL = "wss://api.sarvam.ai/streaming/v1/ws"

# Vocabulary context prompt for tech/AI terms
_REALTIME_PROMPT = (
    "Ava AI voice assistant. User may speak about technology, programming, "
    "LLM, RAG, AI agents, MCP, LangGraph, Streamlit, Python, GPUs, Nvidia, "
    "SpaceX, Starlink, email, calendar, weather, or current news."
)


class SarvamRealtimeSTT:
    """
    Live streaming Speech-to-Text using Sarvam AI WebSocket API.

    Audio from the microphone is pushed into this instance via `push_audio()`.
    Transcript events from Sarvam are drained via `get_events()`.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        stream_type: str = "balanced",
        silence_duration_ms: str = "1000",
    ):
        self.sample_rate = sample_rate
        self.stream_type = stream_type
        self.silence_duration_ms = silence_duration_ms

        self.audio_queue: Queue[bytes] = Queue()
        self.event_queue: Queue[dict[str, Any]] = Queue()

        self.running = False
        self.thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self):
        """Start the background WebSocket worker."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the background worker."""
        self.running = False

    def push_audio(self, audio_bytes: bytes):
        """Push raw PCM16 (16kHz mono) audio bytes."""
        if self.running:
            self.audio_queue.put_nowait(audio_bytes)

    def get_events(self) -> list[dict[str, Any]]:
        """Drain all available events from the queue."""
        events = []
        while True:
            try:
                events.append(self.event_queue.get_nowait())
            except Empty:
                break
        return events

    def _run_event_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._websocket_client())
        finally:
            self._loop.close()

    async def _websocket_client(self):
        headers = {"api-subscription-key": settings.SARVAM_API_KEY}

        params = {
            "encoding": "linear16",
            "sample_rate": str(self.sample_rate),
            "stream_type": self.stream_type,
            "endpointing": "vad",
            "silence_duration_ms": self.silence_duration_ms,
            "prompt": _REALTIME_PROMPT,
        }

        query_str = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{_WS_URL}?{query_str}"

        while self.running:
            try:
                async with websockets.connect(
                    url,
                    additional_headers=headers,
                    ping_interval=20,
                    ping_timeout=20,
                ) as ws:
                    send_task = asyncio.create_task(self._sender(ws))
                    recv_task = asyncio.create_task(self._receiver(ws))

                    done, pending = await asyncio.wait(
                        [send_task, recv_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for t in pending:
                        t.cancel()

            except Exception as e:
                if self.running:
                    print(f"[STT] WebSocket error: {e}, reconnecting in 2s...")
                    self.event_queue.put_nowait({"type": "error", "message": str(e)})
                    await asyncio.sleep(2)

    async def _sender(self, ws):
        while self.running:
            try:
                chunk = self.audio_queue.get_nowait()
            except Empty:
                await asyncio.sleep(0.02)
                continue

            try:
                await ws.send(chunk)
            except Exception:
                break

    async def _receiver(self, ws):
        while self.running:
            try:
                msg = await ws.recv()
                data = json.loads(msg)
                event_type = data.get("type", "")

                if event_type == "transcript.partial":
                    text = data.get("text", "").strip()
                    if text:
                        self.event_queue.put_nowait({"type": "partial", "text": text})

                elif event_type == "transcript.final":
                    text = data.get("text", "").strip()
                    if text:
                        self.event_queue.put_nowait({"type": "final", "text": text})

                elif event_type == "error":
                    self.event_queue.put_nowait({
                        "type": "error",
                        "message": data.get("message", "Unknown error"),
                    })

            except Exception:
                break
