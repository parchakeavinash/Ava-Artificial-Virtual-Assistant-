import asyncio
import base64
import threading
import time
from queue import Empty, Full, Queue

from sarvamai import (
    AsyncSarvamAI,
    RealtimeAudioInput,
    RealtimePing,
)

from app.config import settings


# Streamlit reruns recreate the script namespace, but this module stays loaded.
# WebRTC audio callbacks must send to this object, not a stale script global.
_active_stt = None


def get_active_stt():
    return _active_stt


class SarvamRealtimeSTT:
    """
    Long-lived realtime connection to Sarvam STT.

    WebRTC audio:
        WebRTC callback
            ↓
        audio_queue
            ↓
        async sender
            ↓
        Sarvam WebSocket

    Sarvam events:
        WebSocket
            ↓
        event_queue
            ↓
        Streamlit UI
    """

    def __init__(self):

        self.client = AsyncSarvamAI(
            api_subscription_key=settings.SARVAM_API_KEY
        )

        self.audio_queue = Queue(maxsize=200)
        self.event_queue = Queue()

        self.running = False
        self.thread = None

    def bind(self):

        global _active_stt

        _active_stt = self

    # ---------------------------------------------------------
    # START
    # ---------------------------------------------------------

    def start(self):

        self.bind()

        if self.running:
            return

        print(f"[STT] START called {id(self)}")

        self.running = True

        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
        )

        self.thread.start()

    # ---------------------------------------------------------
    # BACKGROUND EVENT LOOP
    # ---------------------------------------------------------

    def _run(self):

        try:

            asyncio.run(
                self._session_loop()
            )

        except Exception as exc:

            print(
                f"[STT] CONNECTION ERROR: {exc}"
            )

            self.event_queue.put(
                {
                    "type": "error",
                    "message": str(exc),
                }
            )

        finally:

            self.running = False

            print("[STT] connection stopped")

    async def _session_loop(self):

        while self.running:

            try:

                await self._connect()

            except Exception as exc:

                print(
                    f"[STT] session error: {exc}"
                )

                self.event_queue.put(
                    {
                        "type": "error",
                        "message": str(exc),
                    }
                )

            if self.running:

                print("[STT] reconnecting in 1s")

                await asyncio.sleep(1)

    # ---------------------------------------------------------
    # SARVAM CONNECTION
    # ---------------------------------------------------------

    async def _connect(self):

        async with (
            self.client
            .speech_to_text_realtime_streaming
            .connect(
                language_code="en-IN",
                model="saaras:v3-realtime",
                mode="transcribe",
                stream_type="balanced",
                prompt="AI, RAG, retrieval augmented generation, LLM, Python, MCP, database, vector",
                endpointing="vad",
                silence_duration_ms="1000",
                encoding="linear16",
                sample_rate="16000",
            )
            as ws
        ):

            print("[STT] Sarvam connected")

            sender = asyncio.create_task(
                self._send_audio(ws)
            )

            receiver = asyncio.create_task(
                self._receive_events(ws)
            )

            done, pending = await asyncio.wait(
                {sender, receiver},
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()

            for task in done:
                exc = task.exception() if not task.cancelled() else None
                if exc:
                    raise exc

    # ---------------------------------------------------------
    # SEND AUDIO
    # ---------------------------------------------------------

    async def _send_audio(self, ws):

        print("[STT] sender started")

        # 16 kHz × 16-bit × mono = 32000 bytes/sec
        # 3200 bytes ≈ 100 ms (Sarvam realtime recommendation)
        TARGET_BYTES = 3200

        buffer = bytearray()
        last_ping = time.monotonic()

        while self.running:

            try:

                audio = await asyncio.to_thread(
                    self.audio_queue.get,
                    True,
                    0.1,
                )

            except Empty:

                audio = b""

            if audio is None:
                break

            if audio:
                buffer.extend(audio)

            while len(buffer) >= TARGET_BYTES:

                chunk = bytes(
                    buffer[:TARGET_BYTES]
                )

                del buffer[:TARGET_BYTES]

                print(
                    f"[STT] sending chunk: "
                    f"{len(chunk)} bytes"
                )

                encoded_audio = base64.b64encode(
                    chunk
                ).decode("utf-8")

                await ws.send_realtime_audio_input(
                    RealtimeAudioInput(
                        audio=encoded_audio
                    )
                )

            now = time.monotonic()

            if now - last_ping >= 15:

                await ws.send_realtime_ping(
                    RealtimePing()
                )

                last_ping = now

                print("[STT] ping")

    # ---------------------------------------------------------
    # RECEIVE SARVAM EVENTS
    # ---------------------------------------------------------

    async def _receive_events(self, ws):

        print("[STT] receiver started")

        async for message in ws:

            print(
                f"[STT] Sarvam event: "
                f"{message.event}"
            )

            # ---------------------------------------------
            # Partial transcript
            # ---------------------------------------------

            if message.event == "transcript.partial":

                print(
                    f"[STT] PARTIAL: "
                    f"{message.text}"
                )

                self.event_queue.put(
                    {
                        "type": "partial",
                        "text": message.text,
                    }
                )

            # ---------------------------------------------
            # Final transcript
            # ---------------------------------------------

            elif message.event == "transcript.final":

                print(
                    f"[STT] FINAL: "
                    f"{message.text}"
                )

                self.event_queue.put(
                    {
                        "type": "final",
                        "text": message.text,
                    }
                )

            # ---------------------------------------------
            # Error
            # ---------------------------------------------

            elif message.event == "error":

                print(
                    f"[STT] ERROR: "
                    f"{message.code} - "
                    f"{message.message}"
                )

                self.event_queue.put(
                    {
                        "type": "error",
                        "code": message.code,
                        "message": message.message,
                        "fatal": message.is_fatal,
                    }
                )

                if message.is_fatal:
                    break

    # ---------------------------------------------------------
    # AUDIO INPUT
    # ---------------------------------------------------------

    def send_audio(self, audio: bytes):

        try:

            self.audio_queue.put_nowait(audio)

        except Full:

            try:
                self.audio_queue.get_nowait()
            except Empty:
                pass

            self.audio_queue.put_nowait(audio)

    # ---------------------------------------------------------
    # GET EVENTS
    # ---------------------------------------------------------

    def get_events(self):

        events = []

        while True:

            try:

                events.append(
                    self.event_queue.get_nowait()
                )

            except Empty:

                break

        return events

    # ---------------------------------------------------------
    # STOP
    # ---------------------------------------------------------

    def stop(self):

        global _active_stt

        if not self.running:
            return

        print("[STT] STOP")

        self.running = False

        if _active_stt is self:
            _active_stt = None

        self.audio_queue.put(None)
