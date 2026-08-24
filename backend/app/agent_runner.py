import base64
import threading
import time
from queue import Empty, Queue

from app.gemini_agent import GeminiAgent
from app.sarvam_tts import SarvamTTS


# How long to wait (seconds) after the last transcript.final before
# treating the accumulated text as one complete utterance and
# sending it to the LLM. Increase if your speech has longer pauses.
_DEBOUNCE_SECONDS = 1.5


class AgentRunner:
    """
    Background thread that connects STT → LLM → TTS.

    Why a separate thread?
    ─────────────────────
    The STT thread must never be blocked waiting for Gemini or TTS —
    both can take 1-3 seconds. If STT had to wait, audio frames would
    back up and we'd lose speech.

    Instead:
      STT thread  →  push_transcript(text)  →  transcript_queue
      AgentRunner thread reads transcript_queue → debounces → calls Gemini → calls TTS
                         → puts result in tts_queue
      Streamlit fragment drains tts_queue → shows text + plays audio

    Why debounce?
    ─────────────
    Sarvam STT in stream_type="fast" fires transcript.final on every
    micro-pause — so "What is RAG?" arrives as three separate finals:
      "What"  →  "is"  →  "RAG?"
    We buffer them and only forward to Gemini after 1.5s of silence,
    so the full sentence is sent as one coherent question.

    Queues used:
      transcript_queue : str items (one per final transcript fragment)
      tts_queue        : dict items {"text": ..., "audio_b64": ...}
    """

    def __init__(self):

        self.transcript_queue: Queue[str] = Queue()
        self.tts_queue: Queue[dict] = Queue()

        self.running = False
        self.thread: threading.Thread | None = None

        # One GeminiAgent per session — holds multi-turn history
        self.agent = GeminiAgent()

        # TTS converter
        self.tts = SarvamTTS()

        # Debounce state — accessed only inside _run (single thread)
        self._buffer: list[str] = []
        self._last_fragment_time: float = 0.0

    # ---------------------------------------------------------
    # LIFECYCLE
    # ---------------------------------------------------------

    def start(self):
        """Start the background processing thread."""

        if self.running:
            return

        print("[AGENT] Starting agent runner thread")

        self.running = True

        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="AgentRunner",
        )

        self.thread.start()

    def stop(self):
        """Stop the background thread gracefully."""

        if not self.running:
            return

        print("[AGENT] Stopping agent runner")

        self.running = False

        # Unblock the get() call in _run
        self.transcript_queue.put(None)

    # ---------------------------------------------------------
    # INPUT — called by STT event handler
    # ---------------------------------------------------------

    def push_transcript(self, text: str):
        """
        Called from the Streamlit main thread when STT produces
        a final transcript fragment. Non-blocking — just enqueues.
        """

        if not text.strip():
            return

        print(f"[AGENT] Fragment received: {text!r}")

        self.transcript_queue.put_nowait(text)

    # ---------------------------------------------------------
    # OUTPUT — polled by Streamlit fragment
    # ---------------------------------------------------------

    def get_results(self) -> list[dict]:
        """
        Drain tts_queue and return all ready results.
        Each item: {"text": str, "audio_b64": str}
        Called by the Streamlit fragment every 0.5 s.
        """

        results = []

        while True:

            try:
                results.append(self.tts_queue.get_nowait())
            except Empty:
                break

        return results

    # ---------------------------------------------------------
    # BACKGROUND LOOP
    # ---------------------------------------------------------

    def _run(self):
        """
        Main loop — runs in daemon thread.

        Each iteration does one of two things:
          A) A new fragment arrived  → add to buffer, update timer.
          B) No new fragment + debounce expired → flush buffer to LLM.
        """

        print("[AGENT] Runner thread started")

        while self.running:

            try:
                # Short timeout so we can check the debounce timer
                text = self.transcript_queue.get(timeout=0.1)
            except Empty:
                text = None

            # Stop sentinel
            if text is None and not self.running:
                break

            now = time.monotonic()

            if text is not None:

                # Accumulate fragment into buffer
                self._buffer.append(text)
                self._last_fragment_time = now

            elif (
                self._buffer
                and (now - self._last_fragment_time) >= _DEBOUNCE_SECONDS
            ):

                # Debounce window expired → flush
                full_utterance = " ".join(self._buffer).strip()
                self._buffer = []

                print(
                    f"[AGENT] Debounce flush → "
                    f"full utterance: {full_utterance!r}"
                )

                try:
                    self._process(full_utterance)
                except Exception as exc:
                    print(f"[AGENT] ERROR processing utterance: {exc}")
                    self.tts_queue.put_nowait({
                        "text": f"Sorry, I ran into an error: {exc}",
                        "audio_b64": "",
                        "error": True,
                    })

        print("[AGENT] Runner thread stopped")

    def _process(self, user_text: str):
        """
        Full pipeline for one complete user utterance:
          1. Gemini generates text response
          2. Sarvam TTS converts text → audio bytes
          3. Audio is base64-encoded and queued for the UI
        """

        # Step 1: LLM
        response_text = self.agent.respond(user_text)

        if not response_text:
            print("[AGENT] Empty response from Gemini — skipping TTS")
            return

        # Step 2: TTS
        audio_bytes = self.tts.speak(response_text)

        # Step 3: encode for browser <audio> tag
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        print(
            f"[AGENT] Done — "
            f"response={len(response_text)} chars, "
            f"audio={len(audio_bytes)} bytes"
        )

        self.tts_queue.put_nowait({
            "text": response_text,
            "audio_bytes": audio_bytes,
            "audio_b64": audio_b64,
        })


