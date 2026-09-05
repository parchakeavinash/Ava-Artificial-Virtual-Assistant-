import base64
from queue import Empty, Queue
import threading
import time

from app.agent.langchain_agent import LangChainResilientAgent
from app.voice.tts import SarvamTTS

_DEBOUNCE_SECONDS = 1.5


class AgentRunner:
    """
    Background worker thread that receives STT transcript fragments,
    debounces them into full utterances, calls the resilient LangChain Agent (Groq + Gemini fallback),
    and runs Sarvam TTS to produce spoken audio.
    """

    def __init__(self):
        self.transcript_queue: Queue[str] = Queue()
        self.tts_queue: Queue[dict] = Queue()

        self.running = False
        self.thread: threading.Thread | None = None
        self.is_busy = False
        self.current_action = ""

        self.agent = LangChainResilientAgent()
        self.tts = SarvamTTS()
        self.current_session_id: str = "default"

        self._buffer: list[str] = []
        self._last_fragment_time: float = 0.0

    def set_session_id(self, session_id: str):
        """Switch active session for memory tracking."""
        self.current_session_id = session_id

    def start(self):
        """Start the background worker thread."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the background worker thread."""
        self.running = False

    def push_transcript(self, text: str):
        """Push a final transcript chunk into the debounce buffer."""
        cleaned = text.strip()
        if cleaned:
            self.transcript_queue.put_nowait(cleaned)

    def get_results(self) -> list[dict]:
        """Drain all completed responses from the queue."""
        results = []
        while True:
            try:
                results.append(self.tts_queue.get_nowait())
            except Empty:
                break
        return results

    def reset_conversation(self, session_id: str | None = None):
        """Reset conversation session."""
        target_session = session_id or self.current_session_id
        self.agent.clear_history(session_id=target_session)
        self._buffer.clear()

    def _run(self):
        while self.running:
            try:
                fragment = self.transcript_queue.get(timeout=0.1)
                self._buffer.append(fragment)
                self._last_fragment_time = time.time()
            except Empty:
                pass

            now = time.time()
            if self._buffer and (now - self._last_fragment_time) >= _DEBOUNCE_SECONDS:
                full_utterance = " ".join(self._buffer).strip()
                self._buffer = []
                print(f"[RUNNER] Processing utterance: {full_utterance!r}")
                self._process(full_utterance)

    def _process(self, user_text: str):
        self.is_busy = True
        self.current_action = "Ava is thinking & using tools..."

        try:
            response_text = self.agent.respond(user_text, session_id=self.current_session_id)
            if not response_text:
                return

            self.current_action = "Synthesizing voice output..."
            audio_bytes = self.tts.speak(response_text)
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8") if audio_bytes else ""

            self.tts_queue.put_nowait({
                "text": response_text,
                "audio_bytes": audio_bytes,
                "audio_b64": audio_b64,
                "error": False,
            })

        except Exception as exc:
            print(f"[RUNNER] Error processing: {exc}")
            self.tts_queue.put_nowait({
                "text": f"Sorry, I ran into an issue: {exc}",
                "audio_bytes": b"",
                "audio_b64": "",
                "error": True,
            })

        finally:
            self.is_busy = False
            self.current_action = ""
