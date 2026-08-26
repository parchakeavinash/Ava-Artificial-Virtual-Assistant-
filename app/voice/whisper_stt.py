import io
import os
import time
import wave
from queue import Empty, Queue
import threading
from typing import Any

from groq import Groq
from app.config.settings import settings


class WhisperTracker:
    """Tracks Groq Whisper request rate limits and active model status."""

    def __init__(self):
        self.lock = threading.Lock()
        self.minute_window: list[float] = []
        self.requests_today: int = 0
        self.active_model: str = settings.GROQ_WHISPER_PRIMARY
        self.fallback_count: int = 0
        self.last_latency_s: float = 0.0

    def record_request(self, model: str, latency: float):
        with self.lock:
            now = time.time()
            self.minute_window.append(now)
            self.minute_window = [t for t in self.minute_window if now - t < 60]
            self.requests_today += 1
            self.active_model = model
            self.last_latency_s = latency

    def get_stats(self) -> dict[str, Any]:
        with self.lock:
            now = time.time()
            self.minute_window = [t for t in self.minute_window if now - t < 60]
            rpm_used = len(self.minute_window)
            rpm_limit = 20
            rpd_limit = 2000
            return {
                "active_model": self.active_model,
                "rpm_used": rpm_used,
                "rpm_limit": rpm_limit,
                "rpm_remaining": max(0, rpm_limit - rpm_used),
                "rpd_used": self.requests_today,
                "rpd_limit": rpd_limit,
                "rpd_remaining": max(0, rpd_limit - self.requests_today),
                "last_latency_s": round(self.last_latency_s, 2),
            }


tracker = WhisperTracker()


class GroqWhisperSTT:
    """
    Transcribes audio using Groq's Whisper API with automatic fallback
    from 'whisper-large-v3' to 'whisper-large-v3-turbo'.
    """

    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.primary_model = settings.GROQ_WHISPER_PRIMARY
        self.fallback_model = settings.GROQ_WHISPER_FALLBACK

    def transcribe_audio_bytes(self, pcm16_bytes: bytes, sample_rate: int = 16000) -> str:
        """
        Convert PCM16 raw audio bytes to WAV in-memory and transcribe with Groq Whisper.
        """
        if not pcm16_bytes or len(pcm16_bytes) < 3200:  # < 0.1s audio
            return ""

        # Build in-memory WAV
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm16_bytes)

        wav_bytes = wav_buffer.getvalue()

        # Step 1: Try Primary (whisper-large-v3)
        t0 = time.time()
        try:
            transcription = self.client.audio.transcriptions.create(
                file=("audio.wav", wav_bytes),
                model=self.primary_model,
                temperature=0.0,
                response_format="json",
            )
            text = getattr(transcription, "text", "").strip()
            tracker.record_request(self.primary_model, time.time() - t0)
            return text

        except Exception as primary_err:
            print(f"[STT: Whisper] Primary model ({self.primary_model}) error: {primary_err}")
            print(f"[STT: Whisper] Falling back to {self.fallback_model}...")

            # Step 2: Auto-fallback to whisper-large-v3-turbo
            try:
                t1 = time.time()
                transcription = self.client.audio.transcriptions.create(
                    file=("audio.wav", wav_bytes),
                    model=self.fallback_model,
                    temperature=0.0,
                    response_format="json",
                )
                text = getattr(transcription, "text", "").strip()
                tracker.record_request(self.fallback_model, time.time() - t1)
                tracker.fallback_count += 1
                return text

            except Exception as fallback_err:
                print(f"[STT: Whisper] Fallback model also failed: {fallback_err}")
                return ""
