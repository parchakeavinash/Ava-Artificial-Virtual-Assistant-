import base64
import io
import re
import wave
import httpx

from app.config.settings import settings

_SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
_MAX_CHARS_PER_CHUNK = 400


class SarvamTTS:
    """
    Converts text → audio using Sarvam AI 'bulbul:v3' with voice speaker 'shubh'.
    """

    def __init__(
        self,
        target_language_code: str = "en-IN",
        speaker: str = "shubh",
        model: str = "bulbul:v3",
    ):
        self.target_language_code = target_language_code
        self.speaker = speaker
        self.model = model

    def speak(self, text: str) -> bytes:
        """
        Convert text into complete WAV audio bytes.
        Handles long text by chunking and combining WAV streams.
        """
        cleaned = text.strip()
        if not cleaned:
            return b""

        chunks = self._chunk_text(cleaned)
        wav_parts: list[bytes] = []

        for chunk in chunks:
            chunk_wav = self._synthesize_chunk(chunk)
            if chunk_wav:
                wav_parts.append(chunk_wav)

        if not wav_parts:
            return b""
        if len(wav_parts) == 1:
            return wav_parts[0]

        return self._combine_wavs(wav_parts)

    def _chunk_text(self, text: str) -> list[str]:
        if len(text) <= _MAX_CHARS_PER_CHUNK:
            return [text]

        sentences = re.split(r"(?<=[.?!])\s+", text)
        chunks: list[str] = []
        current = ""

        for s in sentences:
            if len(current) + len(s) + 1 <= _MAX_CHARS_PER_CHUNK:
                current = f"{current} {s}".strip()
            else:
                if current:
                    chunks.append(current)
                current = s

        if current:
            chunks.append(current)

        return chunks or [text]

    def _synthesize_chunk(self, text: str) -> bytes:
        headers = {
            "api-subscription-key": settings.SARVAM_API_KEY,
            "Content-Type": "application/json",
        }

        payload = {
            "inputs": [text],
            "target_language_code": self.target_language_code,
            "speaker": self.speaker,
            "model": self.model,
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(_SARVAM_TTS_URL, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

            audios = data.get("audios", [])
            if not audios:
                return b""

            return base64.b64decode(audios[0])

        except Exception as e:
            print(f"[TTS] Error during synthesis: {e}")
            return b""

    @staticmethod
    def _combine_wavs(wav_bytes_list: list[bytes]) -> bytes:
        valid_wavs = [w for w in wav_bytes_list if len(w) > 44]
        if not valid_wavs:
            return b""
        if len(valid_wavs) == 1:
            return valid_wavs[0]

        try:
            with io.BytesIO(valid_wavs[0]) as first_io, wave.open(first_io, "rb") as first_wav:
                params = first_wav.getparams()

            combined_pcm = bytearray()
            for w in valid_wavs:
                try:
                    with io.BytesIO(w) as w_io, wave.open(w_io, "rb") as cur_wav:
                        combined_pcm.extend(cur_wav.readframes(cur_wav.getnframes()))
                except Exception:
                    pass

            out_io = io.BytesIO()
            with wave.open(out_io, "wb") as out_wav:
                out_wav.setparams(params)
                out_wav.writeframes(bytes(combined_pcm))

            return out_io.getvalue()
        except Exception as exc:
            print(f"[TTS] Failed to combine WAVs: {exc}")
            return valid_wavs[0]
