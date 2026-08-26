import base64
import io
import re
import wave

from sarvamai import SarvamAI

from app.config import settings


class SarvamTTS:
    """
    Converts text to speech using Sarvam's TTS REST API.

    Voice: "shubh" — natural male Indian voice on model "bulbul:v3".
    Uses bulbul:v3 for natural prosody, high fidelity, and full text coverage.
    """

    # Max characters Sarvam TTS accepts in one request.
    # Long responses are split into sentences and seamlessly joined.
    _MAX_CHARS = 500

    def __init__(self):

        self.client = SarvamAI(
            api_subscription_key=settings.SARVAM_API_KEY
        )

    # ---------------------------------------------------------
    # PUBLIC
    # ---------------------------------------------------------

    def speak(self, text: str) -> bytes:
        """
        Convert text → audio bytes (WAV format).

        If the text is longer than _MAX_CHARS, it is chunked
        at sentence boundaries and each chunk is synthesized.
        All chunks are seamlessly combined into one complete WAV file.
        """

        text = text.strip()

        if not text:
            return b""

        chunks = self._split(text)

        audio_parts: list[bytes] = []

        for chunk in chunks:

            print(f"[TTS] Speaking chunk ({len(chunk)} chars) with shubh (bulbul:v3)...")

            response = self.client.text_to_speech.convert(
                text=chunk,
                language_code="en-IN",
                speaker="shubh",
                model="bulbul:v3",
                pace=1.0,
                enable_preprocessing=True,
            )

            # response.audios is a list of base64-encoded strings.
            raw_b64 = response.audios[0]
            raw = base64.b64decode(raw_b64)

            audio_parts.append(raw)

        return self._combine_wavs(audio_parts)

    # ---------------------------------------------------------
    # PRIVATE
    # ---------------------------------------------------------

    def _split(self, text: str) -> list[str]:
        """
        Split long text into chunks at sentence boundaries so each
        chunk fits within Sarvam's character limit.
        """

        if len(text) <= self._MAX_CHARS:
            return [text]

        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks: list[str] = []
        current = ""

        for sentence in sentences:

            if len(current) + len(sentence) + 1 <= self._MAX_CHARS:
                current = (current + " " + sentence).strip()
            else:
                if current:
                    chunks.append(current)
                current = sentence

        if current:
            chunks.append(current)

        return chunks

    def _combine_wavs(self, wav_bytes_list: list[bytes]) -> bytes:
        """
        Properly combines multiple WAV audio byte segments by concatenating
        PCM frames under a single unified WAV header.
        """

        if not wav_bytes_list:
            return b""

        if len(wav_bytes_list) == 1:
            return wav_bytes_list[0]

        combined_frames = bytearray()
        params = None

        for wav_raw in wav_bytes_list:

            with wave.open(io.BytesIO(wav_raw), "rb") as w:

                if params is None:
                    params = w.getparams()

                combined_frames.extend(
                    w.readframes(w.getnframes())
                )

        out_io = io.BytesIO()

        with wave.open(out_io, "wb") as out_wav:
            out_wav.setparams(params)
            out_wav.writeframes(combined_frames)

        return out_io.getvalue()


