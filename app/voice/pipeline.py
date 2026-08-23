from app.llm.qwen import QwenClient
from app.voice.stt import SpeechToText
from app.voice.tts import TextToSpeech


class VoiceAgent:

    def __init__(self):
        self.stt = SpeechToText()
        self.llm = QwenClient()
        self.tts = TextToSpeech()

    def process(self, audio_path: str) -> bytes:

        # 1. Speech → Text
        transcript = self.stt.transcribe(audio_path)

        # 2. Text → Qwen3
        response = self.llm.generate(transcript)

        # 3. Text → Speech
        audio = self.tts.generate(response)

        return audio