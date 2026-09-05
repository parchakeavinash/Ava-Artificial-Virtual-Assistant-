import asyncio
import io
import json
import os
import sys
import wave
from urllib.parse import urlencode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import websockets
from app.config.settings import settings
from app.voice.tts import SarvamTTS

async def test_live_stt():
    print("Synthesizing test audio with Sarvam TTS...")
    tts = SarvamTTS()
    wav_bytes = tts.speak("Hello Ava, this is a test. What are my tasks for today?")
    print(f"Generated WAV audio: {len(wav_bytes)} bytes")

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        pcm_data = wf.readframes(wf.getnframes())
        print(f"WAV specs: channels={n_channels}, sampwidth={sampwidth}, framerate={framerate}, pcm_len={len(pcm_data)}")

    # Connect to Sarvam Realtime STT
    params = {
        "language_code": "en-IN",
        "model": "saaras:v3-realtime",
        "stream_type": "balanced",
        "silence_duration_ms": "1000",
    }
    url = f"wss://api.sarvam.ai/speech-to-text-realtime/ws?{urlencode(params)}"
    headers = {"api-subscription-key": settings.SARVAM_API_KEY}

    print("Connecting to Sarvam STT WebSocket...")
    async with websockets.connect(url, additional_headers=headers) as ws:
        async def receiver():
            try:
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    print("\n[RECV EVENT]:", json.dumps(data, indent=2))
            except Exception as e:
                print("Receiver exited:", e)

        recv_task = asyncio.create_task(receiver())

        # Stream PCM in ~100ms chunks (16000 * 2 bytes * 0.1 = 3200 bytes)
        # If the generated audio is 24kHz or something, let's see.
        chunk_size = 3200
        for i in range(0, len(pcm_data), chunk_size):
            chunk = pcm_data[i : i + chunk_size]
            await ws.send(chunk)
            await asyncio.sleep(0.1)

        print("Streaming 2 seconds of silence to trigger VAD endpointing...")
        silence_chunk = bytes(3200)  # 100ms of zeros
        for _ in range(20):
            await ws.send(silence_chunk)
            await asyncio.sleep(0.1)

        print("Waiting 3s for final transcript...")
        await asyncio.sleep(3)
        recv_task.cancel()

if __name__ == "__main__":
    asyncio.run(test_live_stt())
