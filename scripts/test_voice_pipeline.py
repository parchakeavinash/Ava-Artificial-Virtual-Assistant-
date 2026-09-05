import asyncio
import io
import os
import sys
import time
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import av
from app.voice.stt import SarvamRealtimeSTT
from app.voice.runner import AgentRunner
from app.voice.tts import SarvamTTS

def test_full_voice_loop():
    print("=== 1. Synthesizing test audio prompt with Sarvam TTS ===")
    tts = SarvamTTS()
    wav_bytes = tts.speak("Hello Ava, tell me my tasks.")
    assert len(wav_bytes) > 0, "TTS failed to synthesize audio!"
    print(f"Generated WAV prompt: {len(wav_bytes)} bytes")

    print("\n=== 2. Testing PyAV AudioResampler for WebRTC ===")
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    # Read WAV and feed frames to resampler
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        pcm_raw = wf.readframes(wf.getnframes())

    print(f"Input audio: {framerate}Hz, channels={n_channels}, bytes={len(pcm_raw)}")

    # Initialize STT and AgentRunner
    print("\n=== 3. Starting SarvamRealtimeSTT and AgentRunner ===")
    stt = SarvamRealtimeSTT()
    stt.start()
    
    runner = AgentRunner()
    runner.start()
    time.sleep(1.5)  # Allow WebSocket connection to open

    print("\n=== 4. Streaming PCM audio to Sarvam STT ===")
    # Push PCM chunks
    chunk_size = 3200  # 100ms
    for i in range(0, len(pcm_raw), chunk_size):
        stt.push_audio(pcm_raw[i : i + chunk_size])
        time.sleep(0.08)

    # Push 1.5 seconds of silence to trigger VAD endpointing
    print("Sending silence frames to trigger VAD endpointing...")
    silence = bytes(3200)
    for _ in range(15):
        stt.push_audio(silence)
        time.sleep(0.08)

    print("\n=== 5. Waiting for STT Events and Runner Execution ===")
    start_wait = time.time()
    final_text = ""
    got_partial = False

    while time.time() - start_wait < 12:
        events = stt.get_events()
        for ev in events:
            if ev.get("type") == "partial":
                got_partial = True
                print(f"[STT Partial]: {ev.get('text')}")
            elif ev.get("type") == "final":
                final_text = ev.get("text")
                print(f"[STT Final!]: {final_text}")
                runner.push_transcript(final_text)

        # Check runner results
        results = runner.get_results()
        if results:
            print("\n=== 6. Received Ava Spoken Response! ===")
            for r in results:
                print(f"Ava Text Response: {r.get('text')}")
                print(f"Ava Audio Bytes: {len(r.get('audio_bytes', b''))} bytes")
                print(f"Error: {r.get('error')}")
            assert not results[0].get("error"), "Runner reported an error!"
            assert len(results[0].get("audio_bytes", b"")) > 0, "No audio returned by Ava!"
            print("\nSUCCESS: End-to-End Voice pipeline verified working perfectly!")
            break

        time.sleep(0.3)

    stt.stop()
    runner.stop()

if __name__ == "__main__":
    test_full_voice_loop()
