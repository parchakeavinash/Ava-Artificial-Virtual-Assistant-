import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import base64
import json
import websockets
from urllib.parse import urlencode
from app.config.settings import settings

async def test_sarvam_send():
    params = {
        "language_code": "en-IN",
        "model": "saaras:v3-realtime",
        "stream_type": "balanced",
        "silence_duration_ms": "1000",
    }
    url = f"wss://api.sarvam.ai/speech-to-text-realtime/ws?{urlencode(params)}"
    headers = {"api-subscription-key": settings.SARVAM_API_KEY}

    print("--- Test 1: Sending raw binary bytes ---")
    try:
        async with websockets.connect(url, additional_headers=headers) as ws:
            msg = await ws.recv()
            print("Init recv:", msg)
            await ws.send(bytes(3200))
            print("Sent 3200 binary bytes successfully.")
            # Wait for response
            msg2 = await asyncio.wait_for(ws.recv(), timeout=2.0)
            print("Recv after binary:", msg2)
    except Exception as e:
        print("Binary test error:", type(e).__name__, e)

    print("\n--- Test 2: Sending JSON audio_input ---")
    try:
        async with websockets.connect(url, additional_headers=headers) as ws:
            msg = await ws.recv()
            print("Init recv:", msg)
            b64_audio = base64.b64encode(bytes(3200)).decode("utf-8")
            payload = json.dumps({"event": "audio_input", "audio": b64_audio})
            await ws.send(payload)
            print("Sent JSON payload successfully.")
            msg2 = await asyncio.wait_for(ws.recv(), timeout=2.0)
            print("Recv after JSON:", msg2)
    except Exception as e:
        print("JSON test error:", type(e).__name__, e)

if __name__ == "__main__":
    asyncio.run(test_sarvam_send())
