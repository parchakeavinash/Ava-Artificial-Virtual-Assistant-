import asyncio

from sarvamai import AsyncSarvamAI
from app.config import settings


async def main():
    print("Creating Sarvam client...")

    client = AsyncSarvamAI(
        api_subscription_key=settings.SARVAM_API_KEY
    )

    print("Connecting to Sarvam...")

    async with (
        client.speech_to_text_realtime_streaming.connect(
            language_code="en-IN",
            model="saaras:v3-realtime",
            stream_type="fast",
            mode="transcribe",
            encoding="linear16",
            sample_rate="16000",
        )
        as ws
    ):
        print("✅ CONNECTED TO SARVAM")

        async for message in ws:
            print("SARVAM:", message)

            if getattr(message, "event", None) == "error":
                break


asyncio.run(main())