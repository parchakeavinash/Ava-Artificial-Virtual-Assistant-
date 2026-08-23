import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.sarvam_stt import SarvamRealtimeSTT


app = FastAPI(
    title="MCP Voice Agent",
    version="0.1.0",
)


@app.get("/health")
async def health():
    return {
        "status": "ok"
    }


@app.websocket("/ws/voice")
async def voice_websocket(
    websocket: WebSocket,
):
    await websocket.accept()

    sarvam = SarvamRealtimeSTT()

    try:

        # Connect FastAPI → Sarvam
        await sarvam.connect()

        async def browser_to_sarvam():
            while True:

                audio = await websocket.receive_bytes()

                await sarvam.send_audio(audio)

        async def sarvam_to_browser():
            while True:

                message = await sarvam.receive()

                await websocket.send_json(
                    message
                )

        await asyncio.gather(
            browser_to_sarvam(),
            sarvam_to_browser(),
        )

    except WebSocketDisconnect:

        print("Browser disconnected")

    except Exception as e:

        print(f"Voice pipeline error: {e}")

    finally:

        await sarvam.close()