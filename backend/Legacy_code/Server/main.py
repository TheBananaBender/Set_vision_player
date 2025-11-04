from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from models import FramePacket, GameControl, Settings, GameStatus
from game_state import GameState

import asyncio
import json

app = FastAPI()
state = GameState()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/control")
async def control_game(cmd: GameControl):
    if cmd.action == "start":
        state.start()
    elif cmd.action == "stop":
        state.stop()
    elif cmd.action == "reset":
        state.reset()
    return {"status": "ok", "gameRunning": state.running}

@app.post("/settings")
async def update_settings(new_settings: Settings):
    state.update_settings(new_settings.dict())
    return {"status": "ok"}

@app.get("/status", response_model=GameStatus)
async def get_status():
    return state.get_status()

@app.websocket("/ws/frames")
async def stream_frames(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            packet = FramePacket(**json.loads(data))
            if state.running:
                result = state.process_frame(packet.image_base64)
                await websocket.send_json(result or {
                    "human_score": state.human.score,
                    "ai_score": state.ai.score
                })
            await asyncio.sleep(1 / 2)  # control frame rate: 2 fps
    except Exception as e:
        print("WebSocket error:", e)
        await websocket.close()
