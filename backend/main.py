from __future__ import annotations

import os
import time
import json
from datetime import datetime
from collections import deque
from typing import List, Dict, Any

import cv2
import numpy as np
from PIL import Image
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# --- game modules (package-relative imports) ---
from .Players import HumanPlayer, AIPlayer
from .Game_logic import Card, Game
from .vision_models import Pipeline, HandsSensor

# ----------------- constants & dirs -----------------
COLOR = {0: "Red", 1: "Green", 2: "Purple"}
NUMBER = {0: "One", 1: "Two", 2: "Three"}
SHADING = {0: "Solid", 1: "Striped", 2: "Open"}
SHAPE = {0: "Diamond", 1: "Squiggle", 2: "Oval"}

BACKEND_DIR = Path(__file__).resolve().parent
SAVE_DIR = BACKEND_DIR / "saved_frames"
SAVED_CARDS_DIR = BACKEND_DIR / "save_cards"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
SAVED_CARDS_DIR.mkdir(parents=True, exist_ok=True)

FPS = 40
HERZ = 1.0 / FPS

# --------------- helpers ----------------
def warp_card(image_bgr: np.ndarray, box: List[List[float]], output_size=(256, 256)):
    dst_pts = np.array(
        [[0, 0],
         [output_size[0] - 1, 0],
         [output_size[0] - 1, output_size[1] - 1],
         [0, output_size[1] - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(np.float32(box), dst_pts)
    return cv2.warpPerspective(image_bgr, M, output_size)

def bgr_from_encoded(buf: bytes) -> np.ndarray | None:
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def card_from_tuple(t) -> Card:
    # t = (polygon, color, shape, quantity, filling)
    return Card(t[1], t[2], t[3], t[4], polygon=t[0])

# --------------- session state ----------------
class SessionState:
    """
    Holds per-connection game state. Heavy models are reused from app.state.pipeline
    to avoid reloading on every WebSocket connect.
    """
    def __init__(self, app: FastAPI):
        self.game = Game()
        self.human = HumanPlayer("Human", self.game.board, 0, 0)
        self.ai = AIPlayer("AI", self.game.board, difficulty="medium", score=0, id=1)
        self.game.add_player(self.human)
        self.game.add_player(self.ai)
        # If your AI has a background thread, start it:
        try:
            self.ai.start()
        except Exception:
            pass

        self.board = self.game.board
        self.pipeline: Pipeline = app.state.pipeline  # reuse loaded models
        self.hands = HandsSensor()

        self.frame_num = 0
        self.update_cards: List[set] = []
        self.updated_already = False

        # optional smoothing like your original
        self.hand_history = deque([False] * 3, maxlen=3)

        # last received raw BGR (for “save”)
        self._last_frame_bgr: np.ndarray | None = None

    def detect_cards(self, frame_bgr: np.ndarray) -> set:
        res = self.pipeline.detect_and_classify_from_array(frame_bgr)
        return set(card_from_tuple(r) for r in res)

    def process_frame(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Mirrors your original loop logic, but on a single frame,
        and returns a JSON serializable dict for the browser.
        """
        self.frame_num += 1
        start = time.time()
        self._last_frame_bgr = frame_bgr

        # Every 2 frames, check hands
        if self.frame_num % 2 == 0:
            self.hands.is_hands_check(frame_bgr)

        # If no hands -> accumulate 3 detections, then update board & players
        if not self.hands.is_hands:
            if len(self.update_cards) < 3:
                self.update_cards.append(self.detect_cards(frame_bgr))
            elif len(self.update_cards) == 3 and not self.updated_already:
                self.board.update(self.update_cards)
                try:
                    self.ai.notify_new_board()
                except Exception:
                    pass
                self.human.update()
                self.update_cards = []
                self.updated_already = True
        else:
            self.update_cards = []
            self.updated_already = False

        # Build polygons to render
        polys: List[List[List[int]]] = []
        with self.board._lock:
            for c in self.board.cards:
                if c.polygon:
                    # ensure ints
                    polys.append([[int(x), int(y)] for (x, y) in c.polygon])

        elapsed = time.time() - start
        return {
            "type": "result",
            "frame_num": self.frame_num,
            "hands": bool(self.hands.is_hands),
            "polygons": polys,
            "scores": {"human": self.human.score, "ai": self.ai.score},
            "latency_ms": int(elapsed * 1000),
        }

    def save_current_frame_and_crops(self) -> Dict[str, Any]:
        if self._last_frame_bgr is None:
            return {"saved": False, "reason": "no_frame"}
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        frame_path = (SAVE_DIR / f"raw_frame_{ts}.png")
        cv2.imwrite(str(frame_path), self._last_frame_bgr)

        saved_files: List[str] = []
        detected_cards = self.detect_cards(self._last_frame_bgr)
        for card in detected_cards:
            if not card.polygon or len(card.polygon) != 4:
                continue
            try:
                warped = warp_card(self._last_frame_bgr, card.polygon)
                color_str = COLOR.get(card.color, str(card.color))
                shape_str = SHAPE.get(card.shape, str(card.shape))
                quantity_str = NUMBER.get(card.quantity, str(card.quantity))
                shading_str = SHADING.get(card.filling, str(card.filling))
                fname = f"{color_str}_{shape_str}_{quantity_str}_{shading_str}_{time.time_ns()}.png"
                save_path = SAVED_CARDS_DIR / fname
                cv2.imwrite(str(save_path), warped)
                saved_files.append(fname)
            except Exception as e:
                # continue saving others
                print(f"[ERROR] warp/save failed: {e}")
        return {"saved": True, "frame": str(frame_path), "cards": saved_files}

# --------------- app setup ----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Preload heavy models once at startup for snappy WS connects
@app.on_event("startup")
def _load_models_once():
    app.state.pipeline = Pipeline()
    app.state.last_session: SessionState | None = None
    print("[startup] models loaded and ready")

@app.get("/health")
def health():
    return {"ok": True, "t": time.time()}

# --- optional: simple /control for dev convenience (POST {"action":"save"}) ---
from pydantic import BaseModel
class ControlReq(BaseModel):
    action: str

@app.post("/control")
def control(req: ControlReq):
    if req.action == "save":
        sess = getattr(app.state, "last_session", None)
        if not sess:
            return {"ok": False, "reason": "no_active_session"}
        info = sess.save_current_frame_and_crops()
        return {"ok": True, **info}
    return {"ok": False, "reason": "unknown_action"}

# --------------- WebSocket ----------------
@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()

    # one session per socket (simple). For prod, key by a client-supplied session_id.
    sess = SessionState(app)
    app.state.last_session = sess  # expose to /control for dev

    try:
        while True:
            try:
                message = await websocket.receive()  # exactly once per loop
            except WebSocketDisconnect:
                break
            except RuntimeError as e:
                # Happens if receive() is called after a disconnect event
                if "disconnect" in str(e).lower():
                    break
                raise

            # explicit disconnect message
            if message.get("type") == "websocket.disconnect":
                break

            data_b = message.get("bytes")
            data_t = message.get("text")

            if data_b:
                frame_bgr = bgr_from_encoded(data_b)
                if frame_bgr is None:
                    await websocket.send_text(json.dumps({"type": "error", "message": "decode_failed"}))
                    continue
                result = sess.process_frame(frame_bgr)
                await websocket.send_text(json.dumps(result))

            elif data_t:
                # small JSON control messages (e.g., {"type":"save"})
                try:
                    payload = json.loads(data_t)
                except Exception:
                    payload = {}
                t = payload.get("type")
                if t == "save":
                    info = sess.save_current_frame_and_crops()
                    await websocket.send_text(json.dumps({"type": "save_ack", **info}))
                elif t == "ping":
                    await websocket.send_text(json.dumps({"type": "pong", "t": time.time()}))
                else:
                    await websocket.send_text(json.dumps({"type": "ack"}))
    finally:
        # clear only if this socket owned it
        if getattr(app.state, "last_session", None) is sess:
            app.state.last_session = None
