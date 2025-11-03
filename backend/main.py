from __future__ import annotations
import time
import json
from datetime import datetime
from collections import deque
from typing import List, Dict, Any

import cv2
import numpy as np
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect ,UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# --- game modules (package-relative imports) ---
from Players import HumanPlayer, AIPlayer
from Game_logic import Card, Game
from vision_models import Pipeline, HandsSensor

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

        # Improved smoothing for hand detection (larger buffer to handle flickering)
        self.hand_history = deque([False] * 5, maxlen=5)

        # last received raw BGR (for "save")
        self._last_frame_bgr: np.ndarray | None = None
        
        # Track previous polygon state for incremental updates
        self._prev_polygons: set = set()  # Set of polygon tuples for comparison

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
        
        # Update hand history for smoothing
        self.hand_history.append(self.hands.is_hands)
        
        # Count recent hand detections (need at least 3 "no hands" to proceed with update)
        hands_count = sum(self.hand_history)
        no_hands_stable = hands_count <= 1  # Allow 1 false positive in last 5 frames
        
        # If no hands (with smoothing) -> accumulate 3 detections, then update board & players
        if no_hands_stable:
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
            # Only clear if hands detected consistently (at least 3 out of 5 frames)
            if hands_count >= 3:
                self.update_cards = []
                self.updated_already = False

        # Build polygons to render and compute incremental updates
        current_polys: List[List[List[int]]] = []
        poly_to_card_map = {}  # Map polygon tuple to the polygon list for lookup
        
        with self.board._lock:
            for c in self.board.cards:
                if c.polygon:
                    # ensure ints
                    poly_list = [[int(x), int(y)] for (x, y) in c.polygon]
                    current_polys.append(poly_list)
                    # Create tuple for comparison (convert to tuple of tuples for hashing)
                    poly_tuple = tuple(tuple(p) for p in poly_list)
                    poly_to_card_map[poly_tuple] = poly_list

        # Compute incremental updates
        current_poly_set = set(poly_to_card_map.keys())
        
        # Check if first frame (before updating state)
        is_first_frame = not self._prev_polygons
        
        added_polys = current_poly_set - self._prev_polygons
        removed_polys = self._prev_polygons - current_poly_set
        
        # Build incremental update payload
        polys_added = [poly_to_card_map[poly_tup] for poly_tup in added_polys if poly_tup in poly_to_card_map]
        # For removed, we need to store them in a way frontend can identify
        # Convert removed tuples back to lists (poly_tup is tuple of tuples like ((x1,y1), (x2,y2), ...))
        polys_removed = []
        for poly_tup in removed_polys:
            try:
                # Convert tuple of tuples back to list of lists
                poly_list = [list(p) for p in poly_tup]
                polys_removed.append(poly_list)
            except Exception as e:
                print(f"[SessionState] Error converting removed polygon: {e}")
                continue
        
        # Update previous state
        self._prev_polygons = current_poly_set
        
        # Determine if we should send full update (first frame or major changes)
        # First frame: send full update
        # Major changes: more than 50% of polygons changed
        send_full = (is_first_frame or 
                    len(added_polys) + len(removed_polys) > max(len(current_poly_set), 1) * 0.5)
        
        elapsed = time.time() - start
        
        result = {
            "type": "result",
            "frame_num": self.frame_num,
            "hands": bool(self.hands.is_hands),
            "scores": {"human": self.human.score, "ai": self.ai.score},
            "latency_ms": int(elapsed * 1000),
        }
        
        if send_full:
            # Send full polygon list (first frame or major changes)
            result["polygons"] = current_polys
            result["update_type"] = "full"
        else:
            # Send incremental updates
            result["polygons_added"] = polys_added
            result["polygons_removed"] = polys_removed
            result["update_type"] = "incremental"
        
        return result

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
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Preload heavy models once at startup for snappy WS connects
@app.on_event("startup")
def _load_models_once():
    app.state.pipeline = Pipeline()
    app.state.last_session: SessionState | None = None
    app.state.active_sessions: int = 0
    # simple settings store (editable only when no active WS session)
    app.state.settings: Dict[str, Any] = {
        "difficulty": "medium",   # AI difficulty (easy/medium/hard)
        "delay_scale": 1.0,        # scale AI thinking delay
        "sound_on": True,
    }
    print("[startup] models loaded and ready")

@app.get("/health")
def health():
    return {"ok": True, "t": time.time()}

# --- settings & status ---
from pydantic import BaseModel

class SettingsReq(BaseModel):
    difficulty: str | None = None   # "easy" | "medium" | "hard"
    delay_scale: float | None = None
    sound_on: bool | None = None

@app.get("/status")
def status():
    sess = getattr(app.state, "last_session", None)
    running = bool(sess)
    scores = {"human": 0, "ai": 0}
    if sess:
        scores = {"human": sess.human.score, "ai": sess.ai.score}
    return {
        "running": running,
        "active_sessions": app.state.active_sessions,
        "settings": app.state.settings,
        "scores": scores,
    }

@app.post("/settings")
def update_settings(req: SettingsReq):
    if app.state.active_sessions > 0:
        raise HTTPException(status_code=409, detail="Cannot change settings while a game is running")
    # update store
    s = app.state.settings
    if req.difficulty is not None:
        s["difficulty"] = req.difficulty.lower()
    if req.delay_scale is not None:
        s["delay_scale"] = float(req.delay_scale)
    if req.sound_on is not None:
        s["sound_on"] = bool(req.sound_on)
    return {"ok": True, "settings": s}

# --- optional: simple /control for dev convenience (POST {"action":"save"}) ---
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

@app.post("/polygon")
async def polygon(file: UploadFile = File(...)):
    img = await file.read()  # consume upload
    return SessionState.process_frame(img)

    



# --------------- WebSocket ----------------
@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()

    # one session per socket (simple). For prod, key by a client-supplied session_id.
    sess = SessionState(app)
    # apply current settings to AI before starting
    try:
        diff = str(app.state.settings.get("difficulty", "medium")).lower()
        # reconfigure AI's thinking range via its API if available
        if hasattr(sess.ai, "difficulty"):
            sess.ai.difficulty = diff
            if hasattr(sess.ai, "_set_thinking_time"):
                sess.ai._set_thinking_time()
    except Exception:
        pass
    app.state.last_session = sess  # expose to /control for dev
    app.state.active_sessions += 1

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
        if app.state.active_sessions > 0:
            app.state.active_sessions -= 1
