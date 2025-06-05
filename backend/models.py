from pydantic import BaseModel
from typing import Literal


class FramePacket(BaseModel):
    image_base64: str


class GameControl(BaseModel):
    action: Literal["start", "stop", "reset"]


class Settings(BaseModel):
    difficulty: Literal["Easy", "Medium", "Hard"]
    delay: float
    sound_on: bool


class GameStatus(BaseModel):
    player_score: int
    ai_score: int
    ai_thinking: bool
    ai_hint: str | None = None
