class GameState:
    def __init__(self):
        self.running = False
        self.player_score = 0
        self.ai_score = 0
        self.settings = {
            "difficulty": "Medium",
            "delay": 3.0,
            "sound_on": True
        }
        self.last_ai_response = None

    def reset(self):
        self.running = False
        self.player_score = 0
        self.ai_score = 0
        self.last_ai_response = None

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def update_settings(self, settings: dict):
        self.settings.update(settings)

    def get_status(self):
        return {
            "player_score": self.player_score,
            "ai_score": self.ai_score,
            "ai_thinking": self.running,
            "ai_hint": self.last_ai_response
        }
