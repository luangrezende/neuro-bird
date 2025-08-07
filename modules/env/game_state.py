import sys
import os
import threading

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import config

class GameState:
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(GameState, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not GameState._initialized:
            invalid_score = config.get_section('vision')['display']['defaults']['invalid_score']
            self.score = invalid_score
            self._score_lock = threading.Lock()
            GameState._initialized = True
    
    def update_score(self, score: int) -> None:
        with self._score_lock:
            self.score = score
    
    def get_score(self) -> int:
        with self._score_lock:
            return self.score
    
    def reset_score(self) -> None:
        with self._score_lock:
            invalid_score = config.get_section('vision')['display']['defaults']['invalid_score']
            self.score = invalid_score
