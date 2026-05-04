from dataclasses import dataclass
import json


@dataclass
class SpeedAt:
    speed: float
    time_diff: float
    timestamp: float
    distance: float
    shot_type: str
    hitter: str = "unknown"
    hitter_confidence: float = 0.0
    original_hitter: str = "unknown"
    attribution_method: str = "legacy"

    def to_dict(self):
        return {
            "speed": self.speed,
            "time_diff": self.time_diff,
            "timestamp": self.timestamp,
            "distance": self.distance,
            "shot_type": self.shot_type,
            "hitter": self.hitter,
            "hitter_confidence": self.hitter_confidence,
            "original_hitter": self.original_hitter,
            "attribution_method": self.attribution_method,
        }

    def to_json(self):
        return json.dumps(self.to_dict())
