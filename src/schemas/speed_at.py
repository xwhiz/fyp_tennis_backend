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

    def to_dict(self):
        return {
            "speed": self.speed,
            "time_diff": self.time_diff,
            "timestamp": self.timestamp,
            "distance": self.distance,
            "shot_type": self.shot_type,
            "hitter": self.hitter,
        }

    def to_json(self):
        return json.dumps(self.to_dict())
