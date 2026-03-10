"""Knowledge Lifecycle for Research Agent"""
from enum import Enum
from typing import Dict
import time

class Phase(Enum):
    SPROUT = "sprout"
    GREEN_LEAF = "green_leaf"
    YELLOW_LEAF = "yellow_leaf"
    RED_LEAF = "red_leaf"
    SOIL = "soil"

class CapsuleLifecycle:
    def __init__(self):
        self.capsules: Dict[str, dict] = {}
    
    def add(self, cid: str, content: str, priority="P2"):
        self.capsules[cid] = {'content': content, 'priority': priority, 
                            'confidence': 0.7, 'phase': Phase.SPROUT,
                            'created': time.time()}
    
    def access(self, cid: str) -> bool:
        if cid not in self.capsules: return False
        c = self.capsules[cid]
        c['confidence'] = min(1.0, c['confidence'] + 0.03)
        c['phase'] = Phase.GREEN_LEAF if c['confidence'] >= 0.8 else Phase.SPROUT
        return True
    
    def decay(self):
        for c in self.capsules.values():
            d = {'P0': 0, 'P1': 0.004, 'P2': 0.008}.get(c['priority'], 0.008)
            c['confidence'] = max(0, c['confidence'] - d)
