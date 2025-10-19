# override_bus.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import threading
import time
import numpy as np

@dataclass
class OverrideState:
    enabled: bool = False                # Master switch for manual override
    active_robot: Optional[str] = None   # Name of robot currently driven by GUI
    q_targets: Dict[str, np.ndarray] = field(default_factory=dict)  # joint targets
    gripper_closed: Dict[str, bool] = field(default_factory=dict)   # True = closed
    conveyor_pause: bool = True          # If True, pause conveyor while override enabled
    updated_at: float = 0.0

class OverrideBus:
    def __init__(self):
        self._st = OverrideState()
        self._lock = threading.Lock()

    # -------- GUI publishers --------
    def set_enabled(self, enabled: bool):
        with self._lock:
            self._st.enabled = bool(enabled)
            self._st.updated_at = time.time()

    def set_active_robot(self, name: Optional[str]):
        with self._lock:
            self._st.active_robot = name
            self._st.updated_at = time.time()

    def publish_q(self, name: str, q: List[float]):
        qarr = np.asarray(q, dtype=float)
        with self._lock:
            self._st.q_targets[name] = qarr
            self._st.updated_at = time.time()

    def publish_gripper_closed(self, name: str, closed: bool):
        with self._lock:
            self._st.gripper_closed[name] = bool(closed)
            self._st.updated_at = time.time()

    def set_conveyor_pause(self, pause: bool):
        with self._lock:
            self._st.conveyor_pause = bool(pause)
            self._st.updated_at = time.time()


    def is_enabled_for(self, name: str) -> bool:
        with self._lock:
            return self._st.enabled and (self._st.active_robot == name)

    def get_q_for(self, name: str) -> Optional[np.ndarray]:
        with self._lock:
            return self._st.q_targets.get(name, None)

    def get_gripper_closed_for(self, name: str) -> Optional[bool]:
        with self._lock:
            return self._st.gripper_closed.get(name, None)

    def should_pause_conveyor(self) -> bool:
        with self._lock:
            return self._st.enabled and self._st.conveyor_pause

bus = OverrideBus()
