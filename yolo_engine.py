from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIGURATION OBJECT
# ============================================================
@dataclass
class EngineConfig:
    model_path: str = "yolov8n-pose.pt"

    # Demo alert behavior
    demo_alert_on_any_pose: bool = True

    # Overlay settings
    alert_text: str = "SUSPICIOUS BEHAVIOR DETECTED"
    alert_position: Tuple[int, int] = (20, 40)
    alert_font_scale: float = 1.0
    alert_thickness: int = 3
    alert_color_bgr: Tuple[int, int, int] = (0, 0, 255)  # Red (BGR)


# ============================================================
# YOLO ENGINE
# ============================================================
class YoloEngine:
    """
    YOLOv8-Pose Engine
    -----------------
    Input : BGR frame (OpenCV / NumPy)
    Output: Annotated BGR frame + suspicious flag

    Notes:
    - No camera assumptions
    - No Streamlit dependencies
    - Safe for offline video processing
    - Training-ready (pose extraction hook included)
    """

    def __init__(self, model_path: Optional[str] = None):
        self.cfg = EngineConfig(
            model_path=model_path or EngineConfig.model_path
        )
        self.model = YOLO(self.cfg.model_path)

    # --------------------------------------------------------
    # INTERNAL HELPERS
    # --------------------------------------------------------
    @staticmethod
    def _pose_detected(result) -> bool:
        """
        Returns True if at least one human pose is detected.
        """
        try:
            keypoints = result.keypoints
            return (
                keypoints is not None
                and keypoints.xy is not None
                and len(keypoints.xy) > 0
            )
        except Exception:
            return False

    @staticmethod
    def _extract_pose_array(result) -> Optional[np.ndarray]:
        """
        Extract pose keypoints as NumPy array:
        Shape: (num_people, num_joints, 2)

        Useful for:
        - Training datasets
        - Temporal models (LSTM / GRU)
        """
        try:
            if result.keypoints is None or result.keypoints.xy is None:
                return None
            return result.keypoints.xy.cpu().numpy()
        except Exception:
            return None

    # --------------------------------------------------------
    # DEMO ALERT LOGIC (PLACEHOLDER)
    # --------------------------------------------------------
    def _demo_suspicion_logic(self, pose_detected: bool) -> bool:
        """
        DEMO LOGIC (HONEST PLACEHOLDER):
        - Trigger alert if any pose is detected

        Replace later with:
        - Temporal logic
        - Gesture rules
        - LSTM inference
        """
        if self.cfg.demo_alert_on_any_pose:
            return pose_detected
        return False

    # --------------------------------------------------------
    # MAIN PROCESSING FUNCTION
    # --------------------------------------------------------
    def process_frame(
        self,
        frame_bgr: np.ndarray,
        return_pose: bool = False
    ) -> Tuple[np.ndarray, bool, Optional[np.ndarray]]:
        """
        Process a single video frame.

        Parameters
        ----------
        frame_bgr : np.ndarray
            OpenCV frame in BGR format

        return_pose : bool
            If True, returns pose keypoints array for training

        Returns
        -------
        annotated_frame : np.ndarray
            BGR frame with skeleton + alert overlay

        suspicious : bool
            Whether suspicious behavior is detected

        pose_array : Optional[np.ndarray]
            Pose keypoints (for training), or None
        """

        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr, False, None

        # --- YOLOv8 Pose Inference ---
        results = self.model(frame_bgr, verbose=False)
        result = results[0]

        # --- Visualization ---
        annotated = result.plot() if result is not None else frame_bgr.copy()
        if annotated is None:
            annotated = frame_bgr.copy()

        # --- Pose Detection ---
        pose_found = self._pose_detected(result)

        # --- Alert Logic ---
        suspicious = self._demo_suspicion_logic(pose_found)

        # --- Overlay Alert ---
        if suspicious:
            cv2.putText(
                annotated,
                self.cfg.alert_text,
                self.cfg.alert_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.cfg.alert_font_scale,
                self.cfg.alert_color_bgr,
                self.cfg.alert_thickness,
            )

        # --- Pose Extraction (optional, for training) ---
        pose_array = None
        if return_pose:
            pose_array = self._extract_pose_array(result)

        return annotated, suspicious, pose_array
