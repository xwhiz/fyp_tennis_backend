import random

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
from src.core.court_reference import CourtReference
import cv2
import numpy as np
from typing import Optional


def scene_detect(path_video):
    """
    Split video to disjoint fragments based on color histograms
    """
    video_manager = VideoManager([path_video])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector())
    base_timecode = video_manager.get_base_timecode()

    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list(base_timecode)

    if scene_list == []:
        scene_list = [
            (video_manager.get_base_timecode(), video_manager.get_current_timecode())
        ]
    scenes = [[x[0].frame_num, x[1].frame_num] for x in scene_list]
    return scenes


def get_court_img():
    court_reference = CourtReference()
    court = court_reference.build_court_reference()
    court = cv2.dilate(court, np.ones((10, 10), dtype=np.uint8))
    court_img = (np.stack((court, court, court), axis=2) * 255).astype(np.uint8)
    return court_img


def perspective_transform_point(
    point: tuple[Optional[float], Optional[float]],
    homography_matrix: Optional[np.ndarray],
):
    if point[0] is None or homography_matrix is None:
        return point

    point = np.array(point, dtype=np.float32).reshape(1, 1, 2)
    point = cv2.perspectiveTransform(point, homography_matrix)
    return point[0, 0, 0], point[0, 0, 1]


def get_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0

    p = np.polyfit(range(len(values)), values, 1)
    return p[0]


def compress_frame(frame: np.ndarray, quality: int = 80):
    ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return None if not ret else buffer


def check_court_in_scene(court_detector, video_path, start_frame, end_frame, num_samples=5):
    """
    Sample random frames from a scene range, run court detection on each.
    Return True if a court is detected in at least 2 of the sampled frames.
    """
    cap = cv2.VideoCapture(video_path)
    frame_range = range(start_frame, end_frame)
    sample_count = min(num_samples, end_frame - start_frame)
    indices = sorted(random.sample(frame_range, sample_count))
    detections = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        matrices, _ = court_detector.infer_model([frame])
        if matrices[0] is not None:
            detections += 1
    cap.release()
    return detections >= 2
