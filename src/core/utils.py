import random

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
from src.core.court_reference import CourtReference
import cv2
import numpy as np
from typing import Optional


def scene_detect(path_video, show_progress=False):
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
    scene_manager.detect_scenes(frame_source=video_manager, show_progress=show_progress)
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


def classify_serve_type(bounce_x: float, bounce_y: float) -> str:
    """
    Classify a serve based on where the ball lands in court coordinates.

    Service box geometry (from CourtReference):
      Top box:    x in [423, 1242], y in [1110, 1748]
      Bottom box: x in [423, 1242], y in [1748, 2386]
      Center line: x = 832

    Zones (by normalized distance from center line within the half-box):
      T Serve:    < 0.33  (near center T)
      Body Serve: 0.33 - 0.67  (middle zone)
      Wide Serve: > 0.67  (near outer sideline)
    """
    # Service box boundaries
    service_x_min = 423
    service_x_max = 1242
    center_x = 832
    top_service_y_min = 1110
    top_service_y_max = 1748  # net
    bottom_service_y_min = 1748  # net
    bottom_service_y_max = 2386

    # Check if the bounce is in a service box
    in_top_box = (service_x_min <= bounce_x <= service_x_max and
                  top_service_y_min <= bounce_y <= top_service_y_max)
    in_bottom_box = (service_x_min <= bounce_x <= service_x_max and
                     bottom_service_y_min <= bounce_y <= bottom_service_y_max)

    if not in_top_box and not in_bottom_box:
        return "unknown"

    # Determine which half-box the ball is in and compute normalized distance from center
    if bounce_x < center_x:
        half_box_width = center_x - service_x_min  # 409
    else:
        half_box_width = service_x_max - center_x  # 410

    normalized_dist = abs(bounce_x - center_x) / half_box_width

    if normalized_dist < 0.33:
        return "t_serve"
    elif normalized_dist > 0.67:
        return "wide_serve"
    else:
        return "body_serve"


def generate_player_heatmap(court_points: list[tuple[float, float]], alpha: float = 0.6) -> np.ndarray:
    """
    Generate a heatmap image of player positions overlaid on the court reference.

    Args:
        court_points: List of (x, y) positions in court coordinates.
        alpha: Blend factor for the heatmap overlay (0 = court only, 1 = heatmap only).

    Returns:
        BGR image with the heatmap blended onto the court reference.
    """
    court_img = get_court_img()
    h, w = court_img.shape[:2]

    if not court_points:
        return court_img

    # Accumulate points on a blank canvas
    accumulator = np.zeros((h, w), dtype=np.float32)
    for x, y in court_points:
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < w and 0 <= iy < h:
            cv2.circle(accumulator, (ix, iy), 15, 1.0, -1)

    # Smooth into a heatmap
    accumulator = cv2.GaussianBlur(accumulator, (0, 0), sigmaX=25, sigmaY=25)

    # Normalize to 0-255
    max_val = accumulator.max()
    if max_val > 0:
        accumulator = (accumulator / max_val * 255).astype(np.uint8)
    else:
        accumulator = accumulator.astype(np.uint8)

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(accumulator, cv2.COLORMAP_JET)

    # Create a mask so we only overlay where there is actual heat
    mask = (accumulator > 5).astype(np.float32)
    mask_3ch = np.stack([mask, mask, mask], axis=2)

    # Blend: where there is heat, mix heatmap + court; elsewhere, keep court
    blended = (
        court_img.astype(np.float32) * (1 - mask_3ch * alpha)
        + heatmap_colored.astype(np.float32) * mask_3ch * alpha
    ).astype(np.uint8)

    return blended


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
