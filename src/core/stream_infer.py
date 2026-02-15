import cv2
import supervision as sv

from src.core.ball_detector import BallDetector
from src.core.bounce_detector import BounceDetector
from src.core.court_detection_net import CourtDetectorNet
from src.core.person_detector import PersonDetector


def get_ball_track_and_bounces_stream_infer(
    video_path: str, device: str
) -> tuple[list[tuple[float, float]], set[int]]:
    ball_detector = BallDetector("./src/track_net_weights.pt", device)
    bounce_detector = BounceDetector("./src/ctb_regr_bounce.cbm")

    ball_track = []
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        ball_track.extend(ball_detector.infer_model([frame]))

    x_ball = [x[0] for x in ball_track]
    y_ball = [x[1] for x in ball_track]
    bounces = bounce_detector.predict(x_ball, y_ball)

    return ball_track, bounces


def court_detector_stream_infer(video_path: str, device: str):
    court_detector = CourtDetectorNet("./src/model_tennis_court_det.pt", device)

    cap = cv2.VideoCapture(video_path)
    homography_matrices = []
    kps_court = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        homography_matrix, kps_court = court_detector.infer_model([frame])
        homography_matrices.extend(homography_matrix)
        kps_court.extend(kps_court)

    return homography_matrices, kps_court


# def person_detector_stream_infer(video_path: str, device: str):
#     person_detector = PersonDetector(device)

#     with sv.VideoStream(video_path) as video_stream:
#         persons_top = []
#         persons_bottom = []
#         for frame in video_stream:
#             persons_top, persons_bottom = person_detector.track_players(
#                 [frame], homography_matrices, filter_players=False
#             )
#             persons_top.extend(persons_top)
#             persons_bottom.extend(persons_bottom)

#         return persons_top, persons_bottom
