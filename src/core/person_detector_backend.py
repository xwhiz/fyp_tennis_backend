"""Stable string ids for person detector selection (Redis, .env, admin UI)."""

PERSON_DETECTOR_BACKEND_YOLO26X = "yolo26x"
PERSON_DETECTOR_BACKEND_FASTER_RCNN_RESNET50 = "fasterrcnn_resnet50"

_VALID = frozenset(
    {
        PERSON_DETECTOR_BACKEND_YOLO26X,
        PERSON_DETECTOR_BACKEND_FASTER_RCNN_RESNET50,
    }
)


def normalize_person_detector_backend(value: str) -> str:
    v = (value or "").strip().lower()
    if v in _VALID:
        return v
    if v in ("yolo", "yolo26"):
        return PERSON_DETECTOR_BACKEND_YOLO26X
    if v in ("fasterrcnn", "resnet", "faster_rcnn", "fasterrcnn_resnet50_fpn"):
        return PERSON_DETECTOR_BACKEND_FASTER_RCNN_RESNET50
    raise ValueError(
        f"Unknown person_detector backend {value!r}; "
        f"expected one of {sorted(_VALID)}",
    )
