import os

from src.config import settings

PROFILE_IMAGE_DIR = settings.profile_image_dir


def ensure_profile_image_dir() -> None:
    os.makedirs(PROFILE_IMAGE_DIR, exist_ok=True)


def profile_image_url(profile_image_path: str | None) -> str | None:
    if not profile_image_path:
        return None
    return f"/stream/profile-image/{os.path.basename(profile_image_path)}"

