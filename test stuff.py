from pathlib import Path

from yt_dlp import YoutubeDL

link = "https://www.youtube.com/watch?v=-jtV7IJP8NU"
output_dir = Path("./clean_videos")
output_dir.mkdir(parents=True, exist_ok=True)

ydl_opts = {
    "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
    # Force H.264/AVC video for broad player compatibility (avoids AV1 black-screen playback issues).
    # Audio is optional per request.
    "format": "bestvideo[vcodec^=avc1][ext=mp4]/best[ext=mp4][vcodec^=avc1]/18",
    "noplaylist": True,
    "overwrites": True,
}

with YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(link, download=False)
    print(f"Title: {info.get('title')}")
    print(f"Uploader: {info.get('uploader')}")
    print("Starting download...")
    ydl.download([link])