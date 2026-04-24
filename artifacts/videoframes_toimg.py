import cv2
import os
from pathlib import Path

video_Path = Path("data/raw_videos/ASL_dynamic/S")
clip_file = sorted(video_Path.glob("*.avi"))

def capture_frames():
    for clip in clip_file:
        openClips = cv2.VideoCapture(clip)
        folder_name = clip.stem + "_frames"
        frame_dir = video_Path / folder_name
        if not openClips.isOpened():
            continue

        if not frame_dir.exists():
            frame_dir.mkdir()

        frame_count = 0
        while True:
            success, frame = openClips.read()

            if not success:
                break

            frame_name = frame_dir / f"frame_{frame_count:04d}.jpg"
            cv2.imwrite(frame_name, frame)
            frame_count += 1

        openClips.release()

    for filename in os.listdir(video_Path):
        print(filename)

capture_frames()
