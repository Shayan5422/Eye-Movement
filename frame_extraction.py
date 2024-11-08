# frame_extraction.py

import cv2
import os
from tqdm import tqdm

def extract_frames(video_path, output_dir='frames', prefix='frame'):
    """
    Extracts frames from a video file.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save extracted frames.
        prefix (str): Prefix for the frame filenames.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return
    
    frame_count = 0
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for _ in tqdm(range(frame_total), desc=f"Extracting frames from {os.path.basename(video_path)}"):
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_dir, f"{prefix}_frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)  # Save frame as JPEG file
        frame_count += 1
    
    cap.release()
    print(f"Extracted {frame_count} frames from {video_path}.")

def process_all_videos(videos_dir='videos', frames_dir='frames'):
    """
    Processes all videos in the specified directory and extracts frames.

    Args:
        videos_dir (str): Directory containing video files.
        frames_dir (str): Directory to save extracted frames.
    """
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    
    video_files = [f for f in os.listdir(videos_dir) if f.endswith('.avi') or f.endswith('.mp4')]
    
    for video_file in video_files:
        label = video_file.split('_')[0]  # Assuming filename format 'label_something.avi'
        video_path = os.path.join(videos_dir, video_file)
        output_subdir = os.path.join(frames_dir, label)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        extract_frames(video_path, output_dir=output_subdir, prefix=video_file.split('.')[0])

if __name__ == "__main__":
    process_all_videos(videos_dir='videos', frames_dir='frames')
