# video_capture.py

import cv2
import os

def record_video(duration=2, output_dir='videos', filename='sample'):
    """
    Records a short video from the webcam.

    Args:
        duration (int): Duration of the video in seconds.
        output_dir (str): Directory to save the videos.
        filename (str): Name of the output video file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(0)  # Initialize webcam
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Get default camera resolution
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20  # Frames per second
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(os.path.join(output_dir, f"{filename}.avi"), fourcc, fps, (frame_width, frame_height))
    
    print("Recording started. Press 'q' to stop early.")
    
    frame_count = 0
    total_frames = duration * fps
    
    while frame_count < total_frames:
        ret, frame = cap.read()
        if ret:
            out.write(frame)  # Write frame to video file
            cv2.imshow('Recording', frame)
            frame_count += 1
            
            # Press 'q' to quit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Failed to grab frame.")
            break
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Recording finished. Video saved as {filename}.avi")

if __name__ == "__main__":
    # Example: Record a 2-second video named 'movement1'
    label = input("Enter movement label (e.g., 'upward_eyebrow'): ")
    filename = input("Enter filename (e.g., 'movement1'): ")
    record_video(duration=2, filename=filename)