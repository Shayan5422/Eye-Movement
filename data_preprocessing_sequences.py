# data_preprocessing_sequences.py

import os
import cv2
import dlib
import numpy as np
from imutils import face_utils
from tqdm import tqdm
import pickle

def get_facial_landmarks(detector, predictor, image):
    """
    Detects facial landmarks in an image.

    Args:
        detector: dlib face detector.
        predictor: dlib shape predictor.
        image (numpy.ndarray): Input image.

    Returns:
        dict: Coordinates of eyes and eyebrows.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    
    if len(rects) == 0:
        return None  # No face detected
    
    # Assuming the first detected face is the target
    rect = rects[0]
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    
    landmarks = {}
    # Define landmarks for left and right eyes and eyebrows
    landmarks['left_eye'] = shape[36:42]      # Left eye landmarks
    landmarks['right_eye'] = shape[42:48]     # Right eye landmarks
    landmarks['left_eyebrow'] = shape[17:22]  # Left eyebrow landmarks
    landmarks['right_eyebrow'] = shape[22:27] # Right eyebrow landmarks
    
    return landmarks

def extract_roi(image, landmarks, region='left_eye', padding=5):
    """
    Extracts a region of interest (ROI) from the image based on landmarks.

    Args:
        image (numpy.ndarray): Input image.
        landmarks (dict): Facial landmarks.
        region (str): Region to extract ('left_eye', 'right_eye', 'left_eyebrow', 'right_eyebrow').
        padding (int): Padding around the ROI.

    Returns:
        numpy.ndarray: Extracted ROI.
    """
    points = landmarks.get(region)
    if points is None:
        return None
    
    # Compute the bounding box
    x, y, w, h = cv2.boundingRect(points)
    x = max(x - padding, 0)
    y = max(y - padding, 0)
    w = w + 2 * padding
    h = h + 2 * padding
    
    roi = image[y:y+h, x:x+w]
    return roi

def preprocess_video_sequence(sequence_dir, detector, predictor, img_size=(64, 64)):
    """
    Preprocesses a sequence of frames from a video.

    Args:
        sequence_dir (str): Directory containing frames of a video.
        detector: dlib face detector.
        predictor: dlib shape predictor.
        img_size (tuple): Desired image size for ROIs.

    Returns:
        list: List of preprocessed frames as numpy arrays.
    """
    frames = sorted([f for f in os.listdir(sequence_dir) if f.endswith('.jpg') or f.endswith('.png')])
    preprocessed_sequence = []
    
    for frame_name in frames:
        frame_path = os.path.join(sequence_dir, frame_name)
        image = cv2.imread(frame_path)
        if image is None:
            continue
        
        landmarks = get_facial_landmarks(detector, predictor, image)
        if landmarks is None:
            continue  # Skip frames with no detected face
        
        # Extract ROIs for eyes and eyebrows
        rois = {}
        rois['left_eye'] = extract_roi(image, landmarks, 'left_eye')
        rois['right_eye'] = extract_roi(image, landmarks, 'right_eye')
        rois['left_eyebrow'] = extract_roi(image, landmarks, 'left_eyebrow')
        rois['right_eyebrow'] = extract_roi(image, landmarks, 'right_eyebrow')
        
        # Process ROIs
        roi_images = []
        for region in ['left_eye', 'right_eye', 'left_eyebrow', 'right_eyebrow']:
            roi = rois.get(region)
            if roi is not None:
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                roi = cv2.resize(roi, img_size)
                roi = roi.astype('float32') / 255.0        # Normalize to [0,1]
                roi = np.expand_dims(roi, axis=-1)          # Add channel dimension
                roi_images.append(roi)
        
        if len(roi_images) == 0:
            continue  # Skip if no ROIs were extracted
        
        # Concatenate ROIs horizontally to form a single image
        combined_roi = np.hstack(roi_images)
        preprocessed_sequence.append(combined_roi)
    
    return preprocessed_sequence

def preprocess_dataset(dataset_dir='dataset', output_dir='preprocessed_sequences', img_size=(64, 64)):
    """
    Preprocesses the entire dataset by processing each video sequence.

    Args:
        dataset_dir (str): Directory containing labeled data.
        output_dir (str): Directory to save preprocessed sequences.
        img_size (tuple): Desired image size for ROIs.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize dlib's face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    
    if not os.path.exists(predictor_path):
        print(f"Error: {predictor_path} not found. Download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        return
    
    predictor = dlib.shape_predictor(predictor_path)
    
    classes = os.listdir(dataset_dir)
    for cls in classes:
        cls_path = os.path.join(dataset_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        output_cls_dir = os.path.join(output_dir, cls)
        if not os.path.exists(output_cls_dir):
            os.makedirs(output_cls_dir)
        
        print(f"Processing class: {cls}")
        sequences = os.listdir(cls_path)
        for seq in tqdm(sequences, desc=f"Class {cls}"):
            seq_path = os.path.join(cls_path, seq)
            if not os.path.isdir(seq_path):
                continue
            preprocessed_sequence = preprocess_video_sequence(seq_path, detector, predictor, img_size=img_size)
            if len(preprocessed_sequence) == 0:
                continue  # Skip sequences with no valid frames
            
            # Stack frames to form a 3D array (frames, height, width, channels)
            sequence_array = np.stack(preprocessed_sequence, axis=0)
            
            # Save the preprocessed sequence as a numpy file
            npy_filename = os.path.join(output_cls_dir, f"{seq}.npy")
            np.save(npy_filename, sequence_array)
    
    print("Data preprocessing completed.")

if __name__ == "__main__":
    preprocess_dataset(dataset_dir='dataset', output_dir='preprocessed_sequences', img_size=(64, 64))
