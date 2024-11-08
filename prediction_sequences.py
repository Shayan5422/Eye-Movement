# prediction_sequences.py

import tensorflow as tf
import cv2
import numpy as np
import dlib
from imutils import face_utils
import os
import pickle
from collections import deque
import threading
import queue
import time

def load_model(model_path='final_model_sequences.keras'):
    """
    Loads the trained model.

    Args:
        model_path (str): Path to the saved model.

    Returns:
        tensorflow.keras.Model: Loaded model.
    """
    model = tf.keras.models.load_model(model_path)
    return model

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

def preprocess_frame(image, detector, predictor, img_size=(64, 64)):
    """
    Preprocesses a single frame: detects landmarks, extracts ROIs, and prepares the input.

    Args:
        image (numpy.ndarray): Input frame.
        detector: dlib face detector.
        predictor: dlib shape predictor.
        img_size (tuple): Desired image size for ROIs.

    Returns:
        numpy.ndarray: Preprocessed frame as a concatenated ROI image.
    """
    landmarks = get_facial_landmarks(detector, predictor, image)
    if landmarks is None:
        return None  # No face detected

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
            roi = np.expand_dims(roi, axis=-1)         # Add channel dimension
            roi_images.append(roi)

    if len(roi_images) == 0:
        return None  # No ROIs extracted

    # Concatenate ROIs horizontally to form a single image
    combined_roi = np.hstack(roi_images)
    return combined_roi

def movement_to_text(label_map):
    """
    Creates a mapping from class indices to text.

    Args:
        label_map (dict): Mapping from class names to indices.

    Returns:
        dict: Mapping from indices to text descriptions.
    """
    movement_to_text_map = {
        'upward_eyebrow': 'Eyebrow Raised',
        'downward_eyebrow': 'Eyebrow Lowered',
        'left_eye': 'Left Eye Movement',
        'right_eye': 'Right Eye Movement',
        # Add more mappings as needed
    }

    # Create index to text mapping
    index_to_text = {}
    for cls, idx in label_map.items():
        text = movement_to_text_map.get(cls, cls)
        index_to_text[idx] = text
    return index_to_text

def prediction_worker(model, input_queue, output_queue, max_seq_length):
    """
    Worker thread for handling model predictions.

    Args:
        model (tensorflow.keras.Model): Trained model.
        input_queue (queue.Queue): Queue to receive sequences for prediction.
        output_queue (queue.Queue): Queue to send prediction results.
        max_seq_length (int): Fixed sequence length for the model.
    """
    while True:
        sequence = input_queue.get()
        if sequence is None:
            break  # Sentinel to stop the thread

        # Pad or truncate the sequence to match the model's expected input
        if sequence.shape[0] < max_seq_length:
            pad_width = max_seq_length - sequence.shape[0]
            padding = np.zeros((pad_width, *sequence.shape[1:]), dtype=sequence.dtype)
            sequence_padded = np.concatenate((sequence, padding), axis=0)
        else:
            sequence_padded = sequence[:max_seq_length]

        # Expand dimensions to match model input (1, frames, height, width, channels)
        sequence_padded = np.expand_dims(sequence_padded, axis=0)

        # Perform prediction
        prediction = model.predict(sequence_padded)
        class_idx = np.argmax(prediction)
        confidence = np.max(prediction)

        # Put the result in the output queue
        output_queue.put((class_idx, confidence))

def main():
    # Load the trained model
    model = load_model('final_model_sequences.keras')

    # Load label map
    with open('dataset_sequences.pkl', 'rb') as f:
        data = pickle.load(f)
    label_map = data['label_map']
    index_to_text = movement_to_text(label_map)

    # Initialize dlib's face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor_path = 'shape_predictor_68_face_landmarks.dat'

    if not os.path.exists(predictor_path):
        print(f"Error: {predictor_path} not found. Download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        return

    predictor = dlib.shape_predictor(predictor_path)

    # Initialize queues for communication between threads
    input_queue = queue.Queue()
    output_queue = queue.Queue()

    # Define sequence length (number of frames)
    max_seq_length = 20  # Adjust based on your training data

    # Start the prediction worker thread
    pred_thread = threading.Thread(target=prediction_worker, args=(model, input_queue, output_queue, max_seq_length))
    pred_thread.daemon = True
    pred_thread.start()

    # Start video capture
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting real-time prediction. Press 'q' to quit.")

    # Initialize a deque to store the sequence of preprocessed frames
    frame_buffer = deque(maxlen=max_seq_length)

    # Variable to store the latest prediction result
    latest_prediction = "Initializing..."

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Preprocess the current frame
        preprocessed_frame = preprocess_frame(frame, detector, predictor, img_size=(64, 64))
        if preprocessed_frame is not None:
            frame_buffer.append(preprocessed_frame)
        else:
            # If no face detected, append a zero array to maintain sequence length
            frame_buffer.append(np.zeros((64, 256, 1), dtype='float32'))

        # If the buffer is full, send the sequence to the prediction thread
        if len(frame_buffer) == max_seq_length:
            # Convert deque to numpy array
            sequence_array = np.array(frame_buffer)
            input_queue.put(sequence_array)

        # Check if there's a new prediction result
        try:
            while True:
                class_idx, confidence = output_queue.get_nowait()
                movement = index_to_text.get(class_idx, "Unknown")
                latest_prediction = f"{movement} ({confidence*100:.2f}%)"
        except queue.Empty:
            pass  # No new prediction

        # Display the prediction on the frame
        cv2.putText(frame, latest_prediction, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Real-time Movement Prediction', frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Stop the prediction thread
    input_queue.put(None)  # Sentinel to stop the thread
    pred_thread.join()

if __name__ == "__main__":
    main()
