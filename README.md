# Eye-Movement
Eye and Eyebrow Movement Recognition

# Eye and Eyebrow Movement Recognition

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.8.0%2B-brightgreen.svg)

## 📖 Table of Contents

- [📚 Introduction](#-introduction)
- [🚀 Features](#-features)
- [🔧 Installation](#-installation)
- [💻 Usage](#-usage)
  - [1. Video Capture](#1-video-capture)
  - [2. Frame Extraction](#2-frame-extraction)
  - [3. Data Labeling](#3-data-labeling)
  - [4. Data Preprocessing](#4-data-preprocessing)
  - [5. Dataset Preparation](#5-dataset-preparation)
  - [6. Model Building and Training](#6-model-building-and-training)
  - [7. Model Evaluation](#7-model-evaluation)
  - [8. Real-time Prediction](#8-real-time-prediction)
- [🧠 Model Details](#-model-details)
- [📋 Requirements](#-requirements)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [🙏 Acknowledgements](#-acknowledgements)

## 📚 Introduction

This project is an **Eye and Eyebrow Movement Recognition** system designed to detect and classify three types of facial movements: **Yes**, **No**, and **Normal**. Leveraging deep learning techniques, the system captures short video clips (1-2 seconds) of user movements, processes them, and provides real-time predictions using GPU acceleration via Metal on macOS.

## 🚀 Features

- **Real-time Prediction:** Continuously processes webcam feed to detect eye and eyebrow movements.
- **GPU Acceleration:** Utilizes macOS Metal Performance Shaders (MPS) for faster computations.
- **Customizable Movements:** Currently supports 'Yes', 'No', and 'Normal' movements with the ability to extend to more.
- **User-Friendly Scripts:** Modular Python scripts for each stage of the pipeline, from data collection to prediction.
- **Visual Feedback:** Overlays predictions directly on the video feed for immediate user feedback.

## 🔧 Installation

### 1. Prerequisites

- **Hardware:** Mac with Apple Silicon (M1, M1 Pro, M1 Max, M2, etc.)
- **Operating System:** macOS 12.3 (Monterey) or newer.
- **Python:** Version 3.9 or higher.

### 2. Clone the Repository

```bash
git clone https://github.com/yourusername/eye-eyebrow-movement-recognition.git
cd eye-eyebrow-movement-recognition
```

### 3. Install Homebrew (if not already installed)

Homebrew is a package manager for macOS that simplifies the installation of software.

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 4. Install Micromamba

Micromamba is a lightweight package manager compatible with Conda environments.

```bash
brew install micromamba
```

### 5. Create and Activate a Virtual Environment

We'll use Micromamba to create an isolated environment for our project.

```bash
# Create a new environment named 'eye_movement' with Python 3.9
micromamba create -n eye_movement python=3.9

# Activate the environment
micromamba activate eye_movement
```

### 6. Install Required Libraries

We'll install TensorFlow with Metal support (`tensorflow-macos` and `tensorflow-metal`) along with other necessary libraries.

```bash
# Install TensorFlow for macOS
pip install tensorflow-macos

# Install TensorFlow Metal plugin for GPU acceleration
pip install tensorflow-metal

# Install other dependencies
pip install opencv-python dlib imutils tqdm scikit-learn matplotlib seaborn h5py
```

> **Note:** Installing `dlib` can sometimes be challenging on macOS. If you encounter issues, consider installing it via Conda or refer to [dlib's official installation instructions](http://dlib.net/compile.html).

### 7. Download Dlib's Pre-trained Shape Predictor

This model is essential for facial landmark detection.

```bash

# Download the shape predictor
curl -LO http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

# Decompress the file
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

Ensure that the `shape_predictor_68_face_landmarks.dat` file is in the same directory as your scripts.

## 💻 Usage

The project consists of several scripts, each handling a specific stage of the pipeline.

### 1. Video Capture

**Script:** `video_capture.py`

This script captures short video clips (1-2 seconds) using your webcam. Each video corresponds to a specific movement label.

```bash
python video_capture.py
```

**Steps:**

1. **Run the Script:**

    ```bash
    python video_capture.py
    ```

2. **Input Prompts:**
   - **Movement Label:** Enter a label representing the movement (e.g., `yes`, `no`, `normal`).
   - **Filename:** Enter a filename for the video (e.g., `movement1`).

3. **Recording:**
   - The webcam will start recording. Perform the specified movement.
   - Press `'q'` to stop recording early if needed.

4. **Output:**
   - The recorded video will be saved in `videos/<label>/<filename>.avi`.

**Example Directory Structure After Recording:**

```
videos/
    yes/
        movement1.avi
    no/
        movement2.avi
    normal/
        movement3.avi
    # Add more labels as needed
```

---

### 2. Frame Extraction

**Script:** `frame_extraction.py`

This script extracts frames from each recorded video and organizes them into labeled directories.

```bash
python frame_extraction.py
```

**Steps:**

1. **Ensure Videos are Organized:**
   - Videos should be in the `videos/` directory, organized by label.

2. **Run the Script:**

    ```bash
    python frame_extraction.py
    ```

3. **Output:**
   - Extracted frames will be saved in `frames/<label>/` directories.

**Directory Structure After Extraction:**

```
frames/
    yes/
        movement1_frame_0.jpg
        movement1_frame_1.jpg
        ...
    no/
        movement2_frame_0.jpg
        movement2_frame_1.jpg
        ...
    normal/
        movement3_frame_0.jpg
        ...
    # Add more labels as needed
```

---

### 3. Data Labeling

**Manual Organization**

After frame extraction, organize your dataset as follows:

```
dataset/
    yes/
        movement1/
            frame_0.jpg
            frame_1.jpg
            ...
        movement5/
            frame_0.jpg
            frame_1.jpg
            ...
    no/
        movement2/
            frame_0.jpg
            frame_1.jpg
            ...
    normal/
        movement3/
            frame_0.jpg
            ...
    # Add more classes as needed
```

**Instructions:**

1. **Create `dataset/` Directory:**

    ```bash
    mkdir -p dataset/yes
    mkdir -p dataset/no
    mkdir -p dataset/normal
    ```

2. **Move Extracted Frames to Respective Folders:**

    ```bash
    # For 'yes'
    mkdir -p dataset/yes/movement1
    mv frames/yes/movement1_frame_*.jpg dataset/yes/movement1/
    
    # For 'no'
    mkdir -p dataset/no/movement2
    mv frames/no/movement2_frame_*.jpg dataset/no/movement2/
    
    # For 'normal'
    mkdir -p dataset/normal/movement3
    mv frames/normal/movement3_frame_*.jpg dataset/normal/movement3/
    
    # Repeat for all videos and labels
    ```

**Note:** Ensure that each movement class has a sufficient number of videos (each containing 1-2 seconds of movement) for effective training.

---

### 4. Data Preprocessing

**Script:** `data_preprocessing_sequences.py`

This script preprocesses the data by detecting faces, extracting regions of interest (eyes and eyebrows), resizing, and preparing sequences for the model.

```bash
python data_preprocessing_sequences.py
```

**Steps:**

1. **Ensure Dataset is Organized:**
   - The `dataset/` directory should be structured as described in the Data Labeling section.

2. **Run the Script:**

    ```bash
    python data_preprocessing_sequences.py
    ```

3. **Output:**
   - Preprocessed sequences will be saved in `preprocessed_sequences/<label>/<sequence>.npy` directories.

**Directory Structure After Preprocessing:**

```
preprocessed_sequences/
    yes/
        movement1.npy
        movement5.npy
        ...
    no/
        movement2.npy
        ...
    normal/
        movement3.npy
        ...
    # Add more classes as needed
```

Each `.npy` file contains a 3D NumPy array representing the sequence of frames for a single movement.

---

### 5. Dataset Preparation

**Script:** `dataset_preparation_sequences.py`

This script loads the preprocessed sequences, pads/truncates them to a fixed length, assigns labels, and splits the data into training and testing sets.

```bash
python dataset_preparation_sequences.py
```

**Steps:**

1. **Ensure Preprocessed Sequences are Ready:**
   - The `preprocessed_sequences/` directory should be populated as per the previous step.

2. **Run the Script:**

    ```bash
    python dataset_preparation_sequences.py
    ```

3. **Output:**
   - The script will output the total number of samples, maximum sequence length, and shapes of padded sequences.
   - It will split the data into training and testing sets and save them as `dataset_sequences.pkl`.

**Expected Output:**

```
Total samples: 100
Maximum sequence length: 30
Padded sequences shape: (100, 30, 64, 256, 1)
Training samples: 80
Testing samples: 20
Dataset saved to dataset_sequences.pkl.
```

*(Note: The actual numbers will vary based on your dataset.)*

---

### 6. Model Building and Training

**Script:** `model_building_sequences.py`

This script builds and trains a CNN-LSTM model to handle both spatial and temporal features of the movement sequences, leveraging GPU acceleration via Metal.

```bash
python model_building_sequences.py
```

**Steps:**

1. **Ensure Dataset is Prepared:**
   - The `dataset_sequences.pkl` file should be present in the project directory.

2. **Run the Script:**

    ```bash
    python model_building_sequences.py
    ```

3. **Output:**
   - The script will build, train, and save the model as `final_model_sequences.keras`.
   - The best model based on validation accuracy is saved as `best_model_sequences.keras`.
   - Training history is saved as `history_sequences.pkl`.

**Notes:**

- **Batch Size:** Adjust `batch_size` based on your system's memory. Smaller batch sizes are more memory-efficient but may require more epochs.
- **Epochs:** Set to a high number (e.g., 50) with early stopping to prevent overfitting.
- **Sequence Length:** Ensure that the `max_seq_length` used during padding matches the expected input for the model.

---

### 7. Model Evaluation

**Script:** `model_evaluation_sequences.py`

This script evaluates the trained model's performance on the test set and visualizes the training history.

```bash
python model_evaluation_sequences.py
```

**Steps:**

1. **Ensure Models are Saved:**
   - `best_model_sequences.keras` and `history_sequences.pkl` should be present.

2. **Run the Script:**

    ```bash
    python model_evaluation_sequences.py
    ```

3. **Output:**
   - **Test Accuracy and Loss**
   - **Confusion Matrix:** Visual representation of true vs. predicted labels.
   - **Classification Report:** Detailed metrics for each class, including precision, recall, and F1-score.
   - **Training History Plots:** Graphs of training and validation accuracy and loss over epochs.

**Example Output:**

```
Test Accuracy: 85.00%
Test Loss: 0.5678
```

*(Plus the confusion matrix, classification report, and training history plots.)*

---

### 8. Real-time Prediction

**Script:** `prediction_sequences.py`

This script uses the trained model to predict movements from new video sequences and converts them into meaningful text in real-time, leveraging GPU acceleration via Metal.

```bash
python prediction_sequences.py
```

**Steps:**

1. **Ensure Models are Saved:**
   - `final_model_sequences.keras` and `shape_predictor_68_face_landmarks.dat` should be present.

2. **Run the Script:**

    ```bash
    python prediction_sequences.py
    ```

3. **Operation:**
   - The webcam will start capturing video.
   - Perform the specified eye and eyebrow movements.
   - The script will continuously display the video feed with real-time predictions overlaid.
   - Press `'q'` to quit the application.

**Notes:**

- **Sequence Length:** Ensure that the `sequence_length` parameter matches the number of frames used during training. Adjust `max_seq_length` during preprocessing accordingly.
- **Frame Rate:** The number of frames captured per second (fps) affects the sequence length. Ensure consistency between training and prediction.
- **Real-Time Performance:** Processing sequences can introduce latency. Optimize the model and code for real-time performance as needed.

---

## 🧠 Model Details

### **Supported Movements**

- **Yes:** Eyebrow Raised
- **No:** Eyebrow Lowered
- **Normal:** Neutral facial expression

### **Model Architecture**

The model employs a **CNN-LSTM** architecture to capture both spatial and temporal features:

1. **TimeDistributed CNN Layers:**
   - Extract spatial features from each frame.
   - Consist of convolutional, pooling, and batch normalization layers.

2. **LSTM Layer:**
   - Captures temporal dependencies across the sequence of frames.

3. **Dense Layers:**
   - Fully connected layers that perform the final classification.

### **Performance Metrics**

- **Accuracy:** 85%
- **Loss:** 0.5678
- **Confusion Matrix:** Demonstrates model performance across different classes.
- **Classification Report:** Provides precision, recall, and F1-score for each class.

*(Note: Metrics are illustrative. Actual performance may vary based on dataset size and quality.)*

---

## 📋 Requirements

- **Python:** 3.9 or higher
- **Libraries:**
  - TensorFlow-MacOS
  - TensorFlow-Metal
  - OpenCV-Python
  - dlib
  - imutils
  - tqdm
  - scikit-learn
  - matplotlib
  - seaborn
  - h5py

*(All dependencies are listed in the Installation section.)*

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps to contribute:

1. **Fork the Repository**
2. **Create a New Branch:**

    ```bash
    git checkout -b feature/YourFeatureName
    ```

3. **Commit Your Changes:**

    ```bash
    git commit -m "Add Your Feature"
    ```

4. **Push to the Branch:**

    ```bash
    git push origin feature/YourFeatureName
    ```

5. **Open a Pull Request**

Please ensure that your contributions adhere to the project guidelines and include relevant documentation and tests.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [TensorFlow](https://www.tensorflow.org/)
- [OpenCV](https://opencv.org/)
- [dlib](http://dlib.net/)
- [imutils](https://github.com/jrosebr1/imutils)
- [Hugging Face](https://huggingface.co/)
- [Metal Performance Shaders (MPS)](https://developer.apple.com/documentation/metalperformanceshaders)
```
