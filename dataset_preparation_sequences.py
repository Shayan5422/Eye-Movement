# dataset_preparation_sequences.py

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import pickle

def load_sequences(preprocessed_dir='preprocessed_sequences'):
    """
    Loads preprocessed sequences and their labels.

    Args:
        preprocessed_dir (str): Directory containing preprocessed sequences.

    Returns:
        tuple: Lists of sequences and labels, label mapping dictionary.
    """
    X = []
    y = []
    label_map = {}
    classes = sorted(os.listdir(preprocessed_dir))
    
    for idx, cls in enumerate(classes):
        label_map[cls] = idx
        cls_path = os.path.join(preprocessed_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        sequence_files = [f for f in os.listdir(cls_path) if f.endswith('.npy')]
        for seq_file in sequence_files:
            seq_path = os.path.join(cls_path, seq_file)
            sequence = np.load(seq_path)
            X.append(sequence)
            y.append(idx)
    
    # X remains a list of numpy arrays with varying shapes
    y = np.array(y)
    y = to_categorical(y, num_classes=len(label_map))
    
    return X, y, label_map

def pad_sequences_fixed(X, max_seq_length):
    """
    Pads or truncates sequences to a fixed length.

    Args:
        X (list of numpy.ndarray): List of sequences with shape (frames, height, width, channels).
        max_seq_length (int): Desired sequence length.

    Returns:
        numpy.ndarray: Padded/truncated sequences.
    """
    padded_X = []
    for seq in X:
        if seq.shape[0] < max_seq_length:
            pad_width = max_seq_length - seq.shape[0]
            padding = np.zeros((pad_width, *seq.shape[1:]), dtype=seq.dtype)
            padded_seq = np.concatenate((seq, padding), axis=0)
        else:
            padded_seq = seq[:max_seq_length]
        padded_X.append(padded_seq)
    return np.array(padded_X)

def save_dataset(X_train, X_test, y_train, y_test, label_map, output_path='dataset_sequences.pkl'):
    """
    Saves the dataset into a pickle file.

    Args:
        X_train, X_test, y_train, y_test: Split data.
        label_map (dict): Mapping from class names to indices.
        output_path (str): Path to save the pickle file.
    """
    with open(output_path, 'wb') as f:
        pickle.dump({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'label_map': label_map
        }, f)
    print(f"Dataset saved to {output_path}.")

def load_dataset_pickle(pickle_path='dataset_sequences.pkl'):
    """
    Loads the dataset from a pickle file.

    Args:
        pickle_path (str): Path to the pickle file.

    Returns:
        tuple: Split data and label mapping.
    """
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data['X_train'], data['X_test'], data['y_train'], data['y_test'], data['label_map']

if __name__ == "__main__":
    # Load sequences
    X, y, label_map = load_sequences(preprocessed_dir='preprocessed_sequences')
    print(f"Total samples: {len(X)}")
    
    # Find the maximum sequence length for padding
    max_seq_length = max([seq.shape[0] for seq in X])
    print(f"Maximum sequence length: {max_seq_length}")
    
    # Pad sequences to have the same length
    X_padded = pad_sequences_fixed(X, max_seq_length)
    print(f"Padded sequences shape: {X_padded.shape}")
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    # Save the dataset
    save_dataset(X_train, X_test, y_train, y_test, label_map, output_path='dataset_sequences.pkl')
