# model_evaluation_sequences.py

import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def load_model(model_path='best_model_sequences.keras'):
    """
    Loads the trained model.

    Args:
        model_path (str): Path to the saved model.

    Returns:
        tensorflow.keras.Model: Loaded model.
    """
    model = tf.keras.models.load_model(model_path)
    return model

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

def plot_history(history):
    """
    Plots the training and validation accuracy and loss.

    Args:
        history (dict): Training history.
    """
    acc = history.get('accuracy', history.get('acc'))
    val_acc = history.get('val_accuracy', history.get('val_acc'))
    
    loss = history['loss']
    val_loss = history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(14,5))
    
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

def evaluate_model(model, X_test, y_test, label_map):
    """
    Evaluates the model on the test set.

    Args:
        model (tensorflow.keras.Model): Trained model.
        X_test (numpy.ndarray): Test sequences.
        y_test (numpy.ndarray): Test labels.
        label_map (dict): Mapping from class names to indices.
    """
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Loss: {loss:.4f}")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_map.keys(), yticklabels=label_map.keys(), cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Classification Report
    print("Classification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=label_map.keys()))

def main():
    # Load the trained model
    model = load_model('best_model_sequences.keras')
    
    # Load the dataset
    X_train, X_test, y_train, y_test, label_map = load_dataset_pickle('dataset_sequences.pkl')
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test, label_map)
    
    # Load and plot training history
    try:
        with open('history_sequences.pkl', 'rb') as f:
            history = pickle.load(f)
        plot_history(history)
    except FileNotFoundError:
        print("Training history not found. Skipping plotting.")

if __name__ == "__main__":
    main()
