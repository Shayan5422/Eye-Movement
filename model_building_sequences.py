# model_building_sequences.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, TimeDistributed, LSTM, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle

def build_cnn_lstm_model(input_shape, num_classes):
    """
    Builds a CNN-LSTM model for sequence classification.

    Args:
        input_shape (tuple): Shape of input sequences (frames, height, width, channels).
        num_classes (int): Number of output classes.

    Returns:
        tensorflow.keras.Model: Compiled model.
    """
    model = Sequential()
    
    # Apply Conv2D to each frame in the sequence
    model.add(TimeDistributed(Conv2D(32, (3,3), activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling2D((2,2))))
    model.add(TimeDistributed(BatchNormalization()))
    
    # Additional Conv2D layers
    model.add(TimeDistributed(Conv2D(64, (3,3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2,2))))
    model.add(TimeDistributed(BatchNormalization()))
    
    # Flatten the output from Conv layers
    model.add(TimeDistributed(Flatten()))
    
    # LSTM layer to capture temporal dependencies
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.5))
    
    # Fully connected layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    
    # Output layer with softmax activation for classification
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile the model with Adam optimizer and categorical cross-entropy loss
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
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

def main():
    # Load the dataset
    X_train, X_test, y_train, y_test, label_map = load_dataset_pickle('dataset_sequences.pkl')
    num_classes = y_train.shape[1]
    input_shape = X_train.shape[1:]  # (frames, height, width, channels)
    
    # Build the CNN-LSTM model
    model = build_cnn_lstm_model(input_shape, num_classes)
    model.summary()
    
    # Define callbacks with updated filepath (.keras)
    checkpoint = ModelCheckpoint(
        'best_model_sequences.keras',  # Changed from .h5 to .keras
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )
    
    # Train the model using GPU
    with tf.device('/GPU:0'):
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=128,  # Adjust based on your system's memory
            validation_data=(X_test, y_test)
        )
    
    # Save the final trained model with .keras extension
    model.save('final_model_sequences.keras')  # Changed from .h5 to .keras
    print("Model training completed and saved as 'final_model_sequences.keras'.")
    
    # Save training history for future reference
    with open('history_sequences.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    print("Training history saved as 'history_sequences.pkl'.")

if __name__ == "__main__":
    main()
