import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Define directories
TRAIN_FOLDER = 'train_faces/'

# Load and preprocess training data
X_train = []
y_train = []

for filename in os.listdir(TRAIN_FOLDER):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(TRAIN_FOLDER, filename)
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.resize(image, (100, 100))  # Resize image to 100x100
            X_train.append(image)
            y_train.append(filename.split('.')[0])  # Extract label from filename

# Convert lists to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Normalize pixel values to range [0, 1]
X_train = X_train / 255.0

# Convert labels to one-hot encoded vectors
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
num_classes = len(label_encoder.classes_)

# Split data into training and validation sets
X_train, X_val, y_train_encoded, y_val_encoded = train_test_split(X_train, y_train_encoded, test_size=0.2, random_state=42)

# Define CNN model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_encoded, epochs=10, batch_size=32, validation_data=(X_val, y_val_encoded))

# Save the trained model
model.save('face_recognition_model.h5')
