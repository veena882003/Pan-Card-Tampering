import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths to the dataset directories
root_dir = r'C:\Users\Veena Yadav\Downloads\BACKEND'
tampered_dir = os.path.join(root_dir, 'tampered')
not_tampered_dir = os.path.join(root_dir, 'non_tampered')

# Load and preprocess the dataset
X = []
y = []

# Function to load and preprocess images
def preprocess_image(image_path, label):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    image = tf.keras.preprocessing.image.img_to_array(image)
    X.append(image)
    y.append(label)

# Load tampered images
for image_file in os.listdir(tampered_dir):
    image_path = os.path.join(tampered_dir, image_file)
    preprocess_image(image_path, 1)  # Label for tampered images

# Load non-tampered images
for image_file in os.listdir(not_tampered_dir):
    image_path = os.path.join(not_tampered_dir, image_file)
    preprocess_image(image_path, 0)  # Label for non-tampered images

X = np.array(X) / 255.0  # Normalize pixel values
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 , random_state=5)

# Data Augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),  # Dropout layer for regularization
    layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Learning Rate Scheduling
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.7, patience=3, verbose=1)

# Train the model
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=32),
    epochs=30,  # Increased number of epochs
    validation_data=(X_test, y_test),
    callbacks=[lr_scheduler]
)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")
print(f"Test loss: {test_loss}")

model.save("pan_tamper_detection_model.h5")


