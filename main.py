import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os


# Define image dimensions (modify if needed)
img_width, img_height = 224, 224

# Define paths to training data
train_data_dir = "/home/pi/Project_Rice/train/train_data_dir/rice_bag"
validation_data_dir = "/home/pi/Project_Rice/train/train_data_dir/not_rice_bag"

# Check and print directory contents for debugging
def check_directory(path):
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

print("Training directory structure:")
check_directory(train_data_dir)
print("\nValidation directory structure:")
check_directory(validation_data_dir)

# Define data generators for image augmentation (optional)
train_datagen = ImageDataGenerator(rescale=1./255,  # Normalize pixel values
                                   shear_range=0.2,  # Randomly shear images
                                   zoom_range=0.2,  # Randomly zoom images
                                   horizontal_flip=True)  # Randomly flip images horizontally

validation_datagen = ImageDataGenerator(rescale=1./255)  # Normalize pixel values for validation data

# Load training and validation data using the generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,  # Adjust batch size as needed based on GPU memory
    class_mode='binary'  # Binary classification (rice bag or not)
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary'
)

# Define the model using a Sequential API and Input layer
model = tf.keras.Sequential([
  tf.keras.Input(shape=(img_height, img_width, 3)),
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for rice bag classification
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on the prepared data generators
model.fit(train_generator, epochs=1, validation_data=validation_generator)  # Adjust epochs as needed

# Open video capture using OpenCV
cap = cv2.VideoCapture(0)  # Change 0 to a video file path if needed

if not cap.isOpened():
    print("Error: open camera failed")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Resize the frame
    frame_resized = cv2.resize(frame, (img_width, img_height))

    # Convert the frame to RGB for TensorFlow
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Normalize the frame
    frame_normalized = frame_rgb / 255.0

    # Predict using the model (predict the probability of being a rice bag)
    prediction = model.predict(np.expand_dims(frame_normalized, axis=0))[0][0]

    # Set a threshold for classification (adjust as needed)
    threshold = 0.7
    if prediction > threshold:
        cv2.putText(frame, "Rice Bag", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Not Rice Bag", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
