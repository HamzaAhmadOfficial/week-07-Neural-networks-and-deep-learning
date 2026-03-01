"""
Task 7.3 — CNN Image Classification (MNIST)

Features:
- CNN architecture
- Data augmentation
- Filter visualization
- Feature map visualization
- Confusion matrix evaluation
- TensorFlow Lite (.tflite) export
"""

# ===============================
# IMPORT LIBRARIES
# ===============================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ===============================
# CREATE DIRECTORIES
# ===============================

OUTPUT_DIR = "outputs"
MODEL_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ===============================
# LOAD DATASET (MNIST)
# ===============================

print("\nLoading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# ===============================
# RESHAPE & NORMALIZE DATA
# ===============================

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# ===============================
# DATA AUGMENTATION
# ===============================

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(X_train)

# ===============================
# BUILD CNN ARCHITECTURE
# ===============================

model = Sequential([
    Input(shape=(28, 28, 1)),

    Conv2D(32, (3, 3), activation="relu", name="conv1"),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu", name="conv2"),
    MaxPooling2D((2, 2)),

    Flatten(),

    Dense(128, activation="relu"),
    Dropout(0.3),

    Dense(10, activation="softmax")
])

# ===============================
# COMPILE MODEL
# ===============================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===============================
# TRAIN MODEL
# ===============================

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=10,
    validation_data=(X_test, y_test),
    verbose=1
)

# ===============================
# VISUALIZE FIRST LAYER FILTERS
# ===============================

print("\nVisualizing filters...")

filters, biases = model.get_layer("conv1").get_weights()

fig, axes = plt.subplots(4, 8, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(filters[:, :, 0, i], cmap="gray")
    ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cnn_filters.png"))
plt.close()

# ===============================
# FEATURE MAP VISUALIZATION FUNCTION
# ===============================

def visualize_feature_maps(model, image):
    """
    Visualize the feature maps of all Conv2D layers for a given input image.
    """
    # Ensure model is built by calling it once
    _ = model.predict(image[:1])

    # Get outputs of convolutional layers only
    layer_outputs = [layer.output for layer in model.layers if isinstance(layer, Conv2D)]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)

    activations = activation_model.predict(image)

    for layer_index, activation in enumerate(activations):
        n_features = activation.shape[-1]

        plt.figure(figsize=(12, 6))
        for i in range(min(n_features, 16)):
            plt.subplot(4, 4, i + 1)
            plt.imshow(activation[0, :, :, i], cmap="gray")
            plt.axis("off")

        plt.savefig(
            os.path.join(OUTPUT_DIR, f"feature_maps_layer_{layer_index}.png")
        )
        plt.close()

# Visualize feature maps using one sample
sample_image = X_test[0].reshape(1, 28, 28, 1)
visualize_feature_maps(model, sample_image)

# ===============================
# MODEL EVALUATION
# ===============================

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {acc:.4f}")

# ===============================
# CONFUSION MATRIX
# ===============================

pred_probs = model.predict(X_test)
pred_labels = np.argmax(pred_probs, axis=1)

cm = confusion_matrix(y_test, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")

plt.savefig(os.path.join(OUTPUT_DIR, "cnn_confusion_matrix.png"))
plt.close()

# ===============================
# SAVE MODEL
# ===============================

h5_path = os.path.join(MODEL_DIR, "cnn_model.h5")
model.save(h5_path)
print("Model saved:", h5_path)

# ===============================
# CONVERT TO TFLITE
# ===============================

print("\nConverting to TensorFlow Lite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_path = os.path.join(MODEL_DIR, "cnn_model.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print("TFLite model saved:", tflite_path)