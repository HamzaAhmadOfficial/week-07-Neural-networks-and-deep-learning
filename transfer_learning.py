"""
Task 7.4 — Transfer Learning with MobileNetV2

Features:
- Pre-trained MobileNetV2 as base
- Freeze base layers and add custom top layers
- Fine-tune some base layers
- Data augmentation
- Compare with model trained from scratch
- Save models in multiple formats: .h5, .tflite, SavedModel
"""

# ===============================
# IMPORT LIBRARIES
# ===============================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ===============================
# DIRECTORIES
# ===============================

DATASET_DIR = "dataset"
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ===============================
# DATA AUGMENTATION
# ===============================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# ===============================
# LOAD PRE-TRAINED MODEL (MOBILENETV2)
# ===============================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base layers
base_model.trainable = False

# ===============================
# ADD CUSTOM TOP LAYERS
# ===============================

inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)  # Freeze base during first training
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(1, activation='sigmoid')(x)  # Binary classification

model = Model(inputs, outputs)

# ===============================
# COMPILE MODEL
# ===============================

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===============================
# TRAIN MODEL (BASE FROZEN)
# ===============================

history_base = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)

# ===============================
# UNFREEZE SOME BASE LAYERS FOR FINE-TUNING
# ===============================

base_model.trainable = True

# Fine-tune from this layer onward
fine_tune_at = 100  # freeze layers before this
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)

# ===============================
# EVALUATE MODEL
# ===============================

val_generator.reset()
pred_probs = model.predict(val_generator)
pred_labels = (pred_probs > 0.5).astype(int).flatten()
true_labels = val_generator.classes

cm = confusion_matrix(true_labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(val_generator.class_indices.keys()))
disp.plot(cmap="Blues")
plt.savefig(os.path.join(OUTPUT_DIR, "transfer_learning_confusion_matrix.png"))
plt.close()

# ===============================
# SAVE MODEL IN MULTIPLE FORMATS
# ===============================

# 1️⃣ H5
h5_path = os.path.join(MODEL_DIR, "transfer_model.h5")
model.save(h5_path)
print("H5 model saved:", h5_path)

# 2️⃣ SavedModel
saved_model_path = os.path.join(MODEL_DIR, "transfer_model_saved")
model.save(saved_model_path)
print("SavedModel saved:", saved_model_path)

# 3️⃣ TensorFlow Lite (.tflite)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
tflite_path = os.path.join(MODEL_DIR, "transfer_model.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
print("TFLite model saved:", tflite_path)

# ===============================
# PERFORMANCE COMPARISON TABLE
# ===============================

# Collect accuracies
acc_base = history_base.history['val_accuracy'][-1]
acc_finetune = history_finetune.history['val_accuracy'][-1]

print("\nPerformance Comparison:")
print(f"{'Model':<20}{'Validation Accuracy':>20}")
print(f"{'Base MobileNetV2':<20}{acc_base:>20.4f}")
print(f"{'Fine-tuned MobileNetV2':<20}{acc_finetune:>20.4f}")