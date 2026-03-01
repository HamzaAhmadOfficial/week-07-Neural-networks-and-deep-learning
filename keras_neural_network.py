"""
Task 7.2 — Neural Network using TensorFlow/Keras

Features:
- Sequential API (Best Practice Input Layer)
- Dense Neural Network
- EarlyStopping & ModelCheckpoint
- Training history plots
- Save model (.h5 + SavedModel)
- Save scaler (production pipeline)
- Load & test saved model
- Confusion Matrix & Classification Report
- Outputs & Models folders
"""

# IMPORT LIBRARIES

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# REPRODUCIBILITY 

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# CREATE DIRECTORIES

OUTPUT_DIR = "outputs"
MODEL_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# LOAD DATASET

print("\nLoading Breast Cancer Dataset...")

data = load_breast_cancer()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

# FEATURE SCALING

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler (REAL ML PIPELINE STEP)
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
joblib.dump(scaler, scaler_path)

print(f"Scaler saved at: {scaler_path}")

# BUILD MODEL (Sequential API)

print("\nBuilding Neural Network...")

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

# COMPILE MODEL

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# CALLBACKS

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

checkpoint_path = os.path.join(MODEL_DIR, "best_model.h5")

model_checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

# TRAIN MODEL

print("\nTraining Model...")

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, model_checkpoint],
    verbose=1
)

# EVALUATE MODEL

print("\nEvaluating Model...")

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# SAVE MODELS

h5_path = os.path.join(MODEL_DIR, "keras_model.h5")
savedmodel_path = os.path.join(MODEL_DIR, "SavedModel")

model.save(h5_path)
model.save(savedmodel_path)

print(f"\nModel saved (.h5): {h5_path}")
print(f"SavedModel directory: {savedmodel_path}")


# PLOT TRAINING HISTORY

print("\nSaving training plots...")

# Loss Curve
plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

loss_plot = os.path.join(OUTPUT_DIR, "loss_curve.png")
plt.savefig(loss_plot)
plt.close()

# Accuracy Curve
plt.figure()
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

acc_plot = os.path.join(OUTPUT_DIR, "accuracy_curve.png")
plt.savefig(acc_plot)
plt.close()

print(f"Plots saved in '{OUTPUT_DIR}' folder.")

# LOAD MODEL & TEST
print("\nLoading saved model for testing...")

loaded_model = load_model(h5_path)

# Predictions
pred_probs = loaded_model.predict(X_test)
pred_labels = (pred_probs > 0.5).astype(int)

# EVALUATION METRICS
print("\nClassification Report:")
print(classification_report(y_test, pred_labels))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred_labels))

print("\nSample Predictions:")
print(np.round(pred_probs[:5], 3))