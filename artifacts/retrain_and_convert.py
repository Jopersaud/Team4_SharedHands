"""
Step 2+3: Retrain the ASL model on updated CSV, then convert to TF.js.
Run after add_dataset.py completes:
    python retrain_and_convert.py
"""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import io, zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflowjs as tfjs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE      = os.path.dirname(os.path.abspath(__file__))
CSV_PATH  = os.path.join(BASE, "asl_landmarks.csv")
MODEL_OUT = os.path.join(BASE, "..", "public", "asl_model")

# ── 1. Load CSV (skip git conflict markers) ──────────────────────────────────
print("Loading dataset...")
rows = []
with open(CSV_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("<") or line.startswith("=") or line.startswith(">"):
            continue
        parts = line.split(",")
        if len(parts) == 64:
            rows.append(parts)

X = np.array([[float(v) for v in r[:63]] for r in rows], dtype=np.float32)
y_raw = np.array([r[63].strip().upper() for r in rows])

# Keep only A-Z
mask = np.array([len(label) == 1 and label.isalpha() for label in y_raw])
X, y_raw = X[mask], y_raw[mask]

print(f"Total samples: {len(X)}")
print(f"Labels: {sorted(set(y_raw))}")

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_raw)
y_cat = tf.keras.utils.to_categorical(y_encoded)
print(f"Classes: {list(encoder.classes_)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Train: {len(X_train)}  Test: {len(X_test)}")

# ── 2. Build model ────────────────────────────────────────────────────────────
print("\nBuilding model...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(63,)),
    tf.keras.layers.Dense(258, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(26, activation="softmax"),
], name="asl_model")

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ── 3. Train ──────────────────────────────────────────────────────────────────
print("\nTraining...")
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=5, restore_best_weights=True
)
model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1,
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {acc * 100:.2f}%")

# ── 4. Save keras model ───────────────────────────────────────────────────────
model.save(os.path.join(BASE, "asl_model.keras"))
print("Saved asl_model.keras")

# ── 5. Convert to TF.js ──────────────────────────────────────────────────────
print(f"\nConverting to TF.js -> {MODEL_OUT}")
os.makedirs(MODEL_OUT, exist_ok=True)
tfjs.converters.save_keras_model(model, MODEL_OUT)

print("Done. Files written:")
for f in os.listdir(MODEL_OUT):
    print(f"  {f}")
