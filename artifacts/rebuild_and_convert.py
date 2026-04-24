import os
import io
import zipfile
import numpy as np
import h5py
import tensorflow as tf
import tensorflowjs as tfjs

BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_IN  = os.path.join(BASE, "asl_model.keras")
MODEL_OUT = os.path.join(BASE, "..", "public", "asl_model")

# ── 1. Extract weights from the .keras zip ──────────────────────────────────
print("Extracting weights...")
with zipfile.ZipFile(MODEL_IN, "r") as z:
    h5bytes = z.read("model.weights.h5")

weights = {}
with h5py.File(io.BytesIO(h5bytes), "r") as f:
    for full_key in f.keys():
        if not full_key.startswith("_layer_checkpoint_dependencies\\"):
            continue
        layer_name = full_key.split("\\", 1)[1]
        group = f[full_key]["vars"]
        weights[layer_name] = [group[str(i)][()] for i in range(len(group))]

# ── 2. Rebuild model (same architecture as asl_model.keras) ─────────────────
print("Rebuilding model...")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(258, activation="relu", input_shape=(63,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64,  activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32,  activation="relu"),
    tf.keras.layers.Dense(26,  activation="softmax"),
], name="asl_model")

# ── 3. Load weights in order (h5 keys: dense, dense_2, dense_4, dense_6, dense_8)
ordered_keys = ["dense", "dense_2", "dense_4", "dense_6", "dense_8"]
all_weights = []
for key in ordered_keys:
    all_weights.extend(weights[key])

model.set_weights(all_weights)
model.summary()

# ── 4. Convert to TF.js ──────────────────────────────────────────────────────
print(f"\nConverting to TF.js → {MODEL_OUT}")
os.makedirs(MODEL_OUT, exist_ok=True)
tfjs.converters.save_keras_model(model, MODEL_OUT)

print("Done. Files written:")
for f in os.listdir(MODEL_OUT):
    print(f"  {f}")
