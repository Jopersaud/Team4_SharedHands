"""
One-time script to convert asl_model.keras to TensorFlow.js format.

Usage:
    pip install tensorflowjs
    cd artifacts
    python convert_model.py

Output: ../public/asl_model/model.json + weight shard(s)
The CRA dev server serves public/ at the root, so the model loads from
http://localhost:3000/asl_model/model.json in the browser.
"""
import os
import tensorflowjs as tfjs
import tf_keras as keras

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_IN  = os.path.join(BASE, "asl_model.keras")
MODEL_OUT = os.path.join(BASE, "..", "public", "asl_model")

print(f"Loading model from: {MODEL_IN}")
model = keras.models.load_model(MODEL_IN, compile=False)
model.summary()

print(f"\nConverting to TF.js format → {MODEL_OUT}")
tfjs.converters.save_keras_model(model, MODEL_OUT)
print("Done. Files written:")
for f in os.listdir(MODEL_OUT):
    print(f"  {f}")
