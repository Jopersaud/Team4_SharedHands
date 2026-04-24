import os, io, zipfile
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import h5py
import tensorflow as tf
import tensorflowjs as tfjs

BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_IN  = os.path.join(BASE, "asl_transformer.keras")
MODEL_OUT = os.path.join(BASE, "..", "public", "asl_transformer_model")

# ── 1. Extract weights ───────────────────────────────────────────────────────
print("Extracting weights...")
with zipfile.ZipFile(MODEL_IN, "r") as z:
    h5bytes = z.read("model.weights.h5")

weights = {}
with h5py.File(io.BytesIO(h5bytes), "r") as f:
    for full_key in f.keys():
        if not full_key.startswith("layers\\"):
            continue
        layer_name = full_key.split("\\", 1)[1]
        group = f[full_key]
        # Recurse to collect all leaf datasets under this layer
        leaf = {}
        def collect(name, obj):
            if isinstance(obj, h5py.Dataset):
                leaf[name] = obj[()]
        group.visititems(collect)
        weights[layer_name] = leaf

# ── 2. Helper to get ordered vars from a layer group ────────────────────────
def get_vars(layer_name):
    grp = weights.get(layer_name, {})
    var_keys = sorted([k for k in grp if k.startswith("vars/")],
                      key=lambda k: int(k.split("/")[1]))
    return [grp[k] for k in var_keys]

# ── 3. Rebuild TransformerBlock as a standard Keras layer ───────────────────
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads=4, key_dim=16, ff_dim=128, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attention  = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.dropout1   = tf.keras.layers.Dropout(dropout)
        self.dropout2   = tf.keras.layers.Dropout(dropout)
        self.norm1      = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2      = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ff1        = tf.keras.layers.Dense(ff_dim, activation="relu")
        self.ff2        = None

    def build(self, input_shape):
        self.ff2 = tf.keras.layers.Dense(input_shape[-1])
        super().build(input_shape)

    def call(self, x, training=False):
        attn = self.attention(x, x)
        attn = self.dropout1(attn, training=training)
        x    = self.norm1(x + attn)
        ff   = self.ff1(x)
        ff   = self.ff2(ff)
        ff   = self.dropout2(ff, training=training)
        x    = self.norm2(x + ff)
        return x

# ── 4. Build the functional model ───────────────────────────────────────────
print("Rebuilding model...")
inputs = tf.keras.Input(shape=(30, 63))
x = tf.keras.layers.Dense(64)(inputs)                          # embedding
x = TransformerBlock(name="tb1")(x)
x = TransformerBlock(name="tb2")(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(27, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs, name="asl_transformer")

# Warm up so all sub-layers are built
model(tf.zeros((1, 30, 63)))
model.summary()

# ── 5. Load weights layer by layer ──────────────────────────────────────────
print("Loading weights...")

def set_mha_weights(mha_layer, prefix):
    """Set Q/K/V/O dense weights for a MultiHeadAttention layer."""
    # keras MHA internals: _query_dense, _key_dense, _value_dense, _output_dense
    q = weights[prefix + r"\attention\_query_dense"]["vars/0"], weights[prefix + r"\attention\_query_dense"]["vars/1"]
    k = weights[prefix + r"\attention\_key_dense"]["vars/0"],   weights[prefix + r"\attention\_key_dense"]["vars/1"]
    v = weights[prefix + r"\attention\_value_dense"]["vars/0"], weights[prefix + r"\attention\_value_dense"]["vars/1"]
    o = weights[prefix + r"\attention\_output_dense"]["vars/0"],weights[prefix + r"\attention\_output_dense"]["vars/1"]
    mha_layer._query_dense.set_weights([q[0], q[1]])
    mha_layer._key_dense.set_weights([k[0], k[1]])
    mha_layer._value_dense.set_weights([v[0], v[1]])
    mha_layer._output_dense.set_weights([o[0], o[1]])

def set_tb_weights(tb_layer, prefix):
    set_mha_weights(tb_layer.attention, prefix)
    tb_layer.ff1.set_weights(get_vars(prefix + r"\ff1"))
    tb_layer.ff2.set_weights(get_vars(prefix + r"\ff2"))
    tb_layer.norm1.set_weights(get_vars(prefix + r"\norm1"))
    tb_layer.norm2.set_weights(get_vars(prefix + r"\norm2"))

# Embedding dense
model.layers[1].set_weights(get_vars("dense"))

# TransformerBlocks (model.layers[2] and [3])
set_tb_weights(model.layers[2], "transformer_b_lock")
set_tb_weights(model.layers[3], "transformer_b_lock_1")

# GlobalAveragePooling has no weights (layer 4)
# Post-pooling dense layers (h5 key names from original saved model)
model.layers[5].set_weights(get_vars("dense_1"))   # Dense(128)
# layer 6 = Dropout
model.layers[7].set_weights(get_vars("dense_2"))   # Dense(64)
# layer 8 = Dropout
model.layers[9].set_weights(get_vars("dense_3"))   # Dense(27)

print("Weights loaded. Running sanity check...")
dummy = np.random.rand(1, 30, 63).astype(np.float32)
out = model.predict(dummy, verbose=0)
print(f"  Output shape: {out.shape}, sum: {out.sum():.4f} (should be ~1.0)")

# ── 6. Convert to TF.js via SavedModel (avoids custom layer serialization) ──
import tempfile
print(f"\nConverting to TF.js -> {MODEL_OUT}")
os.makedirs(MODEL_OUT, exist_ok=True)

with tempfile.TemporaryDirectory() as tmp:
    saved_model_path = os.path.join(tmp, "saved_model")
    tf.saved_model.save(model, saved_model_path)
    tfjs.converters.convert_tf_saved_model(saved_model_path, MODEL_OUT)

print("Done. Files written:")
for f in os.listdir(MODEL_OUT):
    print(f"  {f}")
