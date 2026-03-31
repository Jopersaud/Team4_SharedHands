import numpy as np
import json
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

with open("asl_sequences.json", "r") as f:
    data = json.load(f)

X = np.array([item["sequence"] for item in data])
y = np.array([item["label"] for item in data])

print(f"Data shape: {X.shape}")
print(f"Signs: {sorted(set(y))}")

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = keras.utils.to_categorical(y_encoded)
num_classes = y_categorical.shape[1]

np.save("Transformer_classes.npy", encoder.classes_)
print(f"Classes: {encoder.classes_}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size = 0.2, random_state = 42
)

print(f"Training: {len(X_train)} | Testing: {len(X_test)}")

class TransformerBLock(keras.layers.Layer):
    def __init__(self, num_heads=4, key_dim=16, ff_dim=128, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attention = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.dropout1 = keras.layers.Dropout(dropout)
        self.dropout2 = keras.layers.Dropout(dropout)
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.ff1 = keras.layers.Dense(ff_dim, activation="relu")
        self.ff2 = None # fuck me

    def build(self, input_shape):
        self.ff2 = keras.layers.Dense(input_shape[-1])
        super().build(input_shape)

    def call(self, x, training=False):
        attn_output = self.attention(x, x)
        attn_output = self.dropout1(attn_output, training = training)
        x = self.norm1(x + attn_output)

        ff_output = self.ff1(x)
        ff_output = self.ff2(ff_output)
        ff_output = self.dropout2(ff_output, training=training)
        x = self.norm2(x + ff_output)


        return x
""""" def transformer_block(x, num_heads=4, key_dim=16, ff_dim=128, dropout=0.1):
    attn_output = keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim
    )(x, x)
    attn_output = keras.layers.Dropout(dropout)(attn_output)
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

    ff_output = keras.layers.Dense(ff_dim, activation="relu")(x)
    ff_output = keras.layers.Dense(x.shape[-1])(ff_output)
    ff_output = keras.layers.Dropout(dropout)(ff_output)
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x + ff_output)

    return x
"""
inputs = keras.Input(shape=(30,63))

x = keras.layers.Dense(64)(inputs)

""" # two transformers blocks stacked aso it can give more capacity to learn what motion patterns are
x = keras.layers.Lambda(lambda z: transformer_block(z))(x)
x = keras.layers.Lambda(lambda z: transformer_block(z))(x)
"""
x = TransformerBLock(num_heads=4, key_dim=16, ff_dim=128, dropout=0.1)(x)
x = TransformerBLock(num_heads=4, key_dim=16, ff_dim=128, dropout=0.1)(x)


x = keras.layers.GlobalAveragePooling1D()(x)

x = keras.layers.Dense(128, activation="relu")(x)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.Dense(64, activation="relu")(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer="adam", #adjusts weights
    loss="categorical_crossentropy", # failures
    metrics=['accuracy'] # tracks accuracy
)

model.summary()
history = model.fit(
    X_train, y_train,
    epochs = 80, # passes the data 100 times (smaller data set now chnaged to 80)
    batch_size = 16, # processing 32 samples at a time (changed to 16 due to bigger batches leading to poor learning)
    validation_data=(X_test, y_test), # checks for unseen data for each epoch
    verbose = 1
)
# it basically its like studying for an exam

model.save("asl_transformer.keras")
print("Complete transformer model saved")

loss, accuracy = model.evaluate(X_test, y_test, verbose = 0)
print(f"Accuracy proven to be {accuracy * 100:.2f}%")


