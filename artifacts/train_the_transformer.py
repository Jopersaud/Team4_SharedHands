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
print(f"Classes: {encoder.classes}")

def transformer_block(x, num_heads=4, key_dim=16, ff_dim=128, dropout=0.1):
    attn_output = keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim
    )(x, x)
    attn_output = keras.layers.Dropout(dropout)(attn_output)
    x = keras.layers.layer






model = keras.Sequential([
    keras.layers.Input(shape=(21,3)), # expects 63 numbers in
    keras.layers.Dense(516, activation='relu'),  # 516 'neurons' which learns complex patterns
    keras.layers.Dropout(0.3),  # randomly drops 30% of neurons
    keras.layers.Dense(258, activation='relu'), # 258 'neurons' which learns complex patterns
    keras.layers.Dropout(0.3), # randomly drops 30% of neurons
    keras.layers.Dense(128, activation='relu'), # 128 ' neurons' refines the pattern
    keras.layers.Dropout(0.3), # drops 30% of neurons again
    keras.layers.Dense(64, activation='relu'), # narrowing it down to 64 'neurons'
    keras.layers.Dropout(0.2), # drops 20%
    keras.layers.Dense(32, activation='relu'), # 32 neurons, final layer
    keras.layers.Dense(26, activation='softmax') #output itself with 26 letters
    #soft max = converts raw numbers into probabilities that add up to 100%
    #example could be jeff the blank
    # a - Dog = 55 , b - cat = 33 ... so on and so forth
])

model.compile(
    optimizer="adam", #adjusts weights
    loss="categorical_crossentropy", # failures
    metrics=['accuracy'] # tracks accuracy
)

model.summary()
history = model.fit(
    X_train, y_train,
    epochs = 150, # passes the data 100 times
    batch_size = 32, # processing 32 samples at a time
    validation_data=(X_test, y_test), # checks for unseen data for each epoch
    verbose = 1
)
# it basically its like studying for an exam

model.save("asl_model.keras")
print("Complete model")

loss, accuracy = model.evaluate(X_test, y_test, verbose = 0)
print(f"Accuracy proven to be: {accuracy * 100:.2f}%")
print(f"Training samples: {len(X_train)}")
print(f"Training samples: {len(X_test)}")
print(f"Number of classes: {y_categorical.shape[1]}")


