import tensorflow as tf
from tensorflow import keras
import numpy as np


model = keras.Sequential([
    keras.layers.Dense(4, activation='relu', input_shape=(2,)),  # 2 inputs -> 4 hidden neurons
    keras.layers.Dense(1, activation='sigmoid')                   # 4 hidden -> 1 output
])

# 2. Data (XOR: 0^0=0, 0^1=1, 1^0=1, 1^1=0)
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y = np.array([[0],[1],[1],[0]], dtype=np.float32)


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.1),
    loss='binary_crossentropy'
)

model.fit(X, y, epochs=1000, verbose=0)

print(model.predict(X).round())  # should print [[0],[1],[1],[0]]