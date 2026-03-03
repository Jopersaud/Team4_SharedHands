import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras

df = pd.read_csv("asl_landmarks.csv", header=None)

print(f"Dataset Shape: {df.shape}")
print(f"Letters found: {sorted(df[63].unique())}")

X = df.iloc[:, :63].values
y = df.iloc[:, 63].values

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = keras.utils.to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

model = keras.Sequential([
    keras.layers.Input(shape=(63,)),
    keras.layers.Dense(258, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(26, activation='softmax')
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=['accuracy']
)

model.summary()
history = model.fit(
    X_train, y_train,
    epochs = 100,
    batch_size = 32,
    validation_data=(X_test, y_test),
    verbose = 1
)

model.save("asl_model.keras")
print("Complete model")

loss, accuracy = model.evaluate(X_test, y_test, verbose = 0)
print(f"Accuracy proven to be: {accuracy * 100:.2f}%")
print(f"Training samples: {len(X_train)}")
print(f"Training samples: {len(X_test)}")
print(f"Number of classes: {y_categorical.shape[1]}")


