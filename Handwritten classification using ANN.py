import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# 1. Load MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# 2. Check shape
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# 3. Normalize pixel values (0 to 255 --> 0 to 1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# 4. Flatten images (28x28 --> 784)
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# 5. Build ANN model
model = keras.Sequential([
    keras.Input(shape=(784,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# 6. Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 7. Train model
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# 8. Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)

# 9. Predict
predictions = model.predict(X_test)

# 10. Show prediction for first image
print("Predicted digit:", np.argmax(predictions[0]))
print("Actual digit:", y_test[0])

# 11. Display image
plt.imshow(X_test[0].reshape(28,28), cmap='gray')
plt.title(f"Predicted: {np.argmax(predictions[0])}, Actual: {y_test[0]}")
plt.show()
