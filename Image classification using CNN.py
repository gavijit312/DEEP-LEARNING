import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

# 1. Load dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# 2. Normalize images
X_train = X_train / 255.0
X_test = X_test / 255.0

# 3. Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 4. Build CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)), # Corrected input_shape
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 5. Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Train model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# 7. Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)

# 8. Predict
predictions = model.predict(X_test)

# 9. Show sample prediction
index = 0
plt.imshow(X_test[index])
plt.title(f"Predicted: {class_names[np.argmax(predictions[index])]}, Actual: {class_names[y_test[index][0]]}")
plt.show()
