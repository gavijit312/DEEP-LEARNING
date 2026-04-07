import numpy as np
import matplotlib.pyplot as plt

# Step 1: Sample dataset
X = np.array([
    [2, 3],
    [1, 1],
    [4, 5],
    [6, 7],
    [2, 1],
    [7, 8]
])

y = np.array([0, 0, 0, 1, 0, 1])   # Binary labels

# Step 2: Initialize weights and bias
w = np.array([0.0, 0.0])
b = 0.0

# Step 3: Perceptron prediction function
def predict(x, w, b):
    z = np.dot(w, x) + b
    return 1 if z >= 0 else 0

# Step 4: Training using Perceptron Trick
epochs = 10

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}")

    for i in range(len(X)):
        x_i = X[i]
        y_true = y[i]
        y_pred = predict(x_i, w, b)

        # If prediction is wrong, apply perceptron trick
        if y_true == 1 and y_pred == 0:
            w = w + x_i
            b = b + 1
            print(f"Point {x_i} misclassified as 0, updating: w={w}, b={b}")

        elif y_true == 0 and y_pred == 1:
            w = w - x_i
            b = b - 1
            print(f"Point {x_i} misclassified as 1, updating: w={w}, b={b}")

# Step 5: Final weights and bias
print("\nFinal Weights:", w)
print("Final Bias:", b)

# Step 6: Test predictions
print("\nPredictions:")
for i in range(len(X)):
    print(f"Input: {X[i]}, True: {y[i]}, Predicted: {predict(X[i], w, b)}")

# Step 7: Visualize the decision boundary
plt.figure(figsize=(8, 6))

# Plot the data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=100, edgecolors='k')

# Create a meshgrid to plot the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predict for each point in the meshgrid
Z = np.array([predict(np.array([x_val, y_val]), w, b) for x_val, y_val in np.c_[xx.ravel(), yy.ravel()]])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
plt.title('Perceptron Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()