import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the handwritten digit dataset
digits = load_digits()

X = digits.images
y = digits.target

# Select only two classes: 0 and 1
mask = (y == 0) | (y == 1)

X = X[mask]
y = y[mask]

# Make the dataset small
class_0 = np.where(y == 0)[0][:20]
class_1 = np.where(y == 1)[0][:20]

indices = np.concatenate([class_0, class_1])

X = X[indices]
y = y[indices]

# Normalize pixel values
X = X / 16.0

# Convert 8x8 images into 64 features
X = X.reshape(X.shape[0], -1)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Initialize weights and bias
weights = np.zeros(X_train.shape[1])
bias = 0.0

learning_rate = 0.1
epochs = 20

# Training
for epoch in range(epochs):

    for x, target in zip(X_train, y_train):

        # Weighted sum
        z = np.dot(weights, x) + bias

        # Step activation function
        prediction = 1 if z >= 0 else 0

        # Calculate error
        error = target - prediction

        # Update weights and bias
        weights = weights + learning_rate * error * x
        bias = bias + learning_rate * error


# Training prediction
train_predictions = []

for x in X_train:

    z = np.dot(weights, x) + bias

    prediction = 1 if z >= 0 else 0

    train_predictions.append(prediction)


# Testing prediction
test_predictions = []

for x in X_test:

    z = np.dot(weights, x) + bias

    prediction = 1 if z >= 0 else 0

    test_predictions.append(prediction)


# Calculate accuracy
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print("====================================")
print(" SINGLE LAYER PERCEPTRON")
print("====================================")

print("Total samples:", len(X))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Image size: 8 x 8")
print("Number of features:", X_train.shape[1])

print("------------------------------------")

print("Training Accuracy:",
      train_accuracy * 100, "%")

print("Testing Accuracy:",
      test_accuracy * 100, "%")