import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

training_inputs = np.array([[1, 0, 1, 1],
                            [1, 1, 1, 1],
                            [0, 0, 0, 0],
                            [0, 1, 0, 0]])

training_outputs = np.array([[0, 1, 0, 1]]).T

weights = 2 * np.random.random((4, 1)) - 1

for interation in range(100000):

    input_layer = training_inputs

    outputs = sigmoid(np.dot(input_layer, weights))

    error = training_outputs - outputs

    adjustments = error * sigmoid_derivative(outputs)

    weights += np.dot(input_layer.T, adjustments)

print("training outputs:")
print(outputs)

testing_inputs = [[0, 0, 0, 1],
                  [1, 0, 1, 0],
                  [1, 1, 0, 1],
                  [0, 1, 1, 0]]

predictions = sigmoid(np.dot(testing_inputs, weights))

print("testing outputs")
print(predictions)