import numpy as np
import matplotlib.pyplot as plt

class LSTM:

    def __init__(self, n_inputs):
        self.n_inputs = n_inputs

        self.weights_input_X = .1 * np.random.randn(n_inputs)
        self.weights_input_y = .1 * np.random.randn(n_inputs)
        self.bias_input = 0

        self.weights_input_gate_X = .1 * np.random.randn(n_inputs)
        self.weights_input_gate_y = .1 * np.random.randn(n_inputs)
        self.bias_input_gate = 0

        self.weights_forget_gate_X = .1 * np.random.randn(n_inputs)
        self.weights_forget_gate_y = .1 * np.random.randn(n_inputs)
        self.bias_forget_gate = 0

        self.weights_output_gate_X = .1 * np.random.randn(n_inputs)
        self.weights_output_gate_y = .1 * np.random.randn(n_inputs)
        self.bias_output_gate = 0

        self.cell_state = np.zeros(n_inputs)

        self.dvalues_weights_output_gate_X = np.zeros(n_inputs)
        self.dvalues_weights_output_gate_y = np.zeros(n_inputs)
        self.dvalues_bias_output_gate = np.zeros(n_inputs)
        self.dvalues_weights_forget_gate_X = np.zeros(n_inputs)
        self.dvalues_weights_forget_gate_y = np.zeros(n_inputs)
        self.dvalues_bias_forget_gate = np.zeros(n_inputs)
        self.dvalues_weights_input_gate_X = np.zeros(n_inputs)
        self.dvalues_weights_input_gate_y = np.zeros(n_inputs)
        self.dvalues_bias_input_gate = np.zeros(n_inputs)
        self.dvalues_weights_input_X = np.zeros(n_inputs)
        self.dvalues_weights_input_y = np.zeros(n_inputs)
        self.dvalues_bias_input = np.zeros(n_inputs)

        self.instances = []

    def tanh(self, inputs):
        return np.tanh(inputs)

    def tanh_derivative(self, inputs):
        return 1 - np.tanh(inputs) ** 2
    
    def sigmoid(self, inputs):
        return 1 / (1 + np.exp(-inputs))

    def sigmoid_derivative(self, inputs):
        return self.sigmoid(inputs) * (1 - self.sigmoid(inputs))

    def backward(self, y):
        for instance, y_true in zip(self.instances, y):
            dvalues_loss = self.loss_derivative(instance.output, y_true)

            self.dvalues_weights_output_gate_X += dvalues_loss * self.tanh(instance.cell_state) * self.sigmoid_derivative(instance.network_output) * instance.input
            self.dvalues_weights_output_gate_y += dvalues_loss * self.tanh(instance.cell_state) * self.sigmoid_derivative(instance.network_output) * instance.last_output
            self.dvalues_bias_output_gate += dvalues_loss * self.tanh(instance.cell_state) * self.sigmoid_derivative(instance.network_output)

            dvalues_cell_state = dvalues_loss * instance.gate_output * self.tanh_derivative(instance.cell_state)

            self.dvalues_weights_forget_gate_X += dvalues_cell_state * instance.last_cell_state * self.sigmoid_derivative(instance.network_forget) * instance.input
            self.dvalues_weights_forget_gate_y += dvalues_cell_state * instance.last_cell_state * self.sigmoid_derivative(instance.network_forget) * instance.last_output
            self.dvalues_bias_forget_gate += dvalues_cell_state * instance.last_cell_state * self.sigmoid_derivative(instance.network_forget)

            print(dvalues_cell_state, instance.hidden_state, self.sigmoid_derivative(instance.network_input), instance.input)

            self.dvalues_weights_input_gate_X += dvalues_cell_state * instance.hidden_state * self.sigmoid_derivative(instance.network_input) * instance.input
            self.dvalues_weights_input_gate_y += dvalues_cell_state * instance.hidden_state * self.sigmoid_derivative(instance.network_input) * instance.last_output
            self.dvalues_bias_input_gate += dvalues_cell_state * instance.hidden_state * self.sigmoid_derivative(instance.network_input)

            self.dvalues_weights_input_X += dvalues_cell_state * instance.gate_input * self.tanh_derivative(instance.network_hidden) * instance.input
            self.dvalues_weights_input_y += dvalues_cell_state * instance.gate_input * self.tanh_derivative(instance.network_hidden) * instance.last_output
            self.dvalues_bias_input += dvalues_cell_state * instance.gate_input * self.tanh_derivative(instance.network_hidden)

            self.clear_instances()

    def optimize(self, learning_rate):
        self.weights_input_X -= learning_rate * self.dvalues_weights_output_gate_X
        self.weights_input_y -= learning_rate * self.dvalues_weights_output_gate_y
        self.bias_input -= learning_rate * self.dvalues_bias_output_gate

        print(self.dvalues_weights_forget_gate_X, self.dvalues_weights_input_gate_X)
        self.weights_input_gate_X -= learning_rate * self.dvalues_weights_input_gate_X
        self.weights_input_gate_y -= learning_rate * self.dvalues_weights_input_gate_y
        self.bias_input_gate -= learning_rate * self.dvalues_bias_input_gate

        self.weights_forget_gate_X -= learning_rate * self.dvalues_weights_forget_gate_X
        self.weights_forget_gate_y -= learning_rate * self.dvalues_weights_forget_gate_y
        self.bias_forget_gate -= learning_rate * self.dvalues_bias_forget_gate

        self.weights_output_gate_X -= learning_rate * self.dvalues_weights_input_X
        self.weights_output_gate_y -= learning_rate * self.dvalues_weights_input_y
        self.bias_output_gate -= learning_rate * self.dvalues_bias_input

    def forward(self, X, y):
        if len(self.instances) < len(X):
            for step in X:
                self.instances.append(LSTM_instance(self, self.n_inputs))

        self.outputs = []

        if len(y) == 1:
            y = [y]

        for input, instance, prev_y in zip(X, self.instances, y):
            output = instance.forward(input, prev_y)
            if len(y) < len(X):
                y.append(output)
            self.outputs.append(output)

        return self.outputs
    
    def calc_loss(self, y_true):
        self.total_loss = 0

        for instance, true in zip(self.instances, y_true):
            self.total_loss += instance.calc_loss(true)

        return self.total_loss

    def loss_derivative(self, y_pred, y_true):
        self.dvalues_loss = -2 * (y_true - y_pred)
        return self.dvalues_loss

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            print(epoch)
            outputs = self.forward(X, y)
            self.calc_loss(outputs, y)
            self.backward(y)
            print(self.loss, self.dvalues_weights_output_gate_X)
            self.optimize(learning_rate)

        output = self.forward(X, np.zeros_like(X))
        print(output)
        self.calc_loss(output, y)
        print(self.loss)

        self.clear_instances()

    def clear_instances(self):
        self.instances = []

class LSTM_instance(LSTM):

    def __init__(self, model, n_inputs):
        super().__init__(n_inputs)

        self.weights_input_X = model.weights_input_X
        self.weights_input_y = model.weights_input_y
        self.bias_input = model.bias_input

        self.weights_input_gate_X = model.weights_input_gate_X
        self.weights_input_gate_y = model.weights_input_gate_y
        self.bias_input_gate = model.bias_input_gate
        
        self.weights_forget_gate_X = model.weights_forget_gate_X
        self.weights_forget_gate_y = model.weights_forget_gate_y
        self.bias_forget_gate = model.bias_forget_gate

        self.weights_output_gate_X = model.weights_output_gate_X
        self.weights_output_gate_y = model.weights_output_gate_y
        self.bias_output_gate = model.bias_output_gate

    def forward(self, X, last_y):
        self.input = X
        self.last_output = last_y

        self.network_input = self.input * self.weights_input_gate_X + self.last_output * self.weights_input_gate_y + self.bias_input_gate
        self.gate_input = self.sigmoid(self.network_input)

        self.network_forget = self.input * self.weights_forget_gate_X + self.last_output * self.weights_forget_gate_y + self.bias_forget_gate
        self.gate_forget = self.sigmoid(self.network_forget)

        self.network_output = self.input * self.weights_output_gate_X + self.last_output * self.weights_output_gate_y + self.bias_output_gate
        self.gate_output = self.sigmoid(self.network_output)

        self.network_hidden = self.input * self.weights_input_X + self.last_output * self.weights_input_y + self.bias_input
        self.hidden_state = self.tanh(self.network_hidden)
        self.last_cell_state = self.cell_state
        self.cell_state = self.last_cell_state * self.gate_forget + self.hidden_state * self.gate_input
        self.output = self.gate_output * self.tanh(self.cell_state)

        self.dvalues_weights_output_gate_X *= 0
        self.dvalues_weights_output_gate_y *= 0
        self.dvalues_bias_output_gate *= 0

        self.dvalues_weights_forget_gate_X *= 0
        self.dvalues_weights_forget_gate_y *= 0
        self.dvalues_bias_forget_gate *= 0

        self.dvalues_weights_input_gate_X *= 0
        self.dvalues_weights_input_gate_y *= 0
        self.dvalues_bias_input_gate *= 0

        self.dvalues_weights_input_X *= 0
        self.dvalues_weights_input_y *= 0
        self.dvalues_bias_input *= 0

        return self.output

    def calc_loss(self, y_true):
        self.loss = np.mean((y_true - self.output) ** 2)
        return self.loss
            

X = []
for x in range(100):
    X.append([x/100])
y = []
for y_val in range(100):
    y.append([(y_val+1)/100])

lstm = LSTM(1)

output = lstm.forward(X, [0])

plt.plot(range(100), X)
plt.plot(range(100), output)
plt.show()

for i in range(25):
    lstm.calc_loss(output, y)
    lstm.backward(y)
    lstm.optimize(.1)

    output = lstm.forward(X, [0])
    print(lstm.weights_forget_gate_X, lstm.instances[0].weights_forget_gate_X)

print(len(lstm.instances))


plt.plot(range(100), X)
plt.plot(range(100), output)
plt.show()