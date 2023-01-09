import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

class RNN:

    def __init__(self, n_inputs):
        self.n_inputs = n_inputs
        self.weights_input = .1 * np.random.randn(n_inputs)
        self.weights_hidden_state = .1 * np.random.randn(n_inputs)
        self.bias = 0

    def initialize(self, n_iterations):
        self.instances = []

        for iteration in range(n_iterations):
            self.instances.append(RNN_instance(self))

    def denitialize(self):
        self.instances = []

    def forward(self, X, hidden_state = 0):
        self.initialize(len(X))

        self.outputs = []
        if not hidden_state:
            hidden_state = np.zeros_like(X[0])

        for input, instance in zip(X, self.instances):
            hidden_state = instance.forward(input, hidden_state)
            self.outputs.append(hidden_state)

        self.dvalues_weights_input = 0
        self.dvalues_weights_hidden_state = 0
        self.dvalues_bias = 0

        return self.outputs

    def backward(self, y, truncation_val = 100):
        truncation = 0

        for (index, instance), y_true in zip(enumerate(self.instances), y):
            dvalue_weights_input_update = 0
            dvalue_weights_hidden_state_update = np.copy(self.weights_hidden_state)
            dvalue_bias_update = 0

            if index >= truncation_val:
                truncation = index - truncation_val

            for n in range(truncation, index + 1):
                tanh_derivatives = self.tanh_derivative(self.instances[n].pretanh)

                dvalue_weights_input_update += np.array(self.instances[n].input)
                dvalue_weights_input_update *= tanh_derivatives * self.weights_hidden_state

                dvalue_weights_hidden_state_update *= tanh_derivatives
                dvalue_weights_hidden_state_update += self.instances[n].output * self.weights_hidden_state

                dvalue_bias_update += 1
                dvalue_bias_update *= tanh_derivatives * self.weights_hidden_state

            dvalues = instance.loss_derivative(y_true)

            self.dvalues_weights_input += dvalues * dvalue_weights_input_update / self.weights_hidden_state
            self.dvalues_weights_hidden_state += dvalues * (dvalue_weights_hidden_state_update - self.instances[index].output * self.weights_hidden_state)
            self.dvalues_bias += np.mean(dvalues * dvalue_bias_update / self.weights_hidden_state)

        self.dvalues_weights_input /= len(self.instances)
        self.dvalues_weights_hidden_state /= len(self.instances)
        self.dvalues_bias /= len(self.instances)

        self.denitialize()

    def optimize(self, learning_rate):
        self.weights_input -= self.dvalues_weights_input * learning_rate
        self.weights_hidden_state -= self.dvalues_weights_hidden_state * learning_rate
        self.bias -= self.dvalues_bias * learning_rate

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - self.tanh(x) ** 2

    def calc_total_loss(self, y):
        self.total_loss = 0

        for instance, y_true in zip(self.instances, y):
            self.total_loss += instance.calc_loss(y_true)

        return self.total_loss

    def train(self, X, y, X_val, y_val, epochs, hidden_state = 0, learning_rate = .1, print_every = 100, truncation_val = 100):
        for epoch in range(epochs):
            self.forward(X, hidden_state)
            if not epoch % print_every:
                print("Epoch:", epoch, "Loss:", self.calc_total_loss(y))
            self.backward(y, truncation_val)
            self.optimize(learning_rate)

        self.forward(X_val)
        print("Validation Loss:", self.calc_total_loss(y_val))

    def predict(self, X, hidden_state, pred_steps):
        model.denitialize()
        model.initialize(1)

        outputs = []

        for input in X:
            hidden_state = self.instances[0].forward(input, hidden_state)
            outputs.append(hidden_state)

        for step in range(pred_steps):
            hidden_state = self.instances[0].forward(hidden_state, hidden_state)
            outputs.append(hidden_state)
        
        return outputs


class RNN_instance(RNN):

    def __init__(self, model):
        self.weights_input = model.weights_input
        self.weights_hidden_state = model.weights_hidden_state
        self.bias = model.bias

    def forward(self, X, hidden_state):
        self.input = X
        self.hidden_state = hidden_state

        self.pretanh = X * self.weights_input + hidden_state * self.weights_hidden_state + self.bias
        self.output = self.tanh(self.pretanh)

        return self.output

    def calc_loss(self, y_true):
        self.loss = np.mean((y_true - self.output) ** 2)
        return self.loss

    def loss_derivative(self, y_true):
        return 2 * (self.output - y_true)

def create_stock_price_data(symbol, interval = "1d", period = "max", batches = 0):
    ticker = yf.Ticker(symbol)
    data = ticker.history(period = period, interval = interval)

    avg_prices = []

    for high, low, close in zip(data["High"], data["Low"], data["Close"]):
        avg_prices.append([(high + low + close) / 3])

    if batches:
        adj_prices = []
        batch_size = int(len(avg_prices) / batches)

        print(np.shape(avg_prices))

        for n in range(batches - 1):
            adj_prices = np.concatenate((adj_prices, np.reshape(avg_prices[batch_size * n:batch_size * (n+1)] / np.max(avg_prices[batch_size * n:batch_size * (n+1)]), newshape = -1)))

        adj_prices = np.concatenate((adj_prices, np.reshape(avg_prices[batch_size * (n+1):] / np.max(avg_prices[batch_size * (n+1):]), newshape = -1)))

        avg_prices = np.reshape(adj_prices, newshape = (-1, 1))

    return avg_prices

stock_data = create_stock_price_data("btc-usd", batches = 5)

X = stock_data[:-1]
y = stock_data[1:]

X_val = X[-500:]
y_val = y[-500:]

X = X[:-500]
y = y[:-500]

model = RNN(1)

outputs = model.forward(X)
print(model.weights_hidden_state, model.weights_input, model.bias)

plt.plot(range(len(X)), y)
plt.plot(range(len(X)), outputs)
plt.show()

model.train(X, y, X_val, y_val, 250, X[1], .01, 10, 10)

print(model.weights_hidden_state, model.weights_input, model.bias)

outputs = model.forward(X)

plt.plot(range(len(X)), y)
plt.plot(range(len(X)), outputs)
plt.show()

outputs = model.predict(X_val, X_val[1], 50)

plt.plot(range(len(X_val)), y_val)
plt.plot(range(len(outputs)), outputs)
plt.show()