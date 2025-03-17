# Class Sample Visulaization(ques-1)

import numpy as np
import wandb
from tensorflow.keras.datasets import fashion_mnist

# Initialize wandb
wandb.init(project="fashion-mnist-dataset", name="class_samples")

# Load Fashion-MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Class labels
class_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Log one sample image for each class
images = []
for i in range(10):
    idx = np.where(y_train == i)[0][0]
    image = x_train[idx]
    images.append(wandb.Image(image, caption=class_labels[i]))

# Log images to wandb
wandb.log({"fashion-mnist-visualization": images})

# Finish wandb run
wandb.finish()








# FEED FORWARD NEURAL NETWORK (ques-2)
import argparse
import numpy as np
import wandb
from tensorflow.keras.datasets import fashion_mnist

# Activation Functions and Derivatives
def identity(x): return x
def identity_derivative(x): return np.ones_like(x)

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x): return sigmoid(x) * (1 - sigmoid(x))

def tanh(x): return np.tanh(x)
def tanh_derivative(x): return 1 - np.tanh(x)**2

def relu(x): return np.maximum(0, x)
def relu_derivative(x): return (x > 0).astype(float)

activation_functions = {
    "identity": (identity, identity_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative),
    "ReLU": (relu, relu_derivative)
}

# Loss Functions
def mse(y_true, y_pred): return np.mean((y_true - y_pred) ** 2)
def mse_derivative(y_true, y_pred): return 2 * (y_pred - y_true) / y_true.size

def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-8)) / y_true.shape[0]

def cross_entropy_derivative(y_true, y_pred):
    return -y_true / (y_pred + 1e-8)

loss_functions = {
    "mean_squared_error": (mse, mse_derivative),
    "cross_entropy": (cross_entropy, cross_entropy_derivative)
}

# Neural Network Class
class FeedForwardNN:
    def __init__(self, input_size, output_size, hidden_layers, hidden_size, activation, weight_init):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.activation_func, self.activation_deriv = activation_functions[activation]

        # Weight Initialization
        if weight_init == "random":
            self.weights = [np.random.randn(self.input_size, self.hidden_size) * 0.01]
            self.biases = [np.zeros((1, self.hidden_size))]
            for _ in range(hidden_layers - 1):
                self.weights.append(np.random.randn(self.hidden_size, self.hidden_size) * 0.01)
                self.biases.append(np.zeros((1, self.hidden_size)))
            self.weights.append(np.random.randn(self.hidden_size, self.output_size) * 0.01)
            self.biases.append(np.zeros((1, self.output_size)))
        elif weight_init == "Xavier":
            self.weights = [np.random.randn(self.input_size, self.hidden_size) * np.sqrt(1 / self.input_size)]
            self.biases = [np.zeros((1, self.hidden_size))]
            for _ in range(hidden_layers - 1):
                self.weights.append(np.random.randn(self.hidden_size, self.hidden_size) * np.sqrt(1 / self.hidden_size))
                self.biases.append(np.zeros((1, self.hidden_size)))
            self.weights.append(np.random.randn(self.hidden_size, self.output_size) * np.sqrt(1 / self.hidden_size))
            self.biases.append(np.zeros((1, self.output_size)))

    def forward(self, X):
        self.a = [X]
        self.z = []
        for i in range(self.hidden_layers):
            self.z.append(self.a[-1] @ self.weights[i] + self.biases[i])
            self.a.append(self.activation_func(self.z[-1]))
        self.z.append(self.a[-1] @ self.weights[-1] + self.biases[-1])
        self.a.append(self.softmax(self.z[-1]))  # Output Layer (Softmax)
        return self.a[-1]

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def backward(self, X, y, learning_rate, loss_derivative):
        m = X.shape[0]
        dZ = loss_derivative(y, self.a[-1])
        dW = self.a[-2].T @ dZ / m
        dB = np.sum(dZ, axis=0, keepdims=True) / m

        self.weights[-1] -= learning_rate * dW
        self.biases[-1] -= learning_rate * dB

        for i in range(self.hidden_layers - 1, -1, -1):
            dZ = dZ @ self.weights[i + 1].T * self.activation_deriv(self.z[i])
            dW = self.a[i].T @ dZ / m
            dB = np.sum(dZ, axis=0, keepdims=True) / m
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * dB

# Training Function
def train(args):
    wandb.init(project=args.wandb_project, entity=args.wandb_entity)

    # Load dataset
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0
    X_train, X_test = X_train.reshape(-1, 28 * 28), X_test.reshape(-1, 28 * 28)

    y_train_one_hot = np.eye(10)[y_train]

    # Initialize Model
    model = FeedForwardNN(input_size=784, output_size=10,
                          hidden_layers=args.num_layers, hidden_size=args.hidden_size,
                          activation=args.activation, weight_init=args.weight_init)

    loss_fn, loss_deriv = loss_functions[args.loss]

# Training Loop
    for epoch in range(args.epochs):
        for i in range(0, len(X_train), args.batch_size):
            X_batch = X_train[i:i + args.batch_size]
            y_batch = y_train_one_hot[i:i + args.batch_size]

            y_pred = model.forward(X_batch)
            model.backward(X_batch, y_batch, args.learning_rate, loss_deriv)

        y_train_pred = model.forward(X_train)
        train_loss = loss_fn(y_train_one_hot, y_train_pred)

        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {train_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "loss": train_loss})
# Argument Parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-wp", "--wandb_project", type=str, default="fashion-mnist-dataset)")
    parser.add_argument("-we", "--wandb_entity", type=str, default="ma23c014-indian-institute-of-technology-madras")
    parser.add_argument("-e", "--epochs", type=int, default=1)
    parser.add_argument("-b", "--batch_size", type=int, default=4)
    parser.add_argument("-l", "--loss", type=str, choices=["mean_squared_error", "cross_entropy"], default="cross_entropy")
    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd"], default="sgd")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.1)
    parser.add_argument("-nhl", "--num_layers", type=int, default=1)
    parser.add_argument("-sz", "--hidden_size", type=int, default=4)
    parser.add_argument("-a", "--activation", type=str, choices=["identity", "sigmoid", "tanh", "ReLU"], default="sigmoid")
    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "Xavier"], default="random")

    args, unknown = parser.parse_known_args()
    train(args)








# BACKPROPAGTAION (QUES-3)
import numpy as np

class NeuralNetwork:
    def __init__(self, layers, activation='relu', loss='mse'):
        self.layers = layers
        self.activation = activation
        self.loss_function = loss
        self.weights = []
        self.biases = []
        self.initialise_parameters()
        self.optimizers = {}

    def initialise_parameters(self):
        for i in range(len(self.layers) - 1):
            self.weights.append(np.random.randn(self.layers[i], self.layers[i + 1]) * 0.01)
            self.biases.append(np.zeros((1, self.layers[i + 1])))

    def activation_function(self, x, derivative=False):
        if self.activation == 'relu':
            if derivative:
                return np.where(x > 0, 1, 0)
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            sigmoid = 1 / (1 + np.exp(-x))
            if derivative:
                return sigmoid * (1 - sigmoid)
            return sigmoid

    def loss(self, y_true, y_pred):
        if self.loss_function == 'mse':
            return np.mean((y_true - y_pred)**2)
        else:
            raise ValueError("Unsupported loss function")

    def loss_derivative(self, y_true, y_pred):
        if self.loss_function == 'mse':
            return -(y_true - y_pred)

    def register_optimizer(self, name, optimizer):
        self.optimizers[name] = optimizer

    def forward(self, x):
        activations = [x]
        for i in range(len(self.weights)):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            a = self.activation_function(z)
            activations.append(a)
        return activations

    def backward(self, activations, y_true):
        deltas = []
        grads_w = []
        grads_b = []
        delta = self.loss_derivative(y_true, activations[-1]) * self.activation_function(activations[-1], derivative=True)
        deltas.append(delta)
        for i in range(len(self.weights) - 1, 0, -1):
            delta = (deltas[-1] @ self.weights[i].T) * self.activation_function(activations[i], derivative=True)
            deltas.append(delta)
        deltas.reverse()
        for i in range(len(self.weights)):
            grad_w = activations[i].T @ deltas[i]
            grad_b = np.sum(deltas[i], axis=0, keepdims=True)
            grads_w.append(grad_w)
            grads_b.append(grad_b)
        return grads_w, grads_b

    def update_parameters(self, grads_w, grads_b, optimizer_name, batch_size):
        optimizer = self.optimizers[optimizer_name]
        for i in range(len(self.weights)):
            self.weights[i], self.biases[i] = optimizer.update(
                i, self.weights[i], self.biases[i], grads_w[i], grads_b[i], batch_size
            )

    def fit(self, x, y, epochs, batch_size, optimizer_name):
        for epoch in range(epochs):
            for start in range(0, len(x), batch_size):
                end = start + batch_size
                x_batch, y_batch = x[start:end], y[start:end]
                activations = self.forward(x_batch)
                grads_w, grads_b = self.backward(activations, y_batch)
                self.update_parameters(grads_w, grads_b, optimizer_name, batch_size)


class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, index, w, b, grad_w, grad_b, batch_size):
        w -= self.learning_rate * grad_w / batch_size
        b -= self.learning_rate * grad_b / batch_size
        return w, b


class Momentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v_w = {}
        self.v_b = {}

    def update(self, index, w, b, grad_w, grad_b, batch_size):
        if index not in self.v_w:
            self.v_w[index] = np.zeros_like(w)
            self.v_b[index] = np.zeros_like(b)
        self.v_w[index] = self.momentum * self.v_w[index] - self.learning_rate * grad_w / batch_size
        self.v_b[index] = self.momentum * self.v_b[index] - self.learning_rate * grad_b / batch_size
        w += self.v_w[index]
        b += self.v_b[index]
        return w, b


class Nesterov:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v_w = {}
        self.v_b = {}

    def update(self, index, w, b, grad_w, grad_b, batch_size):
        if index not in self.v_w:
            self.v_w[index] = np.zeros_like(w)
            self.v_b[index] = np.zeros_like(b)
        v_prev_w = self.v_w[index]
        v_prev_b = self.v_b[index]
        self.v_w[index] = self.momentum * self.v_w[index] - self.learning_rate * grad_w / batch_size
        self.v_b[index] = self.momentum * self.v_b[index] - self.learning_rate * grad_b / batch_size
        w += -self.momentum * v_prev_w + (1 + self.momentum) * self.v_w[index]
        b += -self.momentum * v_prev_b + (1 + self.momentum) * self.v_b[index]
        return w, b

class RMSprop:
    def __init__(self, learning_rate=0.01, beta=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.s_w = {}
        self.s_b = {}
    def update(self, index, w, b, grad_w, grad_b, batch_size):
        if index not in self.s_w:
            self.s_w[index] = np.zeros_like(w)
            self.s_b[index] = np.zeros_like(b)
        self.s_w[index] = self.beta * self.s_w[index] + (1 - self.beta) * (grad_w / batch_size)**2
        self.s_b[index] = self.beta * self.s_b[index] + (1 - self.beta) * (grad_b / batch_size)**2
        w -= self.learning_rate * grad_w / (np.sqrt(self.s_w[index]) + self.epsilon)
        b -= self.learning_rate * grad_b / (np.sqrt(self.s_b[index]) + self.epsilon)
        return w, b
        
class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = {}
        self.v_w = {}
        self.m_b = {}
        self.v_b = {}
        self.t = 0
    def update(self, index, w, b, grad_w, grad_b, batch_size):
        if index not in self.m_w:
            self.m_w[index] = np.zeros_like(w)
            self.v_w[index] = np.zeros_like(w)
            self.m_b[index] = np.zeros_like(b)
            self.v_b[index] = np.zeros_like(b)
        self.t += 1
        self.m_w[index] = self.beta1 * self.m_w[index] + (1 - self.beta1) * grad_w / batch_size
        self.v_w[index] = self.beta2 * self.v_w[index] + (1 - self.beta2) * (grad_w / batch_size)**2
        self.m_b[index] = self.beta1 * self.m_b[index] + (1 - self.beta1) * grad_b / batch_size
        self.v_b[index] = self.beta2 * self.v_b[index] + (1 - self.beta2) * (grad_b / batch_size)**2
        m_w_corr = self.m_w[index] / (1 - self.beta1**self.t)
        v_w_corr = self.v_w[index] / (1 - self.beta2**self.t)
        m_b_corr = self.m_b[index] / (1 - self.beta1**self.t)
        v_b_corr = self.v_b[index] / (1 - self.beta2**self.t)
        w -= self.learning_rate * m_w_corr / (np.sqrt(v_w_corr) + self.epsilon)
        b -= self.learning_rate * m_b_corr / (np.sqrt(v_b_corr) + self.epsilon)
        return w, b






# HYPERPARAMETER TUNING (ques-4)
import numpy as np
import wandb
# Optimizer Class 
class Optimizer:
    def __init__(self, method="sgd", learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.method = method
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.velocity = {}
        self.cache = {}
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, weights, grads, key):
        if self.method == "sgd":
            return weights - self.lr * grads

        elif self.method == "momentum":
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(weights)
            self.velocity[key] = self.beta1 * self.velocity[key] - self.lr * grads
            return weights + self.velocity[key]

        elif self.method == "nag":
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(weights)
            lookahead = weights + self.beta1 * self.velocity[key]
            self.velocity[key] = self.beta1 * self.velocity[key] - self.lr * grads
            return lookahead + self.velocity[key]

        elif self.method == "rmsprop":
            if key not in self.cache:
                self.cache[key] = np.zeros_like(weights)
            self.cache[key] = self.beta1 * self.cache[key] + (1 - self.beta1) * (grads ** 2)
            return weights - self.lr * grads / (np.sqrt(self.cache[key]) + self.epsilon)

        elif self.method == "adam":
            self.t += 1
            if key not in self.m:
                self.m[key] = np.zeros_like(weights)
                self.v[key] = np.zeros_like(weights)
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads ** 2)
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            return weights - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

def load_data():
    from keras.datasets import fashion_mnist  
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0  # Normalize
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    # Split 10% of training data for validation
    val_size = int(0.1 * X_train.shape[0])
    X_val, y_val = X_train[:val_size], y_train[:val_size]
    X_train, y_train = X_train[val_size:], y_train[val_size:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test 

def initialize_weights(layers, init_type):
    weights = {}
    for i in range(len(layers) - 1):
        if init_type == "xavier":
            weights[f'W{i+1}'] = np.random.randn(layers[i], layers[i+1]) / np.sqrt(layers[i])
        elif init_type == "he":
            weights[f'W{i+1}'] = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2 / layers[i])
        else:
            weights[f'W{i+1}'] = np.random.randn(layers[i], layers[i+1]) * 0.1  # Increase from 0.01
        weights[f'b{i+1}'] = np.zeros((1, layers[i+1]))
    return weights
    
def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # Prevent overflow
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def forward_pass(X, weights, activation):
    Z, A = {}, {"A0": X}
    for i in range(1, len(weights) // 2):
        Z[f'Z{i}'] = np.dot(A[f'A{i-1}'], weights[f'W{i}']) + weights[f'b{i}']
        A[f'A{i}'] = np.maximum(0, Z[f'Z{i}']) if activation == "relu" else 1 / (1 + np.exp(-Z[f'Z{i}']))

    # Final layer (Softmax for classification)
    last_layer = len(weights) // 2
    Z[f'Z{last_layer}'] = np.dot(A[f'A{last_layer-1}'], weights[f'W{last_layer}']) + weights[f'b{last_layer}']
    A[f'A{last_layer}'] = softmax(Z[f'Z{last_layer}'])
    
    return Z, A

def backward_pass(X, Y, weights, A, Z, activation, weight_decay):
    grads = {}
    m = X.shape[0]
    dA = A[f'A{len(A)-1}'] - Y  # Softmax with cross-entropy simplifies to this

    for i in reversed(range(1, len(weights) // 2 + 1)):
        if activation == "relu":
            dZ = dA * (A[f'A{i}'] > 0)
        elif activation == "sigmoid":
            dZ = dA * A[f'A{i}'] * (1 - A[f'A{i}'])
        elif activation == "tanh":
            dZ = dA * (1 - A[f'A{i}'] ** 2)  # tanh derivative: 1 - tanh^2(x)
        else:
            raise ValueError("Unsupported activation function")

        grads[f'dW{i}'] = np.dot(A[f'A{i-1}'].T, dZ) / m + weight_decay * weights[f'W{i}']
        grads[f'db{i}'] = np.sum(dZ, axis=0, keepdims=True) / m
        dA = np.dot(dZ, weights[f'W{i}'].T)
    
    return grads


def apply_optimizer(optimizer, weights, grads):
    for i in range(1, len(weights) // 2 + 1):
        weights[f'W{i}'] = optimizer.update(weights[f'W{i}'], grads[f'dW{i}'], f'W{i}')
        weights[f'b{i}'] = optimizer.update(weights[f'b{i}'], grads[f'db{i}'], f'b{i}')

def compute_loss(Y, A, loss_function):
    m = Y.shape[0]
    if loss_function == "cross_entropy":
        return -np.sum(Y * np.log(A + 1e-8)) / m
    else:
        return np.mean((Y - A) ** 2) # MSE

def compute_accuracy(Y, A):
    return np.mean(np.argmax(Y, axis=1) == np.argmax(A, axis=1))

def train(config=None):
    with wandb.init(config=config, reinit=True):
        config = wandb.config
        wandb.run.name = (
            "_hl_" + str(config.num_layers) +
            "_hn_" + str(config.hidden_size) +
            "_opt_" + config.optimizer +
            "_act_" + config.activation +
            "_lr_" + str(config.learning_rate) +
            "_bs_" + str(config.batch_size) +
            "_init_" + config.weight_init +
            "_ep_" + str(config.epochs) +
            "_l2_" + str(config.weight_decay)
        )
        
        X_train, y_train, X_val, y_val, _, _ = load_data()
        num_classes = 10
        y_train, y_val = [np.eye(num_classes)[y] for y in [y_train, y_val]]
        
        layers = [784] + [config.hidden_size] * config.num_layers + [num_classes]
        weights = initialize_weights(layers, config.weight_init)
        
        optimizer = Optimizer(method=config.optimizer, learning_rate=config.learning_rate)
        
        for epoch in range(config.epochs):
            for i in range(0, X_train.shape[0], config.batch_size):
                X_batch = X_train[i:i + config.batch_size]
                y_batch = y_train[i:i + config.batch_size]
                
                Z, A = forward_pass(X_batch, weights, config.activation)
                grads = backward_pass(X_batch, y_batch, weights, A, Z, config.activation, config.weight_decay)
                apply_optimizer(optimizer, weights, grads)
                
            Z_train, A_train = forward_pass(X_train, weights, config.activation)
            train_loss = compute_loss(y_train, A_train[f'A{len(A_train)-1}'], config.loss)
            train_acc = compute_accuracy(y_train, A_train[f'A{len(A_train)-1}'])

            Z_val, A_val = forward_pass(X_val, weights, config.activation)
            val_loss = compute_loss(y_val, A_val[f'A{len(A_val)-1}'], config.loss)
            val_acc = compute_accuracy(y_val, A_val[f'A{len(A_val)-1}'])
            
            wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "train_accuracy": train_acc,
                       "val_loss": val_loss, "val_accuracy": val_acc})


sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "activation": {"values": ["relu", "sigmoid", "tanh"]},
        "batch_size": {"values": [32, 64, 128]},
        "epochs": {"values": [5, 10, 20]},
        "hidden_size": {"values": [32, 64, 128]},
        "learning_rate": {"values": [0.01, 0.001, 0.0001]},
        "num_layers": {"values": [2, 3, 4]},
        "optimizer": {"values": ["sgd", "momentum","nag","adam", "rmsprop"]},
        "weight_decay": {"values": [0.0001, 0.0005, 0.001]},
        "weight_init": {"values": ["random", "xavier"]},
        "loss": {"values": ["cross_entropy"]}
    }}

sweep_id = wandb.sweep(sweep_config, project="fashion-mnist-dataset")
wandb.agent(sweep_id, function=train, count=50)







# CORELATION MATRIX (ques-7)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from keras.datasets import fashion_mnist
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Initialize wandb
wandb.init(project="fashion-mnist-sweep", name="confusion-matrix-log")

# Load and preprocess data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0  # Normalize
X_train, X_test = X_train.reshape(X_train.shape[0], -1), X_test.reshape(X_test.shape[0], -1)

# Convert labels to one-hot encoding
y_train_cat, y_test_cat = to_categorical(y_train), to_categorical(y_test)

# Simple Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train_cat, epochs=5, batch_size=128, validation_split=0.1)

# Predictions
y_pred = np.argmax(model.predict(X_test), axis=1)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGn')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("conf_matrix.png")  # Save image

# Log to wandb
wandb.log({"confusion_matrix": wandb.Image("conf_matrix.png")})

# Close wandb run
wandb.finish()