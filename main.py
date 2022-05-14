from matplotlib.pyplot import axis
import numpy as np
import itertools
from functools import reduce

class NeuralNetwork():
    def __init__(self, layers_dimensions) -> None:
        self.layers_dimensions = layers_dimensions
        self.layer_number = len(self.layers_dimensions)
        self.activation_function = lambda x: 1/(1 + np.exp(-x))
    
    def train(self):
        self.init_weights()
        self.forward_propegation(np.array([0, 1, 1]))
    
    def forward_propegation(self, inputs):
        layers = [inputs]
        for weight in self.weights:
            layers.append(self.activation_function(np.matmul(weight, np.append(layers[-1], 1))))
        return layers

    def init_weights(self):
        """
        Generate array of weights metrics for each layer.
        Each weight and bias is generated with 0 mean and 1 sigma
        """
        self.weights = []
        for index in range(len(self.layers_dimensions) - 1):
            self.weights.append(np.random.normal(size=(self.layers_dimensions[index + 1], self.layers_dimensions[index] + 1)))
        return self.weights

def generate_binary_dataset(input_size):
    inputs = np.array(list(itertools.product([0,1], repeat=input_size)))
    modulo_func = np.vectorize(lambda x: x % 2)
    outputs = modulo_func(np.sum(inputs, axis=1))
    return inputs, outputs


def main():
    generate_binary_dataset(3)
    # network = NeuralNetwork([3,3,1])
    # network.train()

if __name__ == "__main__":
    main()