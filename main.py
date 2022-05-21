import numpy as np
import itertools
import matplotlib.pyplot as plt

class NeuralNetwork():
    ITERATION_NUMBER = 2000
    def __init__(self, layers_dimensions, training_rate=2) -> None:
        self.layers_dimensions = layers_dimensions
        self.layer_number = len(self.layers_dimensions)
        self.activation_function = lambda x: 1/(1 + np.exp(-x))
        self.activation_derivative = lambda z: np.multiply(z, (1 - z))
        self.training_rate = training_rate
    
    def train(self, inputs, expected_outputs):
        self.init_weights()
        outputs_error = []
        for _ in range(self.ITERATION_NUMBER):
            layers = self.forward_propegation(inputs)
            outputs_error.append(self._calculate_results_error(expected_outputs, layers[-1]))
            errors = self.backward_propegation(layers, expected_outputs)
            self.update_weights(errors, layers)
        plt.plot(outputs_error)
        return layers[-1], self.weights


    def update_weights(self, errors, layers):
        for index, (layer, error) in enumerate(zip(layers[:-1], errors[1:])):
            self.weights[index] -= self.training_rate * np.matmul(np.transpose(error), self._add_bias_neruon(layer))
    
    def _add_bias_neruon(self, layer):
        return np.c_[layer, np.ones(layer.shape[0])]
    
    def _calculate_error(self, layer, error_term):
        return np.multiply(self.activation_derivative(layer), error_term)

    def _calculate_results_error(self, expected_outputs, outputs):
        return np.sum(np.square(expected_outputs[:, np.newaxis] - outputs)) / 8

    def backward_propegation(self, layers, expected_outputs):
        #TODO check what about the bias - are we using it properly?
        # Seperating the case for the last layers and the other layers.
        errors = [self._calculate_error(layers[-1], (layers[-1] - expected_outputs[:, np.newaxis]))]

        for (weight, layer) in reversed(list(zip(self.weights, layers[:-1]))):
            non_bias_weights = weight[:,:-1]
            error_term = np.matmul(errors[0], non_bias_weights)
            error = np.multiply(self.activation_derivative(layer), error_term)
            errors.insert(0, error)
        return errors

    def forward_propegation(self, inputs):
        layers = [inputs]
        for weight in self.weights:
            layers.append(self.activation_function(np.matmul(self._add_bias_neruon(layers[-1]), np.transpose(weight))))
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
    network = NeuralNetwork([3,3,1])
    network.train(*generate_binary_dataset(3))

if __name__ == "__main__":
    main()