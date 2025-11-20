"""
Neural Network class for MNIST classification
"""

import torch


class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_size=784, hidden_layers=[64, 32], output_size=10):
        super(NeuralNetwork, self).__init__()
        
        # Build layers dynamically from config
        self.layers = torch.nn.ModuleList()
        
        # Input to first hidden layer
        prev_size = input_size
        for hidden_size in hidden_layers:
            self.layers.append(torch.nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Last hidden to output layer
        self.layers.append(torch.nn.Linear(prev_size, output_size))


    def train_forward(self, data):
        data = data.view(data.size(0), -1)  # Flatten
        
        # Apply all layers except the last with ReLU
        for layer in self.layers[:-1]:
            data = torch.nn.functional.relu(layer(data))
        
        # Last layer without activation
        data = self.layers[-1](data)
        return data
