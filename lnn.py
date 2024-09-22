import torch
import torch.nn as nn
import torch.nn.functional as fn


class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_shape, num_actions, hidden_size=256):
        super(LiquidNeuralNetwork, self).__init__()

        # Convolutional layers to process grayscale image input
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)

        self.flatten = nn.Flatten()

        # Update LSTM input size based on conv layer output dimensions (64 * 10 * 10 = 6400)
        self.lstm = nn.LSTM(input_size=64 * 10 * 10, hidden_size=hidden_size, batch_first=True)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, num_actions)

        # Parameter to simulate time-varying liquid-like dynamics
        self.time_varying_weights = nn.Parameter(torch.rand(hidden_size))

    def forward(self, x, hidden_state=None):
        # Forward pass through convolutional layers
        x = fn.relu(self.conv1(x))
        x = fn.relu(self.conv2(x))
        x = fn.relu(self.conv3(x))

        # Flatten the output
        x = self.flatten(x)

        # Reshape for the LSTM (add time dimension)
        x = x.unsqueeze(1)  # (batch_size, sequence_length, input_size)

        # LSTM layer (liquid dynamics)
        lstm_out, hidden_state = self.lstm(x, hidden_state)

        # Apply time-dependent behavior using sine modulation (liquid-like dynamics)
        time_dependent_output = lstm_out * torch.sin(self.time_varying_weights)

        # Output layer
        output = self.fc(time_dependent_output[:, -1, :])  # Take only the last output of the sequence

        return output, hidden_state
