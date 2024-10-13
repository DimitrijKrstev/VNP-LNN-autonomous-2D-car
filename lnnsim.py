import torch
import torch.nn as nn


class LiquidNeuralNetwork(nn.Module):
    def __init__(self, hidden_size, num_actions, tau=1.0):
        super(LiquidNeuralNetwork, self).__init__()

        self.tau = tau  # Time constant to introduce liquidity
        self.hidden_size = hidden_size
        self.num_actions = num_actions

        # Convolutional layers for grayscale 96x96 image input
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),  # Output: (32, 23, 23)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # Output: (64, 10, 10)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # Output: (64, 8, 8)
            nn.ReLU()
        )

        self.fc = None
        self._initialize_fc(num_actions)

        # Recurrent layer to introduce liquid dynamics
        self.W_in = nn.Linear(self.hidden_size, self.hidden_size)  # Input weights
        self.W_out = nn.Linear(self.hidden_size, self.hidden_size)  # Recurrent weights
        self.hidden = None  # Initialize hidden state to None

    def _initialize_fc(self, num_actions):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 96, 96)
            conv_output = self.conv(dummy_input)
            flattened_size = conv_output.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.hidden_size)
        )

        self.out = nn.Linear(self.hidden_size, num_actions)

    def forward(self, x, t=1.0):
        x = x / 255.0  # Normalize pixel values to [0, 1]
        x = self.conv(x)  # Apply convolutional layers
        x = x.view(x.size(0), -1)  # Flatten the convolution output
        x = self.fc(x)  # Feedforward through fully connected layers

        if self.hidden is None:
            self.hidden = torch.zeros(x.size(0), self.hidden_size).to(x.device)

        # Liquid recurrent dynamics
        # dh/dt = (-h + tanh(W_in(x) + W_out(h)) / tau
        dh = (-self.hidden + torch.tanh(self.W_in(x) + self.W_out(self.hidden))) / self.tau

        # Update hidden state with liquid dynamics
        self.hidden = self.hidden + dh * t

        output = self.out(self.hidden)
        return output, self.hidden

    def reset_hidden(self):
        self.hidden = None
