from lnnsim import LiquidNeuralNetwork
import lnn
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DQNAgent:
    def __init__(self, model, num_actions, memory_size=10000, batch_size=32, gamma=0.99, lr=0.001):
        self.model = model
        self.num_actions = num_actions
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.015
        self.epsilon_decay = 0.99995

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # LNN as the Q-network
        if self.model == "lnn-sim":
            self.q_network = LiquidNeuralNetwork(256, num_actions).to(self.device)
            self.target_network = LiquidNeuralNetwork(256, num_actions).to(self.device)
        else:
            self.q_network = lnn.LiquidRecurrentDQN(256, 1, 4, 5).to(self.device)
            self.target_network = lnn.LiquidRecurrentDQN(256, 1, 4, 5).to(self.device)

        self.update_target_network()  # Synchronize networks

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Exploration: Randomly select one of the 5 actions
            return np.random.randint(0, self.num_actions)

        # Exploitation: Use the Q-network to predict the best action
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Inference mode, no need to track gradients
        with torch.no_grad():
            q_values, _ = self.q_network.forward(state)

        return torch.argmax(q_values).item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of experiences
        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        # Convert batches to tensors
        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        self.q_network.reset_hidden()  # Reset the hidden state before processing the batch
        self.target_network.reset_hidden()  # Do the same for the target network

        if self.model == "lnn-sim":
            q_values, _ = self.q_network.forward(state_batch)
            next_q_values, _ = self.target_network.forward(next_state_batch)
        else:
            delta_t = torch.ones((self.batch_size, 3))
            q_values, _ = self.q_network.forward(state_batch, delta_t)
            next_q_values, _ = self.target_network.forward(next_state_batch, delta_t)

        # Get Q-values for current states
        q_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Get Q-values for next states using the target network
        next_q_values = next_q_values.max(1)[0]

        # Calculate target Q-values
        target_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))

        # Loss and optimization
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon (exploration rate)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.q_network.reset_hidden()

    def update_target_network(self):
        with torch.no_grad():
            self.target_network.load_state_dict(self.q_network.state_dict())
