from lnn import LiquidNeuralNetwork
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DQNAgent:
    def __init__(self, input_shape, num_actions, memory_size=10000, batch_size=32, gamma=0.99, lr=0.001):
        self.num_actions = num_actions
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # LNN as the Q-network
        self.q_network = LiquidNeuralNetwork(input_shape, num_actions).to(self.device)
        self.target_network = LiquidNeuralNetwork(input_shape, num_actions).to(self.device)

        self.update_target_network()  # Synchronize networks

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(2, self.num_actions)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Add batch size
        q_values, _ = self.q_network(state)
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

        # Get Q-values for current states
        q_values, _ = self.q_network(state_batch)
        q_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Get Q-values for next states using the target network
        next_q_values, _ = self.target_network(next_state_batch)
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

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
