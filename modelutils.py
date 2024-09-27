import torch
import numpy as np
import cv2


def train_agent(agent, env, save_path, num_episodes=100, target_update_interval=10):
    agent.q_network.train()

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = preprocess_image(state)

        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _, info = env.step(action)

            next_state = preprocess_image(next_state)

            agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            agent.train()

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

        if (episode + 1) % target_update_interval == 0:
            save_agent(agent, save_path + str((episode + 1) / target_update_interval) + '.pth')
            agent.update_target_network()

    save_agent(agent, save_path)


def test_agent(agent, env, num_episodes=5, render=True):
    """
    Function to test the trained Liquid Neural Network agent in the CarRacing environment.

    Args:
        agent: The trained DQNAgent (with a Liquid Neural Network) to be tested.
        env: The CarRacing-v2 gym environment.
        num_episodes: Number of episodes to test the agent on.
        render: Whether to render the environment during testing.
    """
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = preprocess_image(state)
        total_reward = 0
        done = False
        hidden_state = None  # To store the LSTM hidden state

        while not done:
            if render:
                env.render()

            # Convert state to PyTorch tensor and add batch dimension
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # Forward pass through the Q-network to get Q-values for the current state
            with torch.no_grad():
                q_values, hidden_state = agent.q_network(state_tensor, hidden_state)

            # Select action with the highest Q-value (greedy action)
            action = torch.argmax(q_values).item()

            next_state, reward, done, _, info = env.step(action)

            next_state = preprocess_image(next_state)

            total_reward += reward

            state = next_state

        print(f"Test Episode {episode + 1}: Total Reward = {total_reward}")

    env.close()


def save_agent(agent, file_path):
    """
    Save the trained agent model locally.

    Args:
        agent: The trained DQNAgent (or the model to save).
        file_path: Path to save the model (e.g., 'agent_model.pth').
    """
    torch.save(agent.q_network.state_dict(), file_path)
    print(f"Model saved to {file_path}")


def save_full_agent(agent, optimizer, file_path):
    """
    Save the full state of the trained agent, including model weights and optimizer state.

    Args:
        agent: The trained DQNAgent.
        optimizer: The optimizer used for training.
        file_path: Path to save the full state (e.g., 'full_agent.pth').
    """
    checkpoint = {
        'model_state_dict': agent.q_network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'other_data': {
            # You can add anything else you'd like to save here
            # like the current episode number, epsilon value, etc.
        }
    }
    torch.save(checkpoint, file_path)
    print(f"Full agent saved to {file_path}")


def load_agent(agent, file_path):
    """
    Load the trained agent's model from file.

    Args:
        agent: The DQNAgent (the model architecture should be initialized).
        file_path: Path to the saved model (e.g., 'trained_agent.pth').
    """
    agent.q_network.load_state_dict(torch.load(file_path))
    # agent.q_network.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))
    agent.q_network.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {file_path}")


def load_full_agent(agent, optimizer, file_path):
    """
    Load the full state of the trained agent, including model weights and optimizer state.

    Args:
        agent: The DQNAgent (architecture should be initialized).
        optimizer: The optimizer to load the state into.
        file_path: Path to the full saved state (e.g., 'full_agent.pth').
    """
    checkpoint = torch.load(file_path)
    agent.q_network.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # If you saved additional data, you can also load them here
    # For example, if you saved episode number, you could retrieve it like this:
    # episode = checkpoint['other_data']['episode']

    agent.q_network.eval()  # Set the model to evaluation mode
    print(f"Full agent loaded from {file_path}")


def preprocess_image(state):
    # gray_image = Image.fromarray(state).convert('L')
    # gray_array = np.array(gray_image, dtype=np.float32) / 255.0
    # return gray_array

    # Convert image to grayscale
    grayscale_image = np.mean(state, axis=2, dtype=np.float32)

    # Normalize pixel values between 0 and 1
    grayscale_image /= 255.0

    # Resize the image to 96x96 if required (depends on your model's input size)
    grayscale_image = cv2.resize(grayscale_image, (96, 96))

    # Add a channel dimension (1, 96, 96)
    grayscale_image = np.expand_dims(grayscale_image, axis=0)

    return grayscale_image
