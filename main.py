import gymnasium as gym
from modelutils import *
from dqn import *

if __name__ == '__main__':

    env = gym.make('CarRacing-v2', render_mode='ansi')

    # steering, gas, brake
    num_actions = 3
    input_shape = (1, 96, 96)

    agent = DQNAgent(input_shape=input_shape, num_actions=num_actions)
    train_agent(agent, env, "test_save.pth", num_episodes=100)

    test_agent(agent, gym.make('CarRacing-v2', render_mode='human'), num_episodes=5)

    env.close()
