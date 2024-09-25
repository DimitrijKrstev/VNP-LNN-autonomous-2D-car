import argparse
import gymnasium as gym
from modelutils import *
from dqn import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser("dqn-lnn")

    parser.add_argument("--model", type=str, default="lnn-sim")
    parser.add_argument("--mode", type=str, default="train")
    args = parser.parse_args()

    env = gym.make('CarRacing-v2', continuous=False)

    # steering, gas, brake
    num_actions = 5

    # grayscale 96x96 image
    input_shape = (1, 96, 96)

    agent = DQNAgent(model=args.model, input_shape=input_shape, num_actions=num_actions)

    if args.mode == "train":
        train_agent(agent, env, "test_save", num_episodes=100)
    else:
        test_agent(agent, gym.make('CarRacing-v2', render_mode='human'), num_episodes=5)

    env.close()
