import argparse
from modelutils import *
from dqn import *
from carRacing import CarRacing

if __name__ == '__main__':
    parser = argparse.ArgumentParser("dqn-lnn")

    parser.add_argument("--model", type=str, default="lnn-sim")
    parser.add_argument("--mode", type=str, default="train")
    args = parser.parse_args()

    env = CarRacing(continuous=False)

    # steering, gas, brake
    num_actions = 5

    agent = DQNAgent(model=args.model, num_actions=num_actions)

    if args.mode == "train":
        train_agent(agent, env, "test_save", num_episodes=100)
    else:
        test_agent(agent, CarRacing(continuous=False, render_mode='human'), num_episodes=5)

    env.close()
