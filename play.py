from ppo import PPOAgent

import argparse

import gym3
from procgen import ProcgenGym3Env


ENVS = [
    "bigfish",
    "coinrun",
    "ninja",
    "leaper",
    "starpilot",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Let an agent play in an environment.")
    parser.add_argument(
        "--env_name",
        type=str,
        default="coinrun",
        choices=ENVS,
        help="Name of the environment to play",
    )
    parser.add_argument(
        "--env_mode",
        type=str,
        default="easy",
        choices=["easy", "hard"],
        help="Difficulty mode of the environment. Choose between 'easy' and 'hard'.",
    )
    args = parser.parse_args()

    agent = PPOAgent(args.env_name, args.env_mode)
    agent.load_model(f"./pretrained/{args.env_name}_{args.env_mode}_25m.pt")

    env = ProcgenGym3Env(num=1, env_name=args.env_name, distribution_mode=args.env_mode, render_mode="rgb_array")
    env = gym3.ViewerWrapper(env, info_key="rgb")

    _, obs, _ = env.observe()
    while True:
        action, _ = agent.act(obs["rgb"])
        env.act(action)
        _, obs, _ = env.observe()


if __name__ == "__main__":
    main()
