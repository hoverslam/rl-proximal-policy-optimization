from ppo import PPOAgent, PPOTrainer

import os
import argparse


ENVS = [
    "bigfish",
    "bossfight",
    "caveflyer",
    "chaser",
    "climber",
    "coinrun",
    "dodgeball",
    "fruitbot",
    "heist",
    "jumper",
    "leaper",
    "maze",
    "miner",
    "ninja",
    "plunder",
    "starpilot",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train agent on Procgen Benchmark.")
    parser.add_argument(
        "--env_name",
        type=str,
        choices=ENVS,
        help="Name of the environment to train on.",
    )
    parser.add_argument(
        "--env_mode",
        type=str,
        default="easy",
        choices=["easy", "hard"],
        help="Difficulty mode of the environment. Choose between 'easy' and 'hard'.",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=64,
        help="Number of parallel environments to use during training.",
    )
    parser.add_argument(
        "--num_iters",
        type=int,
        default=1525,
        help="Number of training iterations to perform.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=256,
        help="Number of steps to collect from each environment per iteration.",
    )
    parser.add_argument(
        "--num_levels",
        type=int,
        default=200,
        help="Number of levels generated by the environment.",
    )
    parser.add_argument(
        "--num_chkpts",
        type=int,
        default=None,
        help="Number of evenly spaced checkpoints to save during training. If None, checkpoints are not saved.",
    )
    parser.add_argument(
        "--num_evals",
        type=int,
        default=None,
        help="Number of evenly spaced evaluations to perform during training. If None, evaluations are not performed.",
    )
    args = parser.parse_args()

    print("#", 20 * "-")
    for arg, value in vars(args).items():
        print(f"# {arg}: {value}")
    print("#", 20 * "-", end="\n\n")

    agent = PPOAgent(args.env_name, args.env_mode)
    trainer = PPOTrainer(agent)
    trainer.train(
        num_envs=args.num_envs,
        num_iterations=args.num_iters,
        num_steps=args.num_steps,
        num_epochs=3,
        num_levels=args.num_levels,
        num_batches=8,
        learning_rate=5e-4,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coef=0.01,
        vf_coef=0.5,
        num_checkpoints=args.num_chkpts,
        num_evaluations=args.num_evals,
    )

    # Write the scores to a CSV file in the results folder
    if args.num_evals is not None:
        result_dir = "./results"
        os.makedirs(result_dir, exist_ok=True)

        fname = f"{agent.env_name}_{agent.env_mode}"
        results = trainer.logger["scores"]
        with open(f"{result_dir}/{fname}.csv", "w", newline="") as csv:
            csv.write("timestep,set,agent,score\n")
            for set, iteration in results.items():
                for ts, scores in iteration:
                    for agent, score in enumerate(scores):
                        csv.write(f"{ts},{set},{agent},{score}\n")


if __name__ == "__main__":
    main()
