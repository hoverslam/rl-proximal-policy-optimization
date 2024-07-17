import numpy as np
import torch


def rgb_to_tensor(rgb: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(rgb).permute(0, 3, 1, 2).to(device) / 255.0


def normalize_rewards(rewards: torch.Tensor, env_name: str, env_mode: str) -> torch.Tensor:
    if env_mode not in ["easy", "hard"]:
        raise ValueError("Reward normalization only available for 'easy' and 'hard' mode.")

    # From: Cobbe et al. (2019) Leveraging Procedural Generation to Benchmark Reinforcement Learning, Appendix C.
    normalization_constants = {
        "bigfish": {"hard": [0.0, 40.0], "easy": [1.0, 40.0]},
        "bossfight": {"hard": [0.5, 13.0], "easy": [0.5, 13.0]},
        "caveflyer": {"hard": [2.0, 13.4], "easy": [3.5, 12.0]},
        "chaser": {"hard": [0.5, 14.2], "easy": [0.5, 13.0]},
        "climber": {"hard": [1.0, 12.6], "easy": [2.0, 12.6]},
        "coinrun": {"hard": [5.0, 10.0], "easy": [5.0, 10.0]},
        "dodgeball": {"hard": [1.5, 19.0], "easy": [1.5, 19.0]},
        "fruitbot": {"hard": [-0.5, 27.2], "easy": [-1.5, 32.4]},
        "heist": {"hard": [2.0, 10.0], "easy": [3.5, 10.0]},
        "jumper": {"hard": [1.0, 10.0], "easy": [3.0, 10.0]},
        "leaper": {"hard": [1.5, 10.0], "easy": [3.0, 10.0]},
        "maze": {"hard": [4.0, 10.0], "easy": [5.0, 10.0]},
        "miner": {"hard": [1.5, 20.0], "easy": [1.5, 13.0]},
        "ninja": {"hard": [2.0, 10.0], "easy": [3.5, 10.0]},
        "plunder": {"hard": [3.0, 30.0], "easy": [4.5, 30.0]},
        "starpilot": {"hard": [1.5, 35.0], "easy": [2.5, 64.0]},
    }

    r_min = normalization_constants[env_name][env_mode][0]
    r_max = normalization_constants[env_name][env_mode][1]

    return (rewards - r_min) / (r_max - r_min)
