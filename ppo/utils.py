import numpy as np
import torch


def rgb_to_tensor(rgb: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(rgb).permute(0, 3, 1, 2).to(device) / 255.0


class RewardNormalizer:
    """Normalizes rewards such that their exponential moving average has a fixed variance."""

    def __init__(self, gamma: float):
        """Initializes the Normalizer with a discount factor.

        Args:
            gamma (float): The discount factor used in the exponential moving average.
        """
        self._gamma = gamma
        self._reward_ema = 0
        self._reward_ema_sq = 0
        self._t = 0

    def __call__(self, rewards: list[np.ndarray]) -> torch.Tensor:
        """Normalizes the input rewards.

        Args:
            rewards (list[np.ndarray]): The rewards to normalize, expected to be a list of numpy arrays
                                        with shape (num_envs, ).

        Returns:
            torch.Tensor: The normalized rewards with shape (num_envs, num_steps, 1).
        """
        num_steps = len(rewards)
        num_envs = rewards[0].shape[0]
        normalized_rewards = np.zeros((num_envs, num_steps, 1))

        for i, reward in enumerate(rewards):
            self._t += 1
            self._reward_ema = self._gamma * self._reward_ema + (1 - self._gamma) * reward
            self._reward_ema_sq = self._gamma * self._reward_ema_sq + (1 - self._gamma) * (reward**2)
            variance = self._reward_ema_sq - self._reward_ema**2
            normalized = (reward - self._reward_ema) / (np.sqrt(variance) + 1e-8)
            normalized_rewards[:, i, 0] = normalized

        return torch.tensor(normalized_rewards)
