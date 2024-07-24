import numpy as np
import torch
from gym3.wrapper import Wrapper


def rgb_to_tensor(rgb: np.ndarray, device: str) -> torch.Tensor:
    """Convert an RGB image to a tensor.

    Args:
        rgb (np.ndarray): The input RGB image as a NumPy array (H x W x C).
        device (str): The device to place the tensor on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: The image as a tensor normalized to [0, 1].
    """
    return torch.from_numpy(rgb).permute(0, 3, 1, 2).to(device) / 255.0


def evaluate_agent(agent, env, max_steps: int) -> list:
    """Evaluate an agent in the given environment for a maximum number of steps.

    Args:
        agent: The agent to be evaluated.
        env: The environment in which to evaluate the agent.
        max_steps (int): The maximum number of steps for the evaluation.

    Returns:
        list: Mean rewards of the agent across episodes.
    """
    _, obs, _ = env.observe()
    total_reward = []
    firsts = []
    step = 0
    while step < max_steps:
        env.act(agent.act(obs["rgb"])[0])
        reward, obs, first = env.observe()
        total_reward.append(reward)
        firsts.append(first)
        step += 1

    return compute_mean_rewards(total_reward, firsts)


def compute_mean_rewards(rewards: list[np.ndarray], firsts: list[np.ndarray]) -> list:
    """Compute the mean epsisode rewards of the agent given a sequence of steps.

    Args:
        rewards (list[np.ndarray]): List of reward arrays.
        firsts (list[np.ndarray]): List of first-step indicators (1 = new episode).

    Returns:
        list: Mean rewards per episode.
    """
    total_rewards = np.stack(rewards, axis=1).sum(axis=1)
    num_episodes = np.stack(firsts, axis=1).sum(axis=1) + 1

    return (total_rewards / num_episodes).tolist()


class NormalizeReward(Wrapper):
    """Normalize rewards in the environment using running mean and variance."""

    def __init__(self, env, gamma: float) -> None:
        """Initialize the Normalizer.

        Args:
            env: The environment to be wrapped.
            gamma (float): Discount factor for rewards.
        """
        super().__init__(env)
        self.env = env
        self.gamma = gamma
        self.return_rms_mean = np.zeros((), dtype=np.float64)
        self.return_rms_var = np.ones((), dtype=np.float64)
        self.return_rms_count = 1e-4
        self.accumulated_reward = np.zeros((self.num,), dtype=np.float32)

    def observe(self) -> tuple[np.ndarray, dict, np.ndarray]:
        reward, obs, first = self.env.observe()
        self.accumulated_reward = self.accumulated_reward * self.gamma * (1 - first) + reward
        self.update_return_rms(self.accumulated_reward)
        normalized_reward = reward / np.sqrt(self.return_rms_var + 1e-8)

        return normalized_reward, obs, first

    def update_return_rms(self, x: np.ndarray) -> None:
        """Update running mean and variance of returns.

        Args:
            x (np.ndarray): Current accumulated rewards.
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.return_rms_mean
        tot_count = self.return_rms_count + batch_count

        new_mean = self.return_rms_mean + delta * batch_count / tot_count
        m_a = self.return_rms_var * self.return_rms_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.return_rms_count * batch_count / tot_count
        new_var = m2 / tot_count
        new_count = tot_count

        self.return_rms_mean, self.return_rms_var, self.return_rms_count = new_mean, new_var, new_count
