import numpy as np
import torch
from gym3.wrapper import Wrapper
from gym3.types import ValType
from procgen import ProcgenGym3Env


def rgb_to_tensor(rgb: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(rgb).permute(0, 3, 1, 2).to(device) / 255.0


def run_episode(env, agent) -> float:
    obs = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        obs = obs[np.newaxis, :]
        action, _ = agent.act(obs)
        obs, reward, done, _ = env.step(action.squeeze(0))
        total_reward += reward

    return total_reward


class NormalizeReward(Wrapper):

    def __init__(
        self, env: ProcgenGym3Env, gamma: float, ob_space: ValType | None = None, ac_space: ValType | None = None
    ) -> None:
        super().__init__(env, ob_space or env.ob_space, ac_space or env.ac_space)
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
