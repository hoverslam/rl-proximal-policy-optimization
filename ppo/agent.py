from ppo.utils import rgb_to_tensor

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Distribution, Categorical


class PPOAgent:

    def __init__(self, env_name: str, env_mode: str) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ImpalaCNN(num_actions=15).to(self._device)
        self.env_name = env_name
        self.env_mode = env_mode

    def act(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            policy, value = self.model(rgb_to_tensor(obs, self._device))
            action = policy.sample().cpu().numpy()

        return action, value.cpu().numpy()

    def save_model(self, fpath: str) -> None:
        torch.save(self.model.state_dict(), fpath)

    def load_model(self, fpath: str) -> None:
        self.model.load_state_dict(torch.load(fpath))


class ImpalaCNN(nn.Module):

    def __init__(self, num_actions: int) -> None:
        super().__init__()
        self._block1 = ImpalaBlock(3, 16)
        self._block2 = ImpalaBlock(16, 32)
        self._block3 = ImpalaBlock(32, 32)
        self._fc = nn.Linear(8 * 8 * 32, 256)
        self._actor = nn.Linear(256, num_actions)
        self._critic = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> tuple[Distribution, torch.Tensor]:
        x = self._block1(x)
        x = self._block2(x)
        x = self._block3(x)

        x = F.relu(x)
        x = x.reshape(x.size(0), -1)
        x = self._fc(x)
        x = F.relu(x)

        policy = Categorical(logits=self._actor(x))
        value = self._critic(x)

        return policy, value


class ImpalaBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self._conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self._max_pool = nn.MaxPool2d(3, 2, 1)
        self._res1 = ResBlock(out_channels)
        self._res2 = ResBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._conv(x)
        out = self._max_pool(out)
        out = self._res1(out)
        out = self._res2(out)

        return out


class ResBlock(nn.Module):

    def __init__(self, block_channels: int) -> None:
        super().__init__()
        self._conv1 = nn.Conv2d(block_channels, block_channels, 3, 1, 1)
        self._conv2 = nn.Conv2d(block_channels, block_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(x)
        out = self._conv1(out)
        out = F.relu(out)
        out = self._conv2(out)

        return out + x
