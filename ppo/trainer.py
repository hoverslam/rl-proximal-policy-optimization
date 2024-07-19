from ppo.agent import PPOAgent
from ppo.utils import rgb_to_tensor

import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from procgen import ProcgenGym3Env


class PPOTrainer:

    def __init__(self, agent: PPOAgent) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = agent.model.to(self._device)
        self._env_name = agent.env_name
        self._env_mode = agent.env_mode

        self._checkpoint_dir = "./checkpoints"
        os.makedirs(self._checkpoint_dir, exist_ok=True)

    def train(
        self,
        num_envs: int,
        num_levels: int,
        num_iterations: int,
        num_steps: int,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        gamma: float,
        gae_lambda: float,
        clip_range: float,
        entropy_coef: float,
        vf_coef: float,
        num_checkpoints: int | None = 10,
    ) -> None:
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=learning_rate)
        env = ProcgenGym3Env(
            num=num_envs,
            env_name=self._env_name,
            distribution_mode=self._env_mode,
            start_level=0,
            num_levels=num_levels,
        )

        self._model.train()
        for i in range(num_iterations):
            start_time = time.time()

            data = self._rollout(env, num_steps, gamma, gae_lambda)
            buffer = RolloutBuffer(data)
            loader = DataLoader(buffer, batch_size=batch_size, shuffle=True)

            for _ in range(num_epochs):
                for obs, actions, old_log_probs, old_values, returns, advantages in loader:
                    policy, new_values = self._model(obs.to(self._device))
                    entropy = policy.entropy().mean()
                    new_values = new_values.squeeze()

                    # Normalize advantages batchwise (OpenAI baseline PPO)
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # Policy loss (clipped)
                    new_log_probs = policy.log_prob(actions)
                    ratios = torch.exp(new_log_probs - old_log_probs)
                    policy_loss = torch.min(
                        ratios * advantages, torch.clamp(ratios, 1 - clip_range, 1 + clip_range) * advantages
                    ).mean()

                    # Value loss (clipped) (OpenAI baseline PPO)
                    new_values_clipped = old_values + torch.clamp(new_values - old_values, -clip_range, clip_range)
                    value_loss_clipped = torch.pow(new_values_clipped - returns, 2.0)
                    value_loss_normal = torch.pow(new_values - returns, 2.0)
                    value_loss = torch.max(value_loss_normal, value_loss_clipped).mean()

                    # Update parameters using the combined loss
                    loss = vf_coef * value_loss - policy_loss - entropy_coef * entropy
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self._model.parameters(), 0.5)  # Gradient clipping (OpenAI baseline PPO)
                    optimizer.step()

            print(f"{(i+1):{len(str(num_iterations))}}/{num_iterations}: runtime={time.time() - start_time:.1f}s")

            # Create <num_checkpoints> evenly spaced checkpoints
            if num_checkpoints is not None:
                start = (num_iterations - 1) // 10
                end = num_iterations - 1
                if i in torch.linspace(start, end, steps=10, dtype=torch.int).tolist():
                    fname = f"{self._env_name}_{self._env_mode}_{i+1}.pt"
                    torch.save(self._model.state_dict(), f"{self._checkpoint_dir}/{fname}")

    def _rollout(self, env: ProcgenGym3Env, num_steps: int, gamma: float, gae_lambda: float) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            data = [[], [], [], [], [], []]  # obs, actions, rewards, log_probs, values, masks
            _, obs, _ = env.observe()

            step = 0
            while step < num_steps:
                obs = rgb_to_tensor(obs["rgb"], self._device)
                policy, value = self._model(obs)
                action = policy.sample()
                log_prob = policy.log_prob(action)

                env.act(action.cpu().numpy())
                reward, next_obs, first = env.observe()

                obs = obs.cpu()  # If on CUDA, torch.stack() destroys the images with weird artifacts (???)
                for i, item in enumerate((obs, action, reward, log_prob, value, first)):
                    data[i].append(item)

                obs = next_obs
                step += 1

            _, next_value = self._model(rgb_to_tensor(next_obs["rgb"], self._device))

        rewards = torch.from_numpy(np.stack(data[2], axis=1)).to(self._device)
        values = torch.stack(data[4], dim=1).squeeze(-1)
        masks = torch.from_numpy(~np.stack(data[5], axis=1)).to(self._device)  # ~ >>> True <=> False
        advantages = self._compute_gaes(rewards, values, next_value, masks, gamma, gae_lambda)
        returns = advantages + values  # Compute returns using GAE (OpenAI baseline PPO)

        return {
            "obs": torch.stack(data[0], dim=1).flatten(0, 1),  # (num_envs * num_steps, channels, height, width)
            "actions": torch.stack(data[1], dim=1).flatten(),  # (num_envs * num_steps, )
            "log_probs": torch.stack(data[3], dim=1).flatten(),  # (num_envs * num_steps, )
            "values": values.flatten(),  # (num_envs * num_steps, )
            "returns": returns.flatten(),  # (num_envs * num_steps, )
            "advantages": advantages.flatten(),  # (num_envs * num_steps, )
        }

    def _compute_returns(self, rewards: torch.Tensor, masks: torch.Tensor, gamma: float) -> torch.Tensor:
        num_envs, num_steps = rewards.shape
        discounted_returns = torch.zeros_like(rewards, device=self._device)
        delta = torch.zeros((num_envs,), device=self._device)

        for t in reversed(range(num_steps)):
            delta = rewards[:, t] + gamma * delta * masks[:, t]
            discounted_returns[:, t] = delta

        return discounted_returns

    def _compute_gaes(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_value: torch.Tensor,
        masks: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> torch.Tensor:
        num_env, num_steps = rewards.shape
        advantages = torch.zeros_like(rewards, dtype=torch.float, device=self._device)
        gae = torch.zeros((num_env,), dtype=torch.float, device=self._device)
        next_value = next_value.squeeze()

        for t in reversed(range(num_steps)):
            delta = rewards[:, t] + gamma * next_value * masks[:, t] - values[:, t]
            gae = delta + gamma * gae_lambda * masks[:, t] * gae
            advantages[:, t] = gae
            next_value = values[:, t]

        return advantages


class RolloutBuffer(Dataset):

    def __init__(self, data: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self._obs = data["obs"]
        self._actions = data["actions"]
        self._log_probs = data["log_probs"]
        self._values = data["values"]
        self._returns = data["returns"]
        self._advantages = data["advantages"]

    def __len__(self) -> int:
        return len(self._actions)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self._obs[idx],
            self._actions[idx],
            self._log_probs[idx],
            self._values[idx],
            self._returns[idx],
            self._advantages[idx],
        )
