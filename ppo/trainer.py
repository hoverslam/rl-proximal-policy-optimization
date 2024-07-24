from ppo.agent import PPOAgent
from ppo.utils import NormalizeReward, Wrapper, rgb_to_tensor, evaluate_agent

import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from procgen import ProcgenGym3Env


class PPOTrainer:
    """A class to train a PPOAgent using the Proximal Policy Optimization (PPO) algorithm."""

    def __init__(self, agent: PPOAgent) -> None:
        """Initialize the PPOTrainer.

        Args:
            agent (PPOAgent): The PPO agent to be trained.
        """
        self._agent = agent
        self._env_name = agent.env_name
        self._env_mode = agent.env_mode

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = agent.model.to(self._device)
        self.logger = {}

    def train(
        self,
        num_envs: int,
        num_levels: int,
        num_iterations: int,
        num_steps: int,
        num_epochs: int,
        num_batches: int,
        learning_rate: float,
        gamma: float,
        gae_lambda: float,
        clip_range: float,
        entropy_coef: float,
        vf_coef: float,
        num_checkpoints: int | None = None,
        num_evaluations: int | None = None,
    ) -> None:
        """Train the PPO agent.

        Args:
            num_envs (int): Number of parallel environments.
            num_levels (int): Number of levels seen during training.
            num_iterations (int): Number of training iterations.
            num_steps (int): Number of steps per rollout.
            num_epochs (int): Number of epochs per update.
            num_batches (int): Number of batches per epoch.
            learning_rate (float): Learning rate for the optimizer.
            gamma (float): Discount factor for rewards.
            gae_lambda (float): Lambda for Generalized Advantage Estimation (GAE).
            clip_range (float): Clipping range for policy and value loss.
            entropy_coef (float): Coefficient for entropy bonus.
            vf_coef (float): Coefficient for value function loss.
            num_checkpoints (int | None, optional): Number of checkpoints to save. Defaults to None.
            num_evaluations (int | None, optional): Number of evaluations during training. Defaults to None.
        """
        checkpoint_dir = "./checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        if num_checkpoints is not None:
            if num_iterations > 2:
                checkpoints = np.linspace(start=0, stop=(num_iterations - 1), num=(num_checkpoints + 2), dtype=int)
                checkpoints = set(checkpoints[1:-1])  # Don't create a checkpoint at the beginning and end

        if num_evaluations is not None:
            evaluation_points = np.linspace(start=0, stop=(num_iterations - 1), num=num_evaluations, dtype=int)
            evaluation_points = set(evaluation_points)
            self.logger["scores"] = {"train": [], "test": []}
            env_eval_train = ProcgenGym3Env(  # Environment to evaluate the agent on known levels
                num=4,
                env_name=self._env_name,
                distribution_mode=self._env_mode,
                start_level=0,
                num_levels=num_levels,
            )
            env_eval_test = ProcgenGym3Env(  # Environment to evaluate the agent on unseen levels
                num=4,
                env_name=self._env_name,
                distribution_mode=self._env_mode,
                start_level=num_levels,
                num_levels=num_levels,
            )

        # Initialize optimizer and environment for training
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=learning_rate)
        env = ProcgenGym3Env(
            num=num_envs,
            env_name=self._env_name,
            distribution_mode=self._env_mode,
            start_level=0,
            num_levels=num_levels,
        )
        env = NormalizeReward(env, gamma=gamma)

        # PPO training loop: rollout => update => rollout => ...
        self._model.train()
        for i in range(num_iterations):
            start_time = time.time()

            data = self._rollout(env, num_steps, gamma, gae_lambda)
            buffer = RolloutBuffer(data)
            batch_size = len(buffer) // num_batches
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

            # Create checkpoint
            if (num_checkpoints is not None) and (i in checkpoints):
                fname = f"{self._env_name}_{self._env_mode}_{i+1}.pt"
                torch.save(self._model.state_dict(), f"{checkpoint_dir}/{fname}")

            # Evaluate agent on train and test set: A modified approach compared to the Procgen paper
            # for computational efficiency while maintaining similar results.
            if (num_evaluations is not None) and (i in evaluation_points):
                timestep = (i + 1) * num_envs * num_steps
                train_scores = evaluate_agent(self._agent, env_eval_train, max_steps=(num_envs * num_steps // 4))
                self.logger["scores"]["train"].append((timestep, train_scores))
                test_scores = evaluate_agent(self._agent, env_eval_test, max_steps=(num_envs * num_steps // 4))
                self.logger["scores"]["test"].append((timestep, test_scores))

                runtime = time.time() - start_time
                print(f"{(i+1):{len(str(num_iterations))}}/{num_iterations}: {runtime=:.1f}s, ", end="")
                print(f"train_score: {sum(train_scores) / len(train_scores):.2f}, ", end="")
                print(f"test_score: {sum(test_scores) / len(test_scores):.2f}")
            else:
                runtime = time.time() - start_time
                print(f"{(i+1):{len(str(num_iterations))}}/{num_iterations}: {runtime=:.1f}s")

        # Save final model after last iteration
        fname = f"{self._env_name}_{self._env_mode}_final.pt"
        torch.save(self._model.state_dict(), f"{checkpoint_dir}/{fname}")

    def _rollout(
        self, env: ProcgenGym3Env | Wrapper, num_steps: int, gamma: float, gae_lambda: float
    ) -> dict[str, torch.Tensor]:
        """Perform a rollout in the environment to collect training data.

        Args:
            env (ProcgenGym3Env | Wrapper): The environment to interact with.
            num_steps (int): Number of steps per rollout.
            gamma (float): Discount factor for rewards.
            gae_lambda (float): Lambda for Generalized Advantage Estimation (GAE).

        Returns:
            dict[str, torch.Tensor]: Collected data including observations, actions, log_probs, values, returns, and advantages.
        """
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
        """Compute discounted returns.

        Args:
            rewards (torch.Tensor): Collected rewards.
            masks (torch.Tensor): Masks indicating episode ends (0 = new episode).
            gamma (float): Discount factor for rewards.

        Returns:
            torch.Tensor: Discounted returns.
        """
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
        """Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards (torch.Tensor): Collected rewards.
            values (torch.Tensor): Value estimates.
            next_value (torch.Tensor): Value estimate for the next state.
            masks (torch.Tensor): Masks indicating episode ends (0 = new episode).
            gamma (float): Discount factor for rewards.
            gae_lambda (float): Lambda for GAE.

        Returns:
            torch.Tensor: Computed advantages.
        """
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
    """A dataset class to handle the rollout data."""

    def __init__(self, data: dict[str, torch.Tensor]) -> None:
        """Initialize the RolloutBuffer.

        Args:
            data (dict[str, torch.Tensor]): Collected rollout data.
        """
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
