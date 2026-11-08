import torch
from torch import nn

from nlp_rl.agents.policy_agent import PolicyAgent
from nlp_rl.environments.text_generation_env import (
    TextGenerationEnvironment,
)
from nlp_rl.config import ExperimentConfig

class REINFORCETrainer:
    """Collect an episode of experience from the environment."""
    
    def __init__(
        self,
        agent: PolicyAgent,
        environment: TextGenerationEnvironment,
        config: ExperimentConfig
    ) -> None:
        self.agent = agent
        self.environment = environment
        self.config = config
        self.optimizer = torch.optim.Adam(
            self.agent.policy_network.parameters(),
            lr=self.config.learning_rate
        )
        
    def collect_episode(self) -> dict:
        state = self.environment.reset()
        
        log_probabilities = []
        rewards = []
                
        done = False
        
        while not done:
            action, log_probability = self.agent.select_action(state)
            
            next_state, reward, done = self.environment.step(action)
            
            log_probabilities.append(log_probability)
            rewards.append(reward)
            
            state = next_state
        
        return {
            "generated_tokens": self.environment.generated_tokens.copy(),
            "generated_text": self.environment.generated_text,
            "rewards": rewards,
            "log_probabilities": log_probabilities,
            "total_reward": sum(rewards)
        }
        
    def calculate_returns(
        self,
        rewards: list[float]
    ) -> list[float]:
        returns = []
        dicscounted_return = 0.0
        
        for reward in reversed(rewards):
            dicscounted_return = (
                reward
                + self.config.discount_factor * dicscounted_return
            )
            
            returns.insert(0, dicscounted_return)
            
        return returns
        
    def train_episode(self) -> dict:
        episode = self.collect_episode()
        
        rewards = episode["rewards"]
        log_probabilities = episode["log_probabilities"]
        
        returns = self.calculate_returns(rewards)
        
        if len(log_probabilities) != len(returns):
            raise ValueError(
                "The number of log probabilities and return "
                "must match."
            )
        
        policy_losses = []
        
        for log_probability, return_value in zip(
            log_probabilities,
            returns
        ):
            policy_losses.append(
                -log_probability * return_value
            )
        
        loss = torch.stack(policy_losses).sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        episode["loss"] = loss.item()
        episode["returns"] = returns

        return episode
    
    def train(self, use_baseline: bool = True) -> list[dict]:
        history = []
        episodes_trained = 0
        
        while episodes_trained < self.config.episodes:
            current_batch_size = min(
                self.config.batch_size,
                self.config.episodes - episodes_trained
            )
            
            batch_result = self.train_batch(
                batch_size=current_batch_size,
                use_baseline=use_baseline    
            )
            
            episodes_trained += current_batch_size
            
            history.append({
                "episode": episodes_trained,
                "total_reward": batch_result["average_reward"],
                "loss": batch_result["loss"],
                "baseline": batch_result["baseline"],
                "batch_size": batch_result["batch_size"],
                "baseline": batch_result["baseline"]
            })
        
            if episodes_trained % 100 == 0 or (
                episodes_trained == self.config.episodes
            ):
                print(
                    f"Episode: {episodes_trained} | "
                    f"Average Reward: {batch_result['average_reward']:.3f} | "
                    f"Baseline: {batch_result['baseline']:.3f} | "
                    f"Loss: {batch_result['loss']:.4f}"
                )

        return history
    
    def calculate_baseline(
        self,
        batch_returns: list[list[float]]
    ) -> float:
        all_returns = [
            return_value
            for episode_returns in batch_returns
            for return_value in episode_returns
        ]
        
        if not all_returns:
            raise ValueError("Cannot calculate baseline from empty returns.")

        return sum(all_returns) / len(all_returns)
    
    def train_batch(
        self, 
        batch_size: int | None = None,
        use_baseline: bool = True
    ) -> dict:
        if batch_size is None:
            batch_size = self.config.batch_size
            
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        episodes = [
            self.collect_episode()
            for _ in range(batch_size)
        ]
        
        batch_returns = []

        for episode in episodes:
            returns = self.calculate_returns(episode["rewards"])

            if len(episode["log_probabilities"]) != len(returns):
                raise ValueError(
                    "The number of log probabilities and returns must match."
                )

            episode["returns"] = returns
            batch_returns.append(returns)

        baseline = (
            self.calculate_baseline(batch_returns)
            if use_baseline
            else 0.0
        )
        
        policy_losses = []
        
        for episode in episodes:
            for log_probability, return_value in zip(
                episode["log_probabilities"],
                episode["returns"]
            ):
                advantage = return_value - baseline
                
                policy_losses.append(
                    -log_probability * advantage
                )
                
        loss = torch.stack(policy_losses).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        average_reward = sum(
            episode["total_reward"]
            for episode in episodes
        ) / len(episodes)
        
        return {
            "episodes": episodes,
            "loss": loss.item(),
            "baseline": baseline,
            "average_reward": average_reward,
            "batch_size": batch_size
        }
    