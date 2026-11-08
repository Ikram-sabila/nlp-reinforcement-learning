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
    
    def train(self) -> list[dict]:
        history = []
        
        for episode_number in range(1, self.config.episodes + 1):
            episode_result = self.train_episode()
            
            history.append({
                "episode": episode_number,
                "total_reward": episode_result["total_reward"],
                "loss": episode_result["loss"]
            })
            
            if episode_number % 100 == 0:
                recent_results = history[-100:]
                
                average_reward = sum(
                    result["total_reward"]
                    for result in recent_results
                ) / len(recent_results)
                
                print(
                    f"Episode: {episode_number} | "
                    f"Average Reward: {average_reward:.3f}"
                )
                
        return history
    
    