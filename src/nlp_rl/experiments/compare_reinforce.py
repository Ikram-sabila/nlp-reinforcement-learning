import copy

import torch

from nlp_rl.agents.policy_agent import PolicyAgent
from nlp_rl.config import ExperimentConfig
from nlp_rl.environments.text_generation_env import TextGenerationEnvironment
from nlp_rl.evaluations.evaluator import Evaluator
from nlp_rl.models.policy_network import PolicyNetwork
from nlp_rl.rewards.token_match_reward import TokenMatchReward
from nlp_rl.training.reinforce_trainer import REINFORCETrainer

class REINFORCEComparison:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
        torch.manual_seed(config.random_seed)
        
        initial_model = PolicyNetwork(
            vocabulary_size=len(config.vocabulary),
            hidden_size=config.hidden_size
        )
        
        self.initial_weights = copy.deepcopy(initial_model.state_dict())
        
    def create_trainer(self) -> tuple[REINFORCETrainer, Evaluator]:
        reward_function = TokenMatchReward(
            target_text=self.config.target_text 
        )
        
        environment = TextGenerationEnvironment(
            config=self.config,
            reward_function=reward_function
        )
        
        policy_network = PolicyNetwork(
            vocabulary_size=len(self.config.vocabulary),
            hidden_size=self.config.hidden_size,
        )

        policy_network.load_state_dict(self.initial_weights)

        agent = PolicyAgent(
            policy_network=policy_network,
            invalid_action_ids=[
                environment.token_to_id["<PAD>"],
                environment.bos_id,
            ],
        )

        trainer = REINFORCETrainer(
            agent=agent,
            environment=environment,
            config=self.config,
        )

        evaluator = Evaluator(
            agent=agent,
            environment=environment,
        )

        return trainer, evaluator
    
    def run_experiment(self, name: str, use_baseline: bool) -> dict:
        print(f"\n{'=' * 50}")
        print(f"Training: {name}")
        print(f"{'=' * 50}")

        trainer, evaluator = self.create_trainer()

        torch.manual_seed(self.config.random_seed + 1)

        history = trainer.train(use_baseline=use_baseline)

        torch.manual_seed(self.config.random_seed + 2)

        evaluation = evaluator.evaluate(episodes=100)

        print(f"\n{name} Results")
        print(f"Training episodes: {len(history)}")
        print(f"Exact matches: {evaluation['exact_matches']}")
        print(
            f"Exact match rate: "
            f"{evaluation['exact_match_rate']:.2%}"
        )

        return {
            "name": name,
            "history": history,
            "evaluation": evaluation,
        }

    def compare(self) -> dict:
        vanilla_result = self.run_experiment(
            name="REINFORCE",
            use_baseline=False,
        )

        baseline_result = self.run_experiment(
            name="REINFORCE with Baseline",
            use_baseline=True,
        )

        print(f"\n{'=' * 50}")
        print("FINAL COMPARISON")
        print(f"{'=' * 50}")

        print(
            f"REINFORCE: "
            f"{vanilla_result['evaluation']['exact_match_rate']:.2%}"
        )

        print(
            f"REINFORCE with Baseline: "
            f"{baseline_result['evaluation']['exact_match_rate']:.2%}"
        )

        return {
            "reinforce": vanilla_result,
            "reinforce_with_baseline": baseline_result,
        }
