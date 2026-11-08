import torch
from collections import Counter

from nlp_rl.config import ExperimentConfig
from nlp_rl.rewards.token_match_reward import TokenMatchReward
from nlp_rl.environments.text_generation_env import (
    TextGenerationEnvironment,
)
from nlp_rl.models.policy_network import PolicyNetwork
from nlp_rl.agents.policy_agent import PolicyAgent
from nlp_rl.training.reinforce_trainer import REINFORCETrainer
from nlp_rl.evaluations.evaluator import Evaluator


def main() -> None:
    config = ExperimentConfig()

    torch.manual_seed(config.random_seed)

    reward_function = TokenMatchReward(
        target_text=config.target_text,
    )

    environment = TextGenerationEnvironment(
        config=config,
        reward_function=reward_function,
    )

    policy_network = PolicyNetwork(
        vocabulary_size=len(config.vocabulary),
        hidden_size=config.hidden_size,
    )

    agent = PolicyAgent(
        policy_network=policy_network,
        invalid_action_ids=[
            environment.token_to_id["<PAD>"],
            environment.bos_id
        ]
    )

    trainer = REINFORCETrainer(
        agent=agent,
        environment=environment,
        config=config,
    )

    print("Starting REINFORCE training...")

    history = trainer.train()

    print("\nTraining completed!")
    print(f"Total episodes: {len(history)}")

    evaluator = Evaluator(
        agent=agent,
        environment=environment,
    )

    results = evaluator.evaluate(episodes=100)
    print("\nGenerated Text Samples:")
    
    for text in results["generated_texts"][:10]:
        print(f"- '{text}'")
        
    print("\nGenerated Text Distribution:")

    for text, count in Counter(results["generated_texts"]).most_common(10):
        print(f"'{text}': {count}")

    print("\nEvaluation Results")
    print(f"Target text: {config.target_text}")
    print(f"Exact matches: {results['exact_matches']}")
    print(
        f"Exact match rate: "
        f"{results['exact_match_rate']:.2%}"
    )


if __name__ == "__main__":
    main()