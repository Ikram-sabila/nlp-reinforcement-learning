import torch
import pytest

from nlp_rl.config import ExperimentConfig
from nlp_rl.rewards.token_match_reward import TokenMatchReward
from nlp_rl.environments.text_generation_env import (
    TextGenerationEnvironment,
)
from nlp_rl.models.policy_network import PolicyNetwork
from nlp_rl.agents.policy_agent import PolicyAgent
from nlp_rl.training.reinforce_trainer import REINFORCETrainer


def create_trainer() -> REINFORCETrainer:
    config = ExperimentConfig()

    reward_function = TokenMatchReward(
        target_text=config.target_text,
    )

    environment = TextGenerationEnvironment(
        config=config,
        reward_function=reward_function,
    )

    model = PolicyNetwork(
        vocabulary_size=len(config.vocabulary),
        hidden_size=config.hidden_size,
    )

    agent = PolicyAgent(
        policy_network=model,
        invalid_action_ids=[
            environment.token_to_id["<PAD>"],
            environment.bos_id,
        ],
    )

    return REINFORCETrainer(
        agent=agent,
        environment=environment,
        config=config,
    )


def test_calculate_returns() -> None:
    trainer = create_trainer()

    returns = trainer.calculate_returns(
        rewards=[1.0, -0.5, -1.0],
    )

    expected_returns = [-0.4751, -1.49, -1.0]

    assert len(returns) == len(expected_returns)

    for actual, expected in zip(returns, expected_returns):
        assert abs(actual - expected) < 1e-5


def test_collect_episode() -> None:
    torch.manual_seed(42)

    trainer = create_trainer()

    episode = trainer.collect_episode()

    assert len(episode["rewards"]) > 0

    assert len(episode["log_probabilities"]) == len(
        episode["rewards"]
    )

    assert episode["total_reward"] == sum(episode["rewards"])

    assert isinstance(episode["generated_text"], str)


def test_train_episode_updates_model_parameters() -> None:
    torch.manual_seed(42)

    trainer = create_trainer()

    parameters_before = [
        parameter.detach().clone()
        for parameter in trainer.agent.policy_network.parameters()
    ]

    episode = trainer.train_episode()

    parameters_after = list(
        trainer.agent.policy_network.parameters()
    )

    assert "loss" in episode
    assert torch.isfinite(torch.tensor(episode["loss"]))

    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(parameters_before, parameters_after)
    )
    
def test_calculate_baseline() -> None:
    trainer = create_trainer()

    baseline = trainer.calculate_baseline(
        batch_returns=[
            [2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0],
        ]
    )

    assert abs(baseline - 2.5) < 1e-6
    
def test_calculate_baseline_rejects_empty_returns() -> None:
    trainer = create_trainer()

    with pytest.raises(ValueError):
        trainer.calculate_baseline([])
        
def test_train_batch_updates_model_parameters() -> None:
    torch.manual_seed(42)

    trainer = create_trainer()

    parameters_before = [
        parameter.detach().clone()
        for parameter in trainer.agent.policy_network.parameters()
    ]

    result = trainer.train_batch(batch_size=2)

    parameters_after = list(
        trainer.agent.policy_network.parameters()
    )

    assert result["batch_size"] == 2
    assert len(result["episodes"]) == 2
    assert "baseline" in result
    assert "loss" in result

    assert torch.isfinite(torch.tensor(result["loss"]))

    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(parameters_before, parameters_after)
    )