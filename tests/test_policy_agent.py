import torch

from nlp_rl.agents.policy_agent import PolicyAgent
from nlp_rl.models.policy_network import PolicyNetwork


def test_select_action_returns_valid_action() -> None:
    torch.manual_seed(42)

    vocabulary_size = 9

    model = PolicyNetwork(
        vocabulary_size=vocabulary_size,
    )

    agent = PolicyAgent(
        policy_network=model,
        invalid_action_ids=[0, 1],
    )

    action, log_probability = agent.select_action([1])

    assert 0 <= action < vocabulary_size
    assert action not in [0, 1]
    assert isinstance(log_probability, torch.Tensor)
    assert log_probability.ndim == 0
    assert torch.isfinite(log_probability)


def test_log_probability_keeps_gradient_connection() -> None:
    model = PolicyNetwork(vocabulary_size=9)

    agent = PolicyAgent(
        policy_network=model,
        invalid_action_ids=[0, 1],
    )

    _, log_probability = agent.select_action([1, 3])

    assert log_probability.requires_grad
    assert log_probability.grad_fn is not None