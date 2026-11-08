import torch

from nlp_rl.config import ExperimentConfig
from nlp_rl.models.policy_network import PolicyNetwork


def test_policy_network_output_shape():
    config = ExperimentConfig()

    network = PolicyNetwork(
        vocabulary_size=len(config.vocabulary),
        hidden_size=config.hidden_size,
    )

    # State awal: satu token, yaitu <BOS>.
    state = torch.tensor([[1]], dtype=torch.long)

    logits = network(state)

    assert logits.shape == (1, len(config.vocabulary))
    assert torch.isfinite(logits).all()
