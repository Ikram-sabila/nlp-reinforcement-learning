import torch
from torch.distributions import Categorical

from nlp_rl.models.policy_network import PolicyNetwork

class PolicyAgent:
    """Agent that select tokens using a policy network"""
    
    def __init__(
        self,
        policy_network: PolicyNetwork,
        invalid_action_ids: list[int] | None = None
    ) -> None:
        self.policy_network = policy_network
        self.invalid_action_ids = invalid_action_ids or []
        
    def select_action(
        self, 
        state: list[int]
    ) -> tuple[int, torch.Tensor]:
        state_tensor = torch.tensor(
            state,
            dtype=torch.long
        ).unsqueeze(0)
        
        logits = self.policy_network(state_tensor)
        masked_logits = logits.clone()
        
        if self.invalid_action_ids:
            masked_logits[:, self.invalid_action_ids] = -torch.inf
                
        distribution = Categorical(logits=masked_logits)
        
        action = distribution.sample()
        log_probability = distribution.log_prob(action)

        return action.item(), log_probability.squeeze()