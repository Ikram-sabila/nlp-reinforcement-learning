from nlp_rl.config import ExperimentConfig
from nlp_rl.rewards.token_match_reward import TokenMatchReward

class TextGenerationEnvironment:
    def __init__(
        self,
        config: ExperimentConfig,
        reward_function: TokenMatchReward
    ) -> None:
        self.config = config
        self.reward_function = reward_function
        
        self.token_to_id = {
            token: index
            for index, token in enumerate(config.vocabulary)    
        }        
        self.id_to_token = {
            index: token
            for token, index in self.token_to_id.items()
        }
        
        self.bos_id = self.token_to_id["<BOS>"]
        self.eos_id = self.token_to_id["<EOS>"]

        self.target_tokens = config.target_text.split()

        self.generated_token_ids: list[int] = []
        self.current_step = 0
        
    def reset(self) -> list[int]:
        """Start a new episode"""
        self.generated_token_ids = []
        self.current_step = 0
        
        return [self.bos_id]
    
    def step(self, action: int) -> tuple[list[int], float, bool]:
        """Apply one token action.

        Returns:
            next_state, reward, done
        """ 
        
        if self.current_step >= self.config.max_steps:
            raise RuntimeError("Episode has already finished.")
        
        if not 0 <= action < len(self.config.vocabulary):
            raise ValueError(f"Invalid action ID: {action}")
        
        token = self.id_to_token[action]
        
        if token in {"<PAD>", "<BOS>"}:
            raise ValueError(
                f"Special token {token} cannot be generated as an action."
            )

        reward = 0.0
        
        if action == self.eos_id:
            done = True
            reward = self.reward_function.calculate_terminal(
                self.generated_tokens 
            )
            return (
                [self.bos_id] + self.generated_token_ids + [self.eos_id],
                reward,
                done
            )
            
        position = len(self.generated_token_ids)
        self.generated_token_ids.append(action)
        self.current_step += 1
        
        reward += self.reward_function.calculate_step(
            token=token,
            position=position
        )
        
        done = self.current_step >= self.config.max_steps
        
        if done:
            reward += self.reward_function.calculate_terminal(
                self.generated_tokens
            )
        
        next_state = [self.bos_id] + self.generated_token_ids
        
        return next_state, reward, done
    
    @property
    def generated_tokens(self) -> list[str]:
        """Return generated text tokens without special tokens."""
        return [
            self.id_to_token[token_id]
            for token_id in self.generated_token_ids
        ]

    @property
    def generated_text(self) -> str:
        """Return the current generated text."""
        return " ".join(self.generated_tokens)
        