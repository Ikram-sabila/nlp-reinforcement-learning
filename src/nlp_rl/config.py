from dataclasses import dataclass

@dataclass(frozen=True)
class ExperimentConfig:
    """Central configuration for the NLP RL experiment"""
    
    target_text: str = "i like cats"
    
    vocabulary: tuple[str, ...] = (
        "<PAD>",
        "<BOS>",
        "<EOS>",
        "i",
        "like",
        "cats",
        "dogs",
        "love",
        "hate",
    )
    
    max_steps: int = 5

    learning_rate: float = 0.001
    discount_factor: float = 0.99
    episodes: int = 2_000

    hidden_size: int = 64
    random_seed: int = 42
    
    def __post_init__(self) -> None:
        if not self.target_text.strip():
            raise ValueError("target_text cannot be empty")
        
        if len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError("Vocabulary must contain unique tokens.")

        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        
        if not 0.0 <= self.discount_factor <= 1.0:
            raise ValueError(
                "discount_factor must be between 0 and 1."
            )

        if self.episodes <= 0:
            raise ValueError("episodes must be positive.")

        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        
        tokens = self.target_text.split()
        
        if not set(tokens).issubset(set(self.vocabulary)):
            raise ValueError("All target tokens must exist in the vocabulary.")