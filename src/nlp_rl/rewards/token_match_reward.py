class TokenMatchReward:
    """Calculate reward for generated token"""
    
    def __init__(
        self,
        target_text: str,
        correct_token_reward: float = 1.0,
        incorrect_token_penalty: float = -0.5,
        exact_match_reward: float = 3.0,
        incomplete_sequence_penalty: float = -1.0
    ) -> None:
        self.target_token = target_text.split()
        self.correct_token_reward = correct_token_reward
        self.incorrect_token_penalty = incorrect_token_penalty
        self.exact_match_reward = exact_match_reward
        self.incomplete_sequence_penalty = (
            incomplete_sequence_penalty
        )
        
    def calculate_step(
        self,
        token: str,
        position: int,
    ) -> float:
        """Reward a token according to its target position."""
        if position >= len(self.target_token):
            return self.incorrect_token_penalty
        
        expected_token = self.target_token[position]
        
        if token == expected_token:
            return self.correct_token_reward
        
        return self.incorrect_token_penalty
    
    def calculate_terminal(
        self,
        generated_token: list[str]
    ) -> float:
        """Reward completion of the target sequence."""
        if generated_token == self.target_token:
            return self.exact_match_reward

        return self.incomplete_sequence_penalty