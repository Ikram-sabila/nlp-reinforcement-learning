from nlp_rl.agents.policy_agent import PolicyAgent
from nlp_rl.environments.text_generation_env import (
    TextGenerationEnvironment,
)

class Evaluator:
    def __init__(
        self,
        agent: PolicyAgent,
        environment: TextGenerationEnvironment
    ) -> None:
        self.agent = agent
        self.environment = environment
        
    def evaluate(self, episodes: int = 100) -> dict:
        exact_matches = 0
        generated_texts = []
        
        for _ in range(episodes):
            state = self.environment.reset()
            done = False
            
            while not done:
                action, _ = self.agent.select_action(state)
                state, _, done = self.environment.step(action)
                
            generated_text = self.environment.generated_text
            generated_texts.append(generated_text)
            
            if generated_text == " ".join(self.environment.target_tokens):
                exact_matches += 1
        
        return {
            "episodes": episodes,
            "exact_matches": exact_matches,
            "exact_match_rate": exact_matches / episodes,
            "generated_texts": generated_texts,
        }