import torch
from torch import nn

class PolicyNetwork(nn.Module):
    """Neural network that scores possible net tokens"""
    
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int = 32,
        hidden_size: int = 64
    ) -> None:
        super().__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dim
        )
        
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,   
        )
        
        self.output_layer = nn.Linear(
            in_features=hidden_size,
            out_features=vocabulary_size
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(state)
        
        _, hidden_state = self.gru(embeddings)
        
        logits = self.output_layer(hidden_state[-1])
        
        return logits