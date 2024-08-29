import torch
import torch.nn as nn

from attention import SelfAttention


class CLIPLayer(nn.Module):
    def __init__(self, atn_num_heads: int, embedding_dim: int):
        super(CLIPLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attention = SelfAttention(atn_num_heads, embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.lin1 = nn.Linear(embedding_dim, 4 * embedding_dim)
        self.lin2 = nn.Linear(4 * embedding_dim, embedding_dim)

    def forward(self, x):
        # transformer architecture, look at the image
        res = x  # attention
        x = self.norm1(x)
        x = self.attention(x)
        x += res
        res = x  # feedforward
        x = self.norm2(x)
        x = self.lin1(x)
        x *= torch.sigmoid(1.702 * x)  # QuickGELU
        x = self.lin2(x)
        x += res
        return x


class CLIPEmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, max_seq_len: int):
        super(CLIPEmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.position_encoding_value = nn.Parameter(torch.zeros(max_seq_len, embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x += self.position_encoding_value
        return x


class CLIP(nn.Module):
    def __init__(self, vocab_size: int = 49408, embedding_dim: int = 768, max_seq_len: int = 77, atn_num_heads: int = 12, num_layers: int = 12):
        super(CLIP, self).__init__()
        self.embedding = CLIPEmbeddingLayer(vocab_size=vocab_size, embedding_dim=embedding_dim, max_seq_len=max_seq_len)  # seq len = 77
        self.clip_layers = [CLIPLayer(atn_num_heads, embedding_dim) for _ in range(num_layers)]
        self.clip_layers.append(
            nn.LayerNorm(embedding_dim)
        )
        self.clip = nn.Sequential(*self.clip_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.type(torch.long)  # convert to long
        state = self.embedding(x)
        state = self.clip(state)
        return state
