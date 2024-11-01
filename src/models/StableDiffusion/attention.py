import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, embedding_dim: int, input_projection_bias: bool = True, output_projection_bias: bool = True):
        self.input_projection = nn.Linear(embedding_dim, 3 * embedding_dim, bias=input_projection_bias)  # K, Q, V matrices, can just make 3 separate layers for the matrices
        self.output_projection = nn.Linear(embedding_dim, embedding_dim, bias=output_projection_bias)
        self.n_heads = n_heads
        self.dim_head = embedding_dim // n_heads

    def forward(self, x: torch.Tensor, mask: bool = False) -> torch.Tensor:
        # as written in the Attention Is All You Need paper: https://arxiv.org/abs/1706.03762
        batch_size, seq_len, d_emb = x.shape
        temp_shape = (batch_size, seq_len, self.n_heads, self.dim_head)
        q, k, v = self.input_projection(x).chunk(3, dim=-1)  # split into 3 tensors for k, q, v matrices, this is basically a multiplication of q, k, v matrices by WQ, WK and WV

        q = q.view(temp_shape).transpose(1, 2)  # B, seq_len, num_heads, dim / num_heads -> B, num_heads, seq_len, dim / num_heads
        k = k.view(temp_shape).transpose(1, 2)
        v = v.view(temp_shape).transpose(1, 2)

        atn_mat = q @ k.transpose(-1, -2)  # Q * K transposed (B, num_heads, seq_len, dim / num_heads) @ (B, num_heads, dim / num_heads, seq_len) -> (B, num_heads, seq_len, seq_len)
        # apply causal mask if specified
        if mask:
            atn_mask = torch.ones_like(atn_mat, dtype=torch.bool).triu(diagonal=1)
            atn_mat.masked_fill_(atn_mask, float('-inf'))

        atn_mat /= torch.sqrt(self.dim_head)  # divide by sqrt(dimension) for numerical stability
        atn_mat = F.softmax(atn_mat, dim=-1)  # apply softmax
        atn_out = atn_mat @ v  # multiply by V matrix (B, num_heads, seq_len, seq_len) @ (B, num_heads, seq_len, dim / num_heads) -> (B, num_heads, seq_len, dim / num_heads)\

        atn_out = atn_out.transpose(1, 2).reshape((batch_size, seq_len, d_emb))  # back to original shape (x.shape)
        atn_out *= self.output_projection  # multiply by WO matrix
        return atn_out


class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, embedding_dim: int, cross_dim: int, input_projection_bias: bool = True, output_projection_bias: bool = True):
        super(CrossAttention, self).__init__()
        self.w_q = nn.Linear(embedding_dim, embedding_dim, bias=input_projection_bias)
        self.w_k = nn.Linear(cross_dim, embedding_dim, bias=input_projection_bias)
        self.w_v = nn.Linear(cross_dim, embedding_dim, bias=input_projection_bias)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim, bias=output_projection_bias)
        self.n_heads = n_heads
        self.dim_head = embedding_dim // n_heads

    def forward(self, x, y):
        # x - query, y - key, value
        # x - latent: B, q_seq_len, q_dim
        # y - context: B, kv_seq_len, kv_dim - kv_seq_len = 77
        batch_size, seq_len, d_emb = x.shape
        temp_shape = (batch_size, -1, self.n_heads, self.dim_head)

        q = self.w_q(x)
        k = self.w_k(y)
        v = self.w_v(y)

        q = q.view(temp_shape).transpose(1, 2)  # B, seq_len, num_heads, dim / num_heads -> B, num_heads, seq_len, dim / num_heads
        k = k.view(temp_shape).transpose(1, 2)
        v = v.view(temp_shape).transpose(1, 2)

        atn_mat = q @ k.transpose(-1, -2)
        atn_mat /= torch.sqrt(self.dim_head)  # divide by sqrt(dimension) for numerical stability
        atn_mat = F.softmax(atn_mat, dim=-1)  # apply softmax
        atn_out = atn_mat @ v  # multiply by V matrix (B, num_heads, seq_len, seq_len) @ (B, num_heads, seq_len, dim / num_heads) -> (B, num_heads, seq_len, dim / num_heads)\

        # reshape back to original size
        atn_out = atn_out.transpose(1, 2).contiguous()
        atn_out = atn_out.view((batch_size, seq_len, d_emb))
        atn_out = self.output_projection(atn_out)
        return atn_out
