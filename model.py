from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_length: int = 2048

    device: str = None

"""Class for RMS Normalization."""
class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization focuses on re-scaling invariance and regularizes the summed inputs simply according to Root Mean Square [1].
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        """
        Args:
        eps -> Epsilon to avoid Division Error.
        dim -> Embedding Dimension.

        """
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (Batch_size, Seq_len, dim) / (Batch_size, Seq_len, 1) -> (Batch_size, Seq_len, dim)
        norm = (x / (torch.sqrt(x.pow(2).mean(-1, keepdim = True)) + self.eps)).type_as(x)
        return self.weight * norm
    
"""________________________________________________________________________________________________________________________________________________________________"""

"""Function to pre-compute the Theta for Rotary Frequencies."""
    
def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0) -> torch.Tensor:
    """
    Computing Thetha_i = 10000 ** -2(i-1)/d, i = [1,2,...,d/2] [2].

    Args:
    head_dim -> Head Dim of Self-Attention.
    seq_len -> Maximum sequence length.
    device -> Device to compute.
    theta -> Fixed theta.

    """
    assert head_dim % 2 == 0, "Head Dimension must be divisble by 2"

    # (Head_dim / 2)
    theta_pow = torch.arange(0, head_dim, 2).float()
    # (Head_dim / 2)
    theta_i = theta ** (-theta_pow / head_dim).to(device)
    # (Seq_len)
    seq_pos = torch.arange(seq_len).to(device)
    # (Seq_length) outer_product(theta_i) -> (Seq_length, Head_dim / 2)
    freqs = torch.outer(seq_pos, theta_i).float()
    # (Seq_length, Head_dim / 2) -> (Seq_length, Head_dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_complex

"""________________________________________________________________________________________________________________________________________________________________"""

"""Function to compute Rotary Embeddings to each tokens."""

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str) -> torch.Tensor:
   
   """
   Rotary positional Encodings[RPoE] is a relative positional encoding applied between two tokens, which indicates the intensity of relationship between them, in terms of Distance parameter [2].
   RPoE are only applied to the Query and the Keys, but not the Values. It is applied after the vector q and k are multiplied with respective
   W matrices in the attention mechanism.
   
   Args:
   x -> Input Tokens Tensor.
   freqs_complex -> Pre-computed frequencies in Polar domain.
   device -> Device to compute.

   """
   # (Batch_size, Seq_len, Head, Head_dim) -> (Batch_size, Seq_len, Head, Head_dim / 2)
   x_complex =  torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
   # (Seq_len, Head_dim / 2) -> (1, Seq_len, 1, Head_dim / 2)
   freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
   # (Batch_size, Seq_len, Head, Head_dim / 2) * (1, Seq_len, 1, Head_dim / 2) -> (Batch_size, Seq_len, Head, Head_dim / 2)
   x_rotated = x_complex * freqs_complex
   # (Batch_size, Seq_len, Head, Head_dim / 2) -> (Batch_size, Seq_len, Head, Head_dim / 2, 2)
   x_rotated = torch.view_as_real(x_rotated)
   # (Batch_size, Seq_len, Head, Head_dim / 2) -> (Batch_size, Seq_len, Head, Head_dim)
   x_out = x_rotated.reshape(*x.shape)

   return x_out.type_as(x).to(device)

"""________________________________________________________________________________________________________________________________________________________________"""

"""Class for FeedForward Module."""
class FeedForward(nn.Module):

    """
    The FeedForward Neural Network similar to Vanilla Transformer, but instead ReLU activation, it uses SwiGLU Activation Function [3].
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        """
        Args:
        args -> ModelArgs.

        """

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3) #[5]
        if args.ffn_dim_multiplier is not None:
            hidden_dim = args.ffn_dim_multiplier * hidden_dim
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias = False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias = False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias = False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (Batch_size, Seq_len, dim) -> (Batch_size, Seq_len, Hidden_dim)
        x_W = self.w1(x)
        # (Batch_size, Seq_len, dim) -> (Batch_size, Seq_len, Hidden_dim)
        x_V = self.w3(x)
        # (Batch_size, Seq_len, Hidden_dim) * (Batch_size, Seq_len, Hidden_dim) -> (Batch_size, Seq_len, Hidden_dim)
        out = F.silu(x_W) * x_V
        # (Batch_size, Seq_len, Hidden_dim) -> (Batch_size, Seq_len, dim)
        out = self.w2(out)

        return out
    
"""________________________________________________________________________________________________________________________________________________________________"""
    
"""Function to create Heads for Keys and Values based on Query Heads."""
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_heads_kv, head_dim = x.shape

    return (x[:, :, :, None, :].expand(batch_size, seq_len, n_heads_kv, n_rep, head_dim)\
                               .reshape(batch_size, seq_len, n_heads_kv * n_rep, head_dim)
           )   

"""Class to compute Self-Attention."""
class SelfAttention(nn.Module):

    """
    Self-Attention which employs Grouped Multi Query Attention that provides the good compromise between Quality and Speed. The main objective of Grouped Multi Query
    Attention is to minimize the memory access/transfer in the GPU [4]. 
    For Inference, attention mechanism uses KV-Cache Techinque. At every step of the inference, we are only interested in the last token output by the model, because 
    we already have previous tokens. However, the model needs access to all the previous tokens to decide on which token to output, since it constitute its context. 
    This KV cache is a solution to make the model do less computation on the token it has already seen during inference.
    
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        """
        Args:
        args -> ModelArgs.

        """

        self.n_heads_kv = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_heads_kv
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, self.n_heads_q * self.head_dim, bias = False)
        self.wk = nn.Linear(args.dim, self.n_heads_kv * self.head_dim, bias = False)
        self.wv = nn.Linear(args.dim, self.n_heads_kv * self.head_dim, bias = False)

        self.wo = nn.Linear(args.dim, args.dim, bias = False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_length, self.n_heads_kv, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_length, self.n_heads_kv, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor) -> torch.Tensor:
        # (Batch_size, 1, dim)
        batch_size, seq_len, _ = x.shape

        # (Batch_size, 1, dim) -> (Batch_size, 1, n_heads_q * head_dim)
        q = self.wq(x)
        # (Batch_size, 1, dim) -> (Batch_size, 1, n_heads_kv * head_dim)
        k = self.wk(x)
        # (Batch_size, 1, dim) -> (Batch_size, 1, n_heads_kv * head_dim)
        v = self.wv(x)

        # (Batch_size, 1, n_heads_q * head_dim) -> (Batch_size, 1, n_heads_q, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (Batch_size, 1, n_heads_kv * head_dim) -> (Batch_size, 1, n_heads_kv, head_dim)
        k = k.view(batch_size, seq_len, self.n_heads_kv, self.head_dim)
        # (Batch_size, 1, n_heads_kv * head_dim) -> (Batch_size, 1, n_heads_kv, head_dim)
        v = v.view(batch_size, seq_len, self.n_heads_kv, self.head_dim)

        # (Batch_size, Seq_len, Head, Head_dim) -> (Batch_size, 1, n_heads_q, head_dim)
        q = apply_rotary_embeddings(q, freqs_complex, x.device)
        # (Batch_size, Seq_len, Head, Head_dim) -> (Batch_size, 1, n_heads_kv, head_dim)
        k = apply_rotary_embeddings(k, freqs_complex, x.device)

        self.cache_k[: batch_size, start_pos : start_pos + seq_len] = k
        self.cache_v[: batch_size, start_pos : start_pos + seq_len] = v

        # (Batch_size, Seq_len_kv, n_heads_kv, head_dim)
        keys = self.cache_k[: batch_size, : start_pos + seq_len]
        # (Batch_size, Seq_len_kv, n_heads_kv, head_dim)
        values = self.cache_v[: batch_size, : start_pos + seq_len]

        # (Batch_size, Seq_len_kv, n_heads_kv, head_dim) -> (Batch_size, Seq_len_kv, n_heads_q, head_dim)
        keys = repeat_kv(keys, self.n_rep)
        # (Batch_size, Seq_len_kv, n_heads_kv, head_dim) -> (Batch_size, Seq_len_kv, n_heads_q, head_dim)
        values = repeat_kv(values, self.n_rep)

        # (Batch_size, 1, n_heads_q, head_dim) -> (Batch_size, n_heads_q, 1, head_dim)
        q = q.transpose(1, 2)
        # (Batch_size, Seq_len_kv, n_heads_kv, head_dim) -> (Batch_size, n_heads_q, Seq_len_kv, head_dim)
        keys = keys.transpose(1, 2)
        # (Batch_size, Seq_len_kv, n_heads_kv, head_dim) -> (Batch_size, n_heads_q, Seq_len_kv, head_dim)
        values = values.transpose(1, 2)

        # (Batch_size, n_heads_q, 1, head_dim) * (Batch_size, n_heads_q, head_dim, Seq_len_kv) -> (Batch_size, n_heads_q, 1, Seq_len_kv)
        attn_scores = q @ keys.transpose(2, 3) / math.sqrt(self.head_dim)
        attn_scores = F.softmax(attn_scores, dim = -1)

        # (Batch_size, n_heads_q, 1, Seq_len_kv) * (Batch_size, n_heads_q, Seq_len_kv, head_dim) -> (Batch_size, n_heads_q, 1, head_dim)
        attn_values = attn_scores @ values

        # (Batch_size, n_heads_q, 1, head_dim) -> (Batch_size, 1, dim)
        attn_values = attn_values.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads_q * self.head_dim)

        return self.wo(attn_values) # (Batch_size, 1, dim) -> (Batch_size, 1, dim)
    
"""________________________________________________________________________________________________________________________________________________________________"""

"""Class to create Encoder Block."""
class EncoderBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        """
        Args:
        args -> ModelArgs.

        """
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        self.attention_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: torch.Tensor, freqs_complex: torch.Tensor) -> torch.Tensor:

        residue = x
        out = self.attention_norm(x)
        out = residue = self.attention(out, start_pos, freqs_complex) + residue
        out = self.ffn_norm(out)
        out = self.feed_forward(out) + residue

        return out

"""________________________________________________________________________________________________________________________________________________________________"""

"""Class to create Transformer."""
class Transformer(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        """
        Args:
        args -> ModelArgs.

        """
        assert args.vocab_size != -1, "Vocab size must be set."
        
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias = False)

        self.head_dim = args.dim // args.n_heads
        self.freqs_complex = precompute_theta_pos_frequencies(self.head_dim, args.max_seq_length * 2, device = args.device)

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        
        batch_size, seq_len = x.shape
        assert seq_len == 1, "Only one token is processed."
        out = self.tok_embeddings(x)

        freqs_complex = self.freqs_complex[start_pos : start_pos + seq_len]

        for layer in self.layers:
            out = layer(out, start_pos, freqs_complex)
        
        out = self.norm(out)
        out = self.output(out).float()

        return out

"""
References:
1. https://arxiv.org/abs/1910.07467
2. https://arxiv.org/abs/2104.09864
3. https://arxiv.org/abs/2002.05202
4. https://arxiv.org/abs/2305.13245 
5. https://arxiv.org/abs/2302.13971 

"""