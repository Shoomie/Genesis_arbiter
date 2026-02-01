import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class DeepNorm(nn.Module):
    """DeepNorm from 'DeepNet: Scaling Transformers to 1000 Layers'
    
    For deep networks (80-144 layers), DeepNorm provides better gradient flow
    by combining residual scaling with LayerNorm.
    
    Formula: DeepNorm(x, residual) = LayerNorm(alpha * x + residual)
    where alpha = (2 * N)^0.25 and N is the number of layers.
    """
    def __init__(self, dim: int, n_layers: int, eps: float = 1e-5):
        super().__init__()
        self.alpha = (2 * n_layers) ** 0.25
        self.norm = nn.LayerNorm(dim, eps=eps)
    
    def forward(self, x, residual):
        """Apply DeepNorm: LayerNorm(alpha * x + residual)"""
        return self.norm(self.alpha * x + residual)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, xq_.shape[1], 1, xq_.shape[-1])
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x, freqs_cis):
        bsz, seqlen, _ = x.shape
        
        # QKV projections
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        
        # Reshape for multi-head attention
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
        
        # Apply RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        
        # Transpose to (batch, n_heads, seq_len, head_dim) for SDPA
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # Use PyTorch's scaled_dot_product_attention (automatically uses FlashAttention if available)
        # This provides 2-4x speedup on character-level sequences with minimal code changes
        output = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=None,
            dropout_p=0.0,  # We don't use attention dropout in this architecture
            is_causal=True   # Enables causal masking for autoregressive generation
        )
        
        # Transpose back and reshape
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, dim, intermediate_size):
        super().__init__()
        self.w1 = nn.Linear(dim, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, dim, bias=False)
        self.w3 = nn.Linear(dim, intermediate_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, layer_id, dim, n_heads, intermediate_size, norm_type="rmsnorm", n_layers=None):
        super().__init__()
        self.layer_id = layer_id
        self.norm_type = norm_type
        self.attention = Attention(dim, n_heads)
        self.feed_forward = FeedForward(dim, intermediate_size)
        
        # Select normalization type
        if norm_type == "deepnorm":
            assert n_layers is not None, "n_layers required for DeepNorm"
            self.attention_norm = DeepNorm(dim, n_layers)
            self.ffn_norm = DeepNorm(dim, n_layers)
        else:  # rmsnorm (default)
            self.attention_norm = RMSNorm(dim)
            self.ffn_norm = RMSNorm(dim)

    def forward(self, x, freqs_cis):
        if self.norm_type == "deepnorm":
            # DeepNorm: norm(alpha * residual + branch_output)
            attn_out = self.attention(x, freqs_cis)
            h = self.attention_norm(x, attn_out)  # x is residual, attn_out is branch
            
            ffn_out = self.feed_forward(h)
            out = self.ffn_norm(h, ffn_out)
        else:  # rmsnorm (standard post-norm)
            h = x + self.attention(self.attention_norm(x), freqs_cis)
            out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Llama(nn.Module):
    def __init__(self, vocab_size=8000, n_layers=12, dim=768, n_heads=12, 
                 intermediate_size=3072, max_seq_len=1024, norm_type="rmsnorm"):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dim = dim
        self.n_heads = n_heads
        self.norm_type = norm_type
        
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(i, dim, n_heads, intermediate_size, norm_type, n_layers)
            for i in range(n_layers)
        ])
        
        # Final norm (always RMSNorm, even for DeepNorm models)
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.register_buffer("freqs_cis", precompute_freqs_cis(dim // n_heads, max_seq_len * 2), persistent=False)
        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

    def forward(self, tokens, labels=None, return_hiddens=False):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[:seqlen]
        
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                h = checkpoint(layer, h, freqs_cis, use_reentrant=False)
            else:
                h = layer(h, freqs_cis)
        
        h = self.norm(h)
        
        if return_hiddens:
            return h, None
            
        logits = self.output(h)
        
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return logits, loss
            
        return logits, None

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
