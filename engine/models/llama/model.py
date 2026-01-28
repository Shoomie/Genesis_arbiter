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
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
        
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        
        # SDPA expects [bsz, n_heads, seqlen, head_dim]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # Use scaled_dot_product_attention for FlashAttention-2 / Memory Efficient Attention
        # is_causal should be True for the decoder mask
        output = F.scaled_dot_product_attention(
            xq, xk, xv, 
            attn_mask=None, 
            dropout_p=0.0, 
            is_causal=True
        )
        
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
    def __init__(self, layer_id, dim, n_heads, intermediate_size):
        super().__init__()
        self.layer_id = layer_id
        self.attention = Attention(dim, n_heads)
        self.feed_forward = FeedForward(dim, intermediate_size)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x, freqs_cis):
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Llama(nn.Module):
    def __init__(self, vocab_size=8000, n_layers=12, dim=768, n_heads=12, intermediate_size=3072, max_seq_len=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(i, dim, n_heads, intermediate_size)
            for i in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(dim // n_heads, max_seq_len * 2)
        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

    def forward(self, tokens, labels=None):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[:seqlen]
        
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                h = checkpoint(layer, h, freqs_cis, use_reentrant=False)
            else:
                h = layer(h, freqs_cis)
        
        h = self.norm(h)
        logits = self.output(h)
        
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return logits, loss
            
        return logits, None

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

def logos_init_hook(model, jehovah_token_id=5, multiplier=1.0):
    """
    Logos Initialization Hook: 
    Sets the initial weight variance of 'Jehovah' token to current baseline scaled by multiplier.
    """
    import math
    with torch.no_grad():
        emb = model.tok_embeddings.weight
        std = emb.std().item()
        # Scale factor: sqrt(multiplier) to get linear variance boost
        scale = math.sqrt(multiplier)
        new_weights = torch.randn(emb.shape[1], device=emb.device) * std * scale
        emb[jehovah_token_id] = new_weights
        print(f"Logos initialization applied to token ID {jehovah_token_id} with multiplier {multiplier}x (scale {scale:.4f})")
