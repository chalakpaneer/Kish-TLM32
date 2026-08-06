"""
TLM32 model definition in PyTorch.

IMPORTANT: every weight tensor here is shaped [in_dim, out_dim] on purpose
(NOT PyTorch's usual nn.Linear convention of [out_dim, in_dim]). That's so
export_weights.py can dump these tensors straight to bytes with zero
transposing and have them land exactly where tlm_model.c's tlm_matmul()
expects them: out[o] = sum_i x[i] * w[i*out_dim + o].

If you change any dimension here, update tlm_config.h to match, or the
exported blob will silently be read at the wrong offsets on-device.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 256
CONTEXT_LEN = 64
EMBED_DIM = 32
N_HEADS = 2
HEAD_DIM = EMBED_DIM // N_HEADS
N_LAYERS = 2
FFN_HIDDEN = 64
LN_EPS = 1e-5


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1_w = nn.Parameter(torch.ones(EMBED_DIM))
        self.ln1_b = nn.Parameter(torch.zeros(EMBED_DIM))
        self.wq = nn.Parameter(torch.randn(EMBED_DIM, EMBED_DIM) * 0.02)
        self.wk = nn.Parameter(torch.randn(EMBED_DIM, EMBED_DIM) * 0.02)
        self.wv = nn.Parameter(torch.randn(EMBED_DIM, EMBED_DIM) * 0.02)
        self.wo = nn.Parameter(torch.randn(EMBED_DIM, EMBED_DIM) * 0.02)
        self.ln2_w = nn.Parameter(torch.ones(EMBED_DIM))
        self.ln2_b = nn.Parameter(torch.zeros(EMBED_DIM))
        self.w1 = nn.Parameter(torch.randn(EMBED_DIM, FFN_HIDDEN) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(FFN_HIDDEN))
        self.w2 = nn.Parameter(torch.randn(FFN_HIDDEN, EMBED_DIM) * 0.02)
        self.b2 = nn.Parameter(torch.zeros(EMBED_DIM))

    def forward(self, x):
        # x: [B, T, EMBED_DIM]
        B, T, _ = x.shape
        h = F.layer_norm(x, (EMBED_DIM,), self.ln1_w, self.ln1_b, eps=LN_EPS)
        q = h @ self.wq
        k = h @ self.wk
        v = h @ self.wv

        q = q.view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)  # [B, H, T, D]
        k = k.view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (HEAD_DIM ** 0.5)  # [B, H, T, T]
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v  # [B, H, T, D]
        out = out.transpose(1, 2).reshape(B, T, EMBED_DIM)
        out = out @ self.wo
        x = x + out

        h2 = F.layer_norm(x, (EMBED_DIM,), self.ln2_w, self.ln2_b, eps=LN_EPS)
        ff = F.relu(h2 @ self.w1 + self.b1)
        ff = ff @ self.w2 + self.b2
        x = x + ff
        return x


class TLM32(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Parameter(torch.randn(VOCAB_SIZE, EMBED_DIM) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(CONTEXT_LEN, EMBED_DIM) * 0.02)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.final_ln_w = nn.Parameter(torch.ones(EMBED_DIM))
        self.final_ln_b = nn.Parameter(torch.zeros(EMBED_DIM))
        self.output_w = nn.Parameter(torch.randn(EMBED_DIM, VOCAB_SIZE) * 0.02)
        self.output_b = nn.Parameter(torch.zeros(VOCAB_SIZE))

    def forward(self, tokens):
        # tokens: [B, T] int64 byte values
        B, T = tokens.shape
        x = self.token_embedding[tokens] + self.pos_embedding[:T]  # [B, T, EMBED_DIM]
        for blk in self.blocks:
            x = blk(x)
        x = F.layer_norm(x, (EMBED_DIM,), self.final_ln_w, self.final_ln_b, eps=LN_EPS)
        logits = x @ self.output_w + self.output_b  # [B, T, VOCAB_SIZE]
        return logits

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
