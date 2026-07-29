"""
A complete GPT in one readable file -- the 1.1B model this project trains, with no
framework abstractions between the maths and the code.

Structure (this is the whole model):

    tokens -> [token embedding + position embedding] -> [Block] x n_layer -> LayerNorm -> Linear -> logits

Each Block is:

    x = x + SelfAttention(LayerNorm(x))
    x = x + MLP(LayerNorm(x))
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 1024  # max context length (how many previous tokens the model can see)
    n_layer: int = 12       # number of transformer blocks stacked
    n_head: int = 12        # number of attention heads per block
    n_embd: int = 768       # embedding dimension (must be divisible by n_head)
    dropout: float = 0.0
    tied: bool = False      # share tok_emb/head weights (GPT-2 convention) -- a knob, not a
                             # hardcoded choice, so untied checkpoints from earlier runs keep
                             # loading correctly under the same model.py


class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with causal masking.

    Implements Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V for each head, run in
    parallel across n_head heads, then recombined. Uses PyTorch's fused
    scaled_dot_product_attention (dispatches to a flash-attention-style kernel on
    supported hardware) rather than computing the full (B, n_head, T, T) attention
    matrix by hand -- mathematically identical, but never materializes that matrix in
    GPU memory, which is what made large batch_size * block_size combinations OOM with
    the naive implementation. is_causal=True handles the "position i can only attend to
    positions <= i" masking internally, so no explicit mask tensor is needed.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # One linear layer produces Q, K, V all at once (3x the width), split afterward.
        # This is purely an implementation convenience -- mathematically identical to
        # three separate nn.Linear(n_embd, n_embd) layers.
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape  # batch, sequence length (<= block_size), embedding dim

        qkv = self.qkv_proj(x)  # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)  # each (B, T, C)

        # Reshape so each head gets its own slice of the embedding, processed in parallel.
        # (B, T, C) -> (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Fused attention: computes softmax(Q K^T / sqrt(d_k)) V (with causal masking
        # and dropout folded in) without ever materializing the full (B, n_head, T, T)
        # score matrix in memory.
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        # Recombine heads: (B, n_head, T, head_dim) -> (B, T, n_head, head_dim) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.out_proj(out))


class MLP(nn.Module):
    """
    Position-wise feedforward network. Applied independently to each token's vector
    (no mixing across positions -- that already happened in attention). Expands to 4x
    width then projects back down; this expansion factor is the GPT-2/nanoGPT convention,
    not a mathematical requirement.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc(x)
        x = F.gelu(x)      # smooth nonlinearity; without this the whole network would
        x = self.proj(x)   # collapse to one big linear function no matter how many layers
        return self.dropout(x)


class Block(nn.Module):
    """
    One transformer block: attention (mixes information across token positions), then
    MLP (processes each position independently). Both wrapped in residual connections
    and pre-norm LayerNorm.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # residual: gradient has a direct path around attn
        x = x + self.mlp(self.ln2(x))   # residual: gradient has a direct path around mlp
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # getattr, not config.tied: a GPTConfig pickled by an older model.py (before this
        # field existed) won't have the attribute in its __dict__ at all -- pickle restores
        # object state directly rather than re-running __init__, so the dataclass default
        # doesn't backfill it. Loading those checkpoints would otherwise AttributeError here.
        if getattr(config, "tied", False):
            self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"model initialized: {n_params / 1e6:.2f}M parameters")

    def _init_weights(self, module):
        # Small random init, as in GPT-2. Not load-bearing for understanding the model,
        # but bad init can make a small model fail to train at all.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, "sequence longer than block_size"

        positions = torch.arange(0, T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(positions)  # (B, T, n_embd), broadcast over B
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        logits = self.head(x)  # (B, T, vocab_size) -- unnormalized next-token scores

        loss = None
        if targets is not None:
            # Cross-entropy over the vocab dimension, averaged over every position in
            # every sequence in the batch. This is the single training objective.
            # ignore_index=-100 lets finetune_train.py mask prompt tokens out of the
            # loss (SFT-style, loss on answer tokens only) by setting their target to
            # -100 -- base pretraining targets never contain -100, so this is a no-op
            # for base_train.py.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, repetition_penalty=1.0,
                 no_penalty_ids=None, penalize_from=0):
        """
        Autoregressive sampling: repeatedly predict the next token, sample one,
        append it, and feed the whole sequence back in. Yields each new token as it's
        sampled (rather than returning the full sequence at the end), so a caller can
        print incrementally instead of waiting for the whole reply. Slow (recomputes
        everything each step) but simple -- this is intentionally not KV-cached.

        Note: no_grad is a `with` block inside the loop, not a `@torch.no_grad()`
        decorator on this function -- a decorator only wraps the (instant, no-op) call
        that creates the generator object, not the actual iteration that happens across
        later next() calls, so it would silently fail to suppress grad tracking during
        generation. The `with` block, entered on first iteration, stays active across
        yields for the generator's whole lifetime.

        repetition_penalty (>1.0 to enable, 1.0 = off): temperature and top_k alone have
        no notion of what's already been said, so sampling can fall into "X is Y, Y is
        Y" loops once the model settles into a confident-but-wrong local pattern.
        Standard fix (Keskar et al. 2019, "CTRL"): every token that has appeared
        anywhere in the sequence so far (prompt included, by default -- see
        penalize_from) gets its logit divided by the penalty if positive, multiplied if
        negative -- pushing already-used tokens down without forbidding them outright.

        no_penalty_ids (optional LongTensor): token ids exempt from repetition_penalty
        even if they appear in idx. Needed for byte-level BPE vocabularies -- a
        multi-byte UTF-8 character (e.g. accented letters) is often split across two or
        more raw continuation-byte tokens, and those specific byte values are shared
        across many unrelated characters. Without this exemption, a passage in the
        prompt containing any accented character can mark a continuation-byte token
        "seen", permanently suppressing it -- and thus permanently breaking -- a
        *different* multi-byte character the model tries to produce later, since
        "seen" here means "appeared anywhere in idx", prompt included.

        penalize_from (int, default 0): only tokens at this position or later in idx
        count as "seen" for repetition_penalty. Default 0 penalizes based on the whole
        sequence, prompt included -- fine for plain completion, but wrong for a
        RAG-style prompt: the Context passage's whole point is to be cited, so
        penalizing every one of its tokens from the first generated token onward
        actively discourages the model from repeating the fact it was just given,
        pushing it toward vaguer paraphrases or a different, unpenalized (and possibly
        wrong) fact instead. Callers building such a prompt should pass the prompt's
        token count here, so the penalty only discourages the model from repeating
        *itself* within its own answer -- the loop-prevention repetition_penalty is
        actually meant for -- not from repeating the source material it's grounded in.
        """
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.config.block_size:]  # crop to last block_size tokens
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature  # only care about next-token prediction

                if repetition_penalty != 1.0:
                    seen = torch.zeros_like(logits).scatter_(1, idx[:, penalize_from:], 1.0).bool()
                    if no_penalty_ids is not None:
                        seen[:, no_penalty_ids] = False
                    penalized = torch.where(logits > 0, logits / repetition_penalty, logits * repetition_penalty)
                    logits = torch.where(seen, penalized, logits)

                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = float("-inf")  # zero out all but top-k choices

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)  # sample, don't argmax
                idx = torch.cat((idx, idx_next), dim=1)
                yield idx_next
