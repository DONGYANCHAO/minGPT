"""
Full definition of a GPT Language Model.

References:
1) Official GPT-2 TensorFlow implementation by OpenAI:
   https://github.com/openai/gpt-2/blob/master/src/model.py
2) HuggingFace Transformers PyTorch implementation:
   https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
from typing import (
    Any, Callable, Dict, Iterable, List, Optional,
    Set, Tuple, Type, Union, cast
)

import torch
import torch.nn as nn
from torch.nn import functional as F

from mingpt.utils import CfgNode as CN


MODEL_CONFIGS: Dict[str, Dict[str, int]] = {
    'openai-gpt':   {'n_layer': 12, 'n_head': 12, 'n_embd': 768},
    'gpt2':         {'n_layer': 12, 'n_head': 12, 'n_embd': 768},
    'gpt2-medium':  {'n_layer': 24, 'n_head': 16, 'n_embd': 1024},
    'gpt2-large':   {'n_layer': 36, 'n_head': 20, 'n_embd': 1280},
    'gpt2-xl':      {'n_layer': 48, 'n_head': 25, 'n_embd': 1600},
    'gopher-44m':   {'n_layer': 8, 'n_head': 16, 'n_embd': 512},
    'gpt-mini':     {'n_layer': 6, 'n_head': 6, 'n_embd': 192},
    'gpt-micro':    {'n_layer': 4, 'n_head': 4, 'n_embd': 128},
    'gpt-nano':     {'n_layer': 3, 'n_head': 3, 'n_embd': 48},
}

PRETRAINED_MODEL_TYPES = {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
OPENAI_VOCAB_SIZE = 50257
OPENAI_BLOCK_SIZE = 1024
DEFAULT_DROPOUT = 0.1
DEFAULT_INIT_STD = 0.02
RESIDUAL_SCALE_FACTOR = 2
GELU_CONSTANT = 0.044715
GELU_SCALE = math.sqrt(2.0 / math.pi)
MILLION = 1e6


class InvalidConfigError(ValueError):
    """Raised when model configuration is invalid."""
    pass


class NewGELU(nn.Module):
    """Gaussian Error Linear Units activation function.

    Implementation matching Google BERT and OpenAI GPT.

    Reference: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(GELU_SCALE * (x + GELU_CONSTANT * torch.pow(x, 3.0))))


class MLP(nn.Module):
    """Multi-layer perceptron for Transformer block.

    Architecture: Linear -> GELU -> Linear -> Dropout
    """

    c_fc: nn.Linear
    c_proj: nn.Linear
    act: NewGELU
    dropout: nn.Dropout

    def __init__(self, n_embd: int, resid_pdrop: float) -> None:
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.act = NewGELU()
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention with causal masking.

    Uses PyTorch's scaled_dot_product_attention for optimized performance.
    """

    c_attn: nn.Linear
    c_proj: nn.Linear
    attn_dropout: nn.Dropout
    resid_dropout: nn.Dropout
    n_head: int
    n_embd: int

    def __init__(self, config: CN) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise InvalidConfigError(
                f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
            )

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        is_causal = True
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=is_causal
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    """Transformer block with pre-layer normalization.

    Architecture: LN -> CausalSelfAttention -> Residual -> LN -> MLP -> Residual
    """

    ln_1: nn.LayerNorm
    attn: CausalSelfAttention
    ln_2: nn.LayerNorm
    mlp: MLP

    def __init__(self, config: CN) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config.n_embd, config.resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT Language Model with causal attention."""

    block_size: int
    transformer: nn.ModuleDict
    lm_head: nn.Linear

    @staticmethod
    def get_default_config() -> CN:
        """Get default model configuration.

        Returns:
            CfgNode with default hyperparameters.
        """
        C = CN()
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd = None
        C.vocab_size = None
        C.block_size = None
        C.embd_pdrop = DEFAULT_DROPOUT
        C.resid_pdrop = DEFAULT_DROPOUT
        C.attn_pdrop = DEFAULT_DROPOUT
        return C

    def __init__(self, config: CN) -> None:
        super().__init__()

        if config.vocab_size is None:
            raise InvalidConfigError("vocab_size must be specified in config")
        if config.block_size is None:
            raise InvalidConfigError("block_size must be specified in config")

        self.block_size = config.block_size

        type_given = config.model_type is not None
        params_given = all([
            config.n_layer is not None,
            config.n_head is not None,
            config.n_embd is not None
        ])

        if not (type_given ^ params_given):
            raise InvalidConfigError(
                "Exactly one of (model_type) or (n_layer, n_head, n_embd) must be specified. "
                f"Got type_given={type_given}, params_given={params_given}"
            )

        if type_given:
            if config.model_type not in MODEL_CONFIGS:
                raise InvalidConfigError(
                    f"Unknown model_type: {config.model_type}. "
                    f"Available: {sorted(MODEL_CONFIGS.keys())}"
                )
            config.merge_from_dict(MODEL_CONFIGS[config.model_type])

        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'drop': nn.Dropout(config.embd_pdrop),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(
                    p, mean=0.0,
                    std=DEFAULT_INIT_STD / math.sqrt(RESIDUAL_SCALE_FACTOR * config.n_layer)
                )

        n_params = sum(p.numel() for p in self.transformer.parameters())
        print(f"Number of parameters: {n_params / MILLION:.2f}M")

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize model weights.

        Args:
            module: PyTorch module to initialize.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=DEFAULT_INIT_STD)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=DEFAULT_INIT_STD)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    @classmethod
    def from_pretrained(cls: Type['GPT'], model_type: str) -> 'GPT':
        """Initialize pretrained GPT model from HuggingFace checkpoint.

        Args:
            model_type: One of {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        Returns:
            GPT model loaded with pretrained weights.

        Raises:
            InvalidConfigError: If model_type is not supported for pretrained loading.
        """
        if model_type not in PRETRAINED_MODEL_TYPES:
            raise InvalidConfigError(
                f"Pretrained loading not available for {model_type}. "
                f"Available: {sorted(PRETRAINED_MODEL_TYPES)}"
            )

        from transformers import GPT2LMHeadModel

        config = cls.get_default_config()
        config.model_type = model_type
        config.vocab_size = OPENAI_VOCAB_SIZE
        config.block_size = OPENAI_BLOCK_SIZE
        model = GPT(config)
        sd = model.state_dict()

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')]
        transposed = [
            'attn.c_attn.weight',
            'attn.c_proj.weight',
            'mlp.c_fc.weight',
            'mlp.c_proj.weight'
        ]

        if len(keys) != len(sd):
            raise RuntimeError(
                f"State dict mismatch: HF has {len(keys)} keys, "
                f"minGPT has {len(sd)} keys"
            )

        for k in keys:
            if any(k.endswith(w) for w in transposed):
                if sd_hf[k].shape[::-1] != sd[k].shape:
                    raise RuntimeError(f"Shape mismatch for {k}: {sd_hf[k].shape} vs {sd[k].shape}")
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                if sd_hf[k].shape != sd[k].shape:
                    raise RuntimeError(f"Shape mismatch for {k}: {sd_hf[k].shape} vs {sd[k].shape}")
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, train_config: CN) -> torch.optim.AdamW:
        """Configure AdamW optimizer with separated weight decay.

        Separates parameters into those with weight decay (Linear weights)
        and those without (biases, LayerNorm, Embedding weights).

        Args:
            train_config: Training configuration with learning_rate, betas, weight_decay.

        Returns:
            Configured AdamW optimizer.

        Raises:
            RuntimeError: If parameter separation has conflicts or omissions.
        """
        decay_modules: Tuple[Type[nn.Module], ...] = (nn.Linear,)
        no_decay_modules: Tuple[Type[nn.Module], ...] = (nn.LayerNorm, nn.Embedding)

        decay: Set[str] = set()
        no_decay: Set[str] = set()

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight'):
                    if isinstance(m, decay_modules):
                        decay.add(fpn)
                    elif isinstance(m, no_decay_modules):
                        no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}

        intersection = decay & no_decay
        if intersection:
            raise RuntimeError(f"Parameters in both decay sets: {intersection}")

        missing = param_dict.keys() - (decay | no_decay)
        if missing:
            raise RuntimeError(f"Parameters not in any decay set: {missing}")

        optim_groups = [
            {
                'params': [param_dict[pn] for pn in sorted(decay)],
                'weight_decay': train_config.weight_decay
            },
            {
                'params': [param_dict[pn] for pn in sorted(no_decay)],
                'weight_decay': 0.0
            },
        ]

        return torch.optim.AdamW(
            optim_groups,
            lr=train_config.learning_rate,
            betas=train_config.betas
        )

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of the model.

        Args:
            idx: Input token indices, shape (batch_size, sequence_length)
            targets: Optional target indices for loss computation, same shape as idx

        Returns:
            Tuple of (logits, loss). Loss is None if targets not provided.
        """
        device = idx.device
        b, t = idx.size()

        if t > self.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {t}, "
                f"block size is only {self.block_size}"
            )

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        wte = cast(nn.Embedding, self.transformer['wte'])
        wpe = cast(nn.Embedding, self.transformer['wpe'])
        drop = cast(nn.Dropout, self.transformer['drop'])
        h = cast(nn.ModuleList, self.transformer['h'])
        ln_f = cast(nn.LayerNorm, self.transformer['ln_f'])

        tok_emb = wte(idx)
        pos_emb = wpe(pos)
        x = drop(tok_emb + pos_emb)

        for block in h:
            x = block(x)

        x = ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """Generate text by autoregressive sampling.

        Args:
            idx: Conditioning sequence, shape (batch_size, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (1.0 = default, < 1.0 = more conservative)
            do_sample: If True, sample from distribution; if False, take argmax
            top_k: If provided, restrict sampling to top k most likely tokens

        Returns:
            Extended sequence with generated tokens, shape (batch_size, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)

            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx
