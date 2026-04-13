"""
Full definition of a GPT Language Model.

This module implements a GPT-style transformer model with:
- Causal self-attention with scaled dot-product optimization
- MLP feed-forward blocks
- Configurable model architectures (GPT-1, GPT-2 variants, etc.)

References:
    1. Official GPT-2 TensorFlow implementation by OpenAI:
       https://github.com/openai/gpt-2/blob/master/src/model.py
    2. Hugging Face Transformers PyTorch implementation:
       https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from mingpt.utils import CfgNode as CN

WEIGHT_INIT_STD: float = 0.02
GELU_COEFFICIENT: float = 0.044715
SQRT_2_OVER_PI: float = math.sqrt(2.0 / math.pi)

MODEL_CONFIGS: Dict[str, Dict[str, int]] = {
    'openai-gpt': {'n_layer': 12, 'n_head': 12, 'n_embd': 768},
    'gpt2': {'n_layer': 12, 'n_head': 12, 'n_embd': 768},
    'gpt2-medium': {'n_layer': 24, 'n_head': 16, 'n_embd': 1024},
    'gpt2-large': {'n_layer': 36, 'n_head': 20, 'n_embd': 1280},
    'gpt2-xl': {'n_layer': 48, 'n_head': 25, 'n_embd': 1600},
    'gopher-44m': {'n_layer': 8, 'n_head': 16, 'n_embd': 512},
    'gpt-mini': {'n_layer': 6, 'n_head': 6, 'n_embd': 192},
    'gpt-micro': {'n_layer': 4, 'n_head': 4, 'n_embd': 128},
    'gpt-nano': {'n_layer': 3, 'n_head': 3, 'n_embd': 48},
}

PRETRAINED_MODEL_TYPES: Set[str] = {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

OPENAI_VOCAB_SIZE: int = 50257
OPENAI_BLOCK_SIZE: int = 1024

MLP_HIDDEN_MULTIPLIER: int = 4

TRANSPOSED_WEIGHT_SUFFIXES: Tuple[str, ...] = (
    'attn.c_attn.weight',
    'attn.c_proj.weight',
    'mlp.c_fc.weight',
    'mlp.c_proj.weight',
)


class NewGELU(nn.Module):
    """
    GELU activation function implementation.

    This is the exact GELU formulation used in Google BERT and OpenAI GPT.
    Reference: Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GELU activation.

        Args:
            x: Input tensor of any shape.

        Returns:
            Tensor with GELU applied, same shape as input.
        """
        return 0.5 * x * (1.0 + torch.tanh(SQRT_2_OVER_PI * (x + GELU_COEFFICIENT * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention layer with output projection.

    This implements masked self-attention where each position can only attend
    to positions before it (causal masking). Uses PyTorch's optimized
    scaled_dot_product_attention for improved performance.

    Attributes:
        c_attn: Combined QKV projection layer.
        c_proj: Output projection layer.
        attn_dropout: Dropout applied after attention.
        resid_dropout: Dropout applied after residual connection.
        n_head: Number of attention heads.
        n_embd: Embedding dimension.
    """

    def __init__(self, config: CN) -> None:
        """
        Initialize causal self-attention layer.

        Args:
            config: Model configuration containing:
                - n_embd: Embedding dimension (must be divisible by n_head)
                - n_head: Number of attention heads
                - block_size: Maximum sequence length
                - attn_pdrop: Attention dropout probability
                - resid_pdrop: Residual dropout probability

        Raises:
            AssertionError: If n_embd is not divisible by n_head.
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0, (
            f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
        )

        self.c_attn: nn.Linear = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj: nn.Linear = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout: nn.Dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout: nn.Dropout = nn.Dropout(config.resid_pdrop)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            )
        )

        self.n_head: int = config.n_head
        self.n_embd: int = config.n_embd
        self.head_dim: int = config.n_embd // config.n_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for causal self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd).

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd).
        """
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """
    Feed-forward MLP block for transformer.

    This implements the standard transformer MLP:
    x -> Linear -> GELU -> Linear -> Dropout

    Attributes:
        c_fc: First linear layer (expands to 4x embedding dimension).
        act: GELU activation function.
        c_proj: Second linear layer (projects back to embedding dimension).
        dropout: Dropout layer for regularization.
    """

    def __init__(self, config: CN) -> None:
        """
        Initialize MLP block.

        Args:
            config: Model configuration containing:
                - n_embd: Embedding dimension
                - resid_pdrop: Dropout probability
        """
        super().__init__()
        hidden_dim = MLP_HIDDEN_MULTIPLIER * config.n_embd
        self.c_fc: nn.Linear = nn.Linear(config.n_embd, hidden_dim)
        self.act: NewGELU = NewGELU()
        self.c_proj: nn.Linear = nn.Linear(hidden_dim, config.n_embd)
        self.dropout: nn.Dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MLP block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd).

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd).
        """
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Transformer block with pre-norm architecture.

    Implements: x -> x + attn(ln(x)) -> x + mlp(ln(x))

    Attributes:
        ln_1: First layer normalization (before attention).
        attn: Causal self-attention layer.
        ln_2: Second layer normalization (before MLP).
        mlp: Feed-forward MLP block.
    """

    def __init__(self, config: CN) -> None:
        """
        Initialize transformer block.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.ln_1: nn.LayerNorm = nn.LayerNorm(config.n_embd)
        self.attn: CausalSelfAttention = CausalSelfAttention(config)
        self.ln_2: nn.LayerNorm = nn.LayerNorm(config.n_embd)
        self.mlp: MLP = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd).

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd).
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """
    GPT Language Model.

    A transformer-based language model with configurable architecture.
    Supports loading pretrained weights from Hugging Face checkpoints.

    Attributes:
        block_size: Maximum sequence length.
        transformer: ModuleDict containing embedding and transformer layers.
        lm_head: Output language model head.
    """

    @staticmethod
    def get_default_config() -> CN:
        """
        Get default model configuration.

        Returns:
            Default configuration with model_type='gpt' and standard dropout values.
        """
        C = CN()
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd = None
        C.vocab_size = None
        C.block_size = None
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    @staticmethod
    def get_available_model_types() -> List[str]:
        """
        Get list of available pretrained model types.

        Returns:
            List of model type strings.
        """
        return list(MODEL_CONFIGS.keys())

    def __init__(self, config: CN) -> None:
        """
        Initialize GPT model.

        Args:
            config: Model configuration. Must provide either model_type or
                   (n_layer, n_head, n_embd), plus vocab_size and block_size.

        Raises:
            AssertionError: If required config values are missing or invalid.
            ValueError: If neither model_type nor manual config is provided.
        """
        super().__init__()

        self._validate_config(config)
        self.block_size: int = config.block_size

        self._resolve_model_config(config)

        self.transformer: nn.ModuleDict = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'drop': nn.Dropout(config.embd_pdrop),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd),
        })
        self.lm_head: nn.Linear = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self._init_weights(config)

        n_params = sum(p.numel() for p in self.transformer.parameters())
        print(f"number of parameters: {n_params / 1e6:.2f}M")

    def _validate_config(self, config: CN) -> None:
        """
        Validate model configuration.

        Args:
            config: Configuration to validate.

        Raises:
            AssertionError: If vocab_size or block_size is missing.
            ValueError: If config is invalid.
        """
        assert config.vocab_size is not None, "vocab_size must be specified"
        assert config.block_size is not None, "block_size must be specified"

        type_given = config.model_type is not None
        params_given = all([
            config.n_layer is not None,
            config.n_head is not None,
            config.n_embd is not None
        ])

        if not (type_given ^ params_given):
            raise ValueError(
                "Exactly one of model_type or (n_layer, n_head, n_embd) must be provided. "
                f"Got model_type={config.model_type}, "
                f"n_layer={config.n_layer}, n_head={config.n_head}, n_embd={config.n_embd}"
            )

        if type_given and config.model_type not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model_type '{config.model_type}'. "
                f"Available types: {list(MODEL_CONFIGS.keys())}"
            )

    def _resolve_model_config(self, config: CN) -> None:
        """
        Resolve model configuration from model_type if needed.

        Args:
            config: Configuration to update with resolved values.
        """
        if config.model_type is not None:
            model_config = MODEL_CONFIGS[config.model_type]
            config.merge_from_dict(model_config)

    def _init_weights(self, config: CN) -> None:
        """
        Initialize model weights.

        Applies normal initialization to linear and embedding layers,
        with special scaled initialization for residual projections.

        Args:
            config: Model configuration for scaled init.
        """
        self.apply(self._init_module_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=WEIGHT_INIT_STD / math.sqrt(2 * config.n_layer))

    def _init_module_weights(self, module: nn.Module) -> None:
        """
        Initialize weights for a single module.

        Args:
            module: Module to initialize.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=WEIGHT_INIT_STD)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=WEIGHT_INIT_STD)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @classmethod
    def from_pretrained(cls, model_type: str) -> GPT:
        """
        Load a pretrained GPT model from Hugging Face checkpoint.

        Args:
            model_type: One of 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'.

        Returns:
            GPT model with pretrained weights.

        Raises:
            AssertionError: If model_type is not supported.
        """
        assert model_type in PRETRAINED_MODEL_TYPES, (
            f"model_type must be one of {PRETRAINED_MODEL_TYPES}, got '{model_type}'"
        )

        from transformers import GPT2LMHeadModel

        config = cls.get_default_config()
        config.model_type = model_type
        config.vocab_size = OPENAI_VOCAB_SIZE
        config.block_size = OPENAI_BLOCK_SIZE
        model = cls(config)
        sd = model.state_dict()

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')]

        assert len(keys) == len(sd), (
            f"Parameter count mismatch: HF has {len(keys)}, model has {len(sd)}"
        )

        for k in keys:
            if any(k.endswith(w) for w in TRANSPOSED_WEIGHT_SUFFIXES):
                assert sd_hf[k].shape[::-1] == sd[k].shape, (
                    f"Shape mismatch for {k}: HF {sd_hf[k].shape} vs model {sd[k].shape}"
                )
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape, (
                    f"Shape mismatch for {k}: HF {sd_hf[k].shape} vs model {sd[k].shape}"
                )
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, train_config: CN) -> torch.optim.AdamW:
        """
        Configure optimizer with weight decay for appropriate parameters.

        Separates parameters into two groups:
        - Parameters that should have weight decay (Linear weights)
        - Parameters that should not (biases, LayerNorm, Embeddings)

        Args:
            train_config: Training configuration containing:
                - learning_rate: Learning rate
                - betas: Adam beta parameters
                - weight_decay: Weight decay coefficient

        Returns:
            Configured AdamW optimizer.
        """
        decay_params: Set[str] = set()
        no_decay_params: Set[str] = set()

        weight_decay_modules = (nn.Linear,)
        no_weight_decay_modules = (nn.LayerNorm, nn.Embedding)

        for module_name, module in self.named_modules():
            for param_name, _ in module.named_parameters():
                full_param_name = f"{module_name}.{param_name}" if module_name else param_name

                if param_name.endswith('bias'):
                    no_decay_params.add(full_param_name)
                elif param_name.endswith('weight') and isinstance(module, weight_decay_modules):
                    decay_params.add(full_param_name)
                elif param_name.endswith('weight') and isinstance(module, no_weight_decay_modules):
                    no_decay_params.add(full_param_name)

        param_dict = {pn: p for pn, p in self.named_parameters()}

        overlap = decay_params & no_decay_params
        assert len(overlap) == 0, f"Parameters in both decay/no_decay sets: {overlap}"

        missing = set(param_dict.keys()) - (decay_params | no_decay_params)
        assert len(missing) == 0, f"Parameters not in any set: {missing}"

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(decay_params)],
                "weight_decay": train_config.weight_decay
            },
            {
                "params": [param_dict[pn] for pn in sorted(no_decay_params)],
                "weight_decay": 0.0
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=train_config.learning_rate,
            betas=train_config.betas
        )
        return optimizer

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through GPT model.

        Args:
            idx: Input token indices of shape (batch_size, seq_len).
            targets: Optional target indices for loss computation.

        Returns:
            Tuple of:
                - logits: Output logits of shape (batch_size, seq_len, vocab_size)
                - loss: Optional cross-entropy loss if targets provided

        Raises:
            AssertionError: If sequence length exceeds block_size.
        """
        device = idx.device
        b, t = idx.size()

        assert t <= self.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        )

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss: Optional[torch.Tensor] = None
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
        """
        Generate new tokens autoregressively.

        Takes a conditioning sequence and generates new tokens by repeatedly
        predicting the next token and appending it to the sequence.

        Args:
            idx: Conditioning sequence of shape (batch_size, seq_len).
            max_new_tokens: Number of new tokens to generate.
            temperature: Sampling temperature (higher = more random).
            do_sample: If True, sample from distribution; else take argmax.
            top_k: If set, only sample from top-k most likely tokens.

        Returns:
            Generated sequence of shape (batch_size, seq_len + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)

            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx
