"""Full definition of a GPT Language Model.

This module implements a complete GPT model with:
    - GELU activation function
    - Causal self-attention with optional Flash Attention optimization
    - Transformer blocks with MLP
    - Full GPT model with embedding, transformer layers, and LM head

References:
    1) OpenAI GPT-2 TensorFlow implementation:
       https://github.com/openai/gpt-2/blob/master/src/model.py
    2) HuggingFace Transformers PyTorch implementation:
       https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
from typing import Optional, Tuple, Dict, Any, Set

import torch
import torch.nn as nn
from torch.nn import functional as F

from mingpt.utils import CfgNode as CN

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Model architecture presets (following HuggingFace naming conventions)
MODEL_PRESETS: Dict[str, Dict[str, int]] = {
    # GPT-1
    "openai-gpt": {"n_layer": 12, "n_head": 12, "n_embd": 768},  # 117M params
    # GPT-2 configs
    "gpt2": {"n_layer": 12, "n_head": 12, "n_embd": 768},        # 124M params
    "gpt2-medium": {"n_layer": 24, "n_head": 16, "n_embd": 1024},  # 350M params
    "gpt2-large": {"n_layer": 36, "n_head": 20, "n_embd": 1280},   # 774M params
    "gpt2-xl": {"n_layer": 48, "n_head": 25, "n_embd": 1600},      # 1558M params
    # Gophers
    "gopher-44m": {"n_layer": 8, "n_head": 16, "n_embd": 512},
    # Tiny models for experimentation
    "gpt-mini": {"n_layer": 6, "n_head": 6, "n_embd": 192},
    "gpt-micro": {"n_layer": 4, "n_head": 4, "n_embd": 128},
    "gpt-nano": {"n_layer": 3, "n_head": 3, "n_embd": 48},
}

# Pretrained model configurations
PRETRAINED_MODELS: Set[str] = {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}

# Default hyperparameters
DEFAULT_EMBD_PDROP: float = 0.1
DEFAULT_RESID_PDROP: float = 0.1
DEFAULT_ATTN_PDROP: float = 0.1

# Weight initialization constants
INIT_STD: float = 0.02

# OpenAI GPT-2 vocab and block size
OPENAI_VOCAB_SIZE: int = 50257
OPENAI_BLOCK_SIZE: int = 1024

# -----------------------------------------------------------------------------
# Activation Functions
# -----------------------------------------------------------------------------


class NewGELU(nn.Module):
    """GELU activation function as used in Google BERT and OpenAI GPT.

    Reference: Gaussian Error Linear Units (GELU) paper
    https://arxiv.org/abs/1606.08415
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GELU activation.

        Args:
            x: Input tensor.

        Returns:
            Activated tensor.
        """
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
        ))


# -----------------------------------------------------------------------------
# Attention Mechanism
# -----------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention layer with optional Flash Attention.

    This implementation uses PyTorch's scaled_dot_product_attention when
    available for optimized performance, with automatic fallback to manual
    implementation. The causal mask ensures attention is only applied to
    previous positions in the sequence.

    Attributes:
        c_attn: Combined QKV projection linear layer.
        c_proj: Output projection linear layer.
        attn_dropout: Dropout applied to attention weights.
        resid_dropout: Dropout applied to output.
        bias: Causal mask buffer.
    """

    def __init__(self, config: CN) -> None:
        """Initialize the attention layer.

        Args:
            config: Configuration with n_embd, n_head, attn_pdrop, resid_pdrop.

        Raises:
            AssertionError: If n_embd is not divisible by n_head.
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0, (
            f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
        )

        self.n_head: int = config.n_head
        self.n_embd: int = config.n_embd
        self.head_dim: int = config.n_embd // config.n_head

        # Combined QKV projection for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # Regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # Causal mask to ensure attention is only applied to the left
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

        # Check for Flash Attention support (PyTorch 2.0+)
        self.use_flash_attn: bool = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through causal self-attention.

        Args:
            x: Input tensor of shape (B, T, C) where
               B = batch size, T = sequence length, C = embedding dimension.

        Returns:
            Output tensor of shape (B, T, C).
        """
        B, T, C = x.size()

        # Calculate Q, K, V for all heads in batch
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape to (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_flash_attn:
            # Use Flash Attention for optimized performance
            # Create causal mask for the current sequence length
            causal_mask = self.bias[:, :, :T, :T] == 0
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=causal_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=False  # We provide our own causal mask
            )
        else:
            # Manual attention computation
            # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
            y = att @ v

        # Re-assemble all head outputs: (B, nh, T, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


# -----------------------------------------------------------------------------
# MLP Module
# -----------------------------------------------------------------------------


class MLP(nn.Module):
    """Feed-forward network (MLP) used in Transformer blocks.

    Architecture: Linear -> GELU -> Linear -> Dropout
    The hidden dimension is 4x the input dimension (standard in GPT).

    Attributes:
        c_fc: First linear layer (expansion).
        c_proj: Second linear layer (projection).
        act: GELU activation function.
        dropout: Dropout layer.
    """

    def __init__(self, config: CN) -> None:
        """Initialize the MLP.

        Args:
            config: Configuration with n_embd and resid_pdrop.
        """
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.act = NewGELU()
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP.

        Args:
            x: Input tensor of shape (B, T, C).

        Returns:
            Output tensor of shape (B, T, C).
        """
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# -----------------------------------------------------------------------------
# Transformer Block
# -----------------------------------------------------------------------------


class Block(nn.Module):
    """Standard Transformer block with pre-normalization.

    Architecture:
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

    Attributes:
        ln_1: First layer normalization.
        attn: Causal self-attention module.
        ln_2: Second layer normalization.
        mlp: Feed-forward network.
    """

    def __init__(self, config: CN) -> None:
        """Initialize the transformer block.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the block.

        Args:
            x: Input tensor of shape (B, T, C).

        Returns:
            Output tensor of shape (B, T, C).
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# -----------------------------------------------------------------------------
# GPT Model
# -----------------------------------------------------------------------------


class GPT(nn.Module):
    """GPT Language Model.

    A decoder-only transformer for language modeling tasks.

    Attributes:
        block_size: Maximum sequence length.
        transformer: ModuleDict containing embeddings, dropout, blocks, and final LN.
        lm_head: Language modeling head (projection to vocab).
    """

    @staticmethod
    def get_default_config() -> CN:
        """Get default configuration for GPT model.

        Returns:
            CfgNode with default model hyperparameters.
        """
        C = CN()
        # Either model_type or (n_layer, n_head, n_embd) must be given
        C.model_type: Optional[str] = None
        C.n_layer: Optional[int] = None
        C.n_head: Optional[int] = None
        C.n_embd: Optional[int] = None
        # These options must be filled in externally
        C.vocab_size: Optional[int] = None
        C.block_size: Optional[int] = None
        # Dropout hyperparameters
        C.embd_pdrop: float = DEFAULT_EMBD_PDROP
        C.resid_pdrop: float = DEFAULT_RESID_PDROP
        C.attn_pdrop: float = DEFAULT_ATTN_PDROP
        return C

    def __init__(self, config: CN) -> None:
        """Initialize the GPT model.

        Args:
            config: Model configuration. Must specify either model_type or
                   (n_layer, n_head, n_embd), plus vocab_size and block_size.

        Raises:
            ValueError: If configuration is invalid.
        """
        super().__init__()
        self._validate_and_apply_config(config)
        self.config: CN = config
        self.block_size: int = config.block_size

        # Build transformer
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "wpe": nn.Embedding(config.block_size, config.n_embd),
            "drop": nn.Dropout(config.embd_pdrop),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": nn.LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)
        self._apply_special_init()

        # Report parameter count
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print(f"Number of parameters: {n_params / 1e6:.2f}M")

    def _validate_and_apply_config(self, config: CN) -> None:
        """Validate configuration and apply model presets if needed.

        Args:
            config: Configuration to validate and modify in-place.

        Raises:
            ValueError: If configuration is invalid or incomplete.
        """
        # Check vocab_size and block_size
        if config.vocab_size is None:
            raise ValueError("vocab_size must be specified in config")
        if config.block_size is None:
            raise ValueError("block_size must be specified in config")

        # Determine configuration source
        type_given = config.model_type is not None
        params_given = all([
            config.n_layer is not None,
            config.n_head is not None,
            config.n_embd is not None
        ])

        if type_given and params_given:
            raise ValueError(
                "Must specify exactly one of: model_type OR (n_layer, n_head, n_embd). "
                "Both were provided."
            )
        if not type_given and not params_given:
            raise ValueError(
                "Must specify exactly one of: model_type OR (n_layer, n_head, n_embd). "
                "Neither was provided."
            )

        if type_given:
            if config.model_type not in MODEL_PRESETS:
                valid_types = ", ".join(MODEL_PRESETS.keys())
                raise ValueError(
                    f"Unknown model_type '{config.model_type}'. "
                    f"Valid options: {valid_types}"
                )
            # Apply preset
            preset = MODEL_PRESETS[config.model_type]
            config.merge_from_dict(preset)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for a module.

        Args:
            module: Module to initialize.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def _apply_special_init(self) -> None:
        """Apply special scaled initialization to residual projections.

        This follows the GPT-2 paper's initialization scheme.
        """
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=INIT_STD / math.sqrt(2 * self.config.n_layer)
                )

    def configure_optimizers(
        self,
        train_config: CN,
        weight_decay: float = 0.1,
        learning_rate: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.95),
    ) -> torch.optim.AdamW:
        """Configure optimizer with weight decay.

        Separates parameters into two groups:
        - Parameters with weight decay (Linear layer weights)
        - Parameters without weight decay (biases, LayerNorm, Embedding)

        Args:
            train_config: Training configuration.
            weight_decay: Weight decay coefficient.
            learning_rate: Learning rate.
            betas: Adam beta parameters.

        Returns:
            Configured AdamW optimizer.

        Raises:
            AssertionError: If parameters are not properly categorized.
        """
        # Categorize parameters
        decay_params: Set[str] = set()
        no_decay_params: Set[str] = set()

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith("bias"):
                    no_decay_params.add(fpn)
                elif pn.endswith("weight"):
                    if isinstance(m, nn.Linear):
                        decay_params.add(fpn)
                    elif isinstance(m, (nn.LayerNorm, nn.Embedding)):
                        no_decay_params.add(fpn)

        # Validate all parameters are categorized
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay_params & no_decay_params
        union_params = decay_params | no_decay_params

        if inter_params:
            raise AssertionError(
                f"Parameters in both decay/no_decay sets: {inter_params}"
            )
        if param_dict.keys() - union_params:
            raise AssertionError(
                f"Parameters not categorized: {param_dict.keys() - union_params}"
            )

        # Create optimizer groups
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(decay_params)],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(no_decay_params)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the model.

        Args:
            idx: Input token indices of shape (B, T).
            targets: Optional target token indices of shape (B, T) for loss computation.

        Returns:
            Tuple of (logits, loss) where loss is None if targets not provided.

        Raises:
            AssertionError: If sequence length exceeds block_size.
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        )

        # Position indices
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        # Token and position embeddings
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final layer norm and projection
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # Compute loss if targets provided
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
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively.

        Args:
            idx: Conditioning sequence of shape (B, T).
            max_new_tokens: Number of new tokens to generate.
            temperature: Sampling temperature (lower = more deterministic).
            do_sample: If True, sample from distribution; else take argmax.
            top_k: If set, only sample from top k tokens.

        Returns:
            Generated sequence of shape (B, T + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]

            # Forward pass
            logits, _ = self(idx_cond)

            # Get logits for last position and apply temperature
            logits = logits[:, -1, :] / temperature

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample or take argmax
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)

            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @classmethod
    def from_pretrained(cls, model_type: str) -> "GPT":
        """Load a pretrained GPT-2 model from HuggingFace.

        Args:
            model_type: One of 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'.

        Returns:
            Initialized GPT model with pretrained weights.

        Raises:
            ValueError: If model_type is not supported.
        """
        if model_type not in PRETRAINED_MODELS:
            valid = ", ".join(PRETRAINED_MODELS)
            raise ValueError(
                f"model_type must be one of: {valid}, got: {model_type}"
            )

        from transformers import GPT2LMHeadModel

        # Create from-scratch minGPT model
        config = cls.get_default_config()
        config.model_type = model_type
        config.vocab_size = OPENAI_VOCAB_SIZE
        config.block_size = OPENAI_BLOCK_SIZE
        model = GPT(config)
        sd = model.state_dict()

        # Load HuggingFace model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Copy weights (handling Conv1D -> Linear conversion)
        keys = [k for k in sd_hf if not k.endswith("attn.masked_bias")]
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        assert len(keys) == len(sd), "State dict key count mismatch"

        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # Transpose Conv1D weights to Linear format
                assert sd_hf[k].shape[::-1] == sd[k].shape, (
                    f"Shape mismatch for {k}: {sd_hf[k].shape} vs {sd[k].shape}"
                )
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Direct copy
                assert sd_hf[k].shape == sd[k].shape, (
                    f"Shape mismatch for {k}: {sd_hf[k].shape} vs {sd[k].shape}"
                )
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
