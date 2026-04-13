"""Byte Pair Encoding (BPE) tokenizer for GPT models.

This module implements the BPE algorithm used by OpenAI's GPT-2 models.
It translates arbitrary UTF-8 strings into sequences of integers, where each
integer represents small chunks of commonly occurring characters.

This implementation is based on OpenAI's GPT-2 encoder:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

With modifications for clarity and PyTorch integration.
"""

import os
import json
from typing import Dict, Tuple, Set, List, Optional, Union
from pathlib import Path

import regex as re
import requests
import torch

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Byte range constants
BYTE_MIN: int = 0
BYTE_MAX: int = 256  # 2^8
BYTE_SHIFT: int = 256  # For mapping "ugly" bytes to nice unicode chars

# ASCII range constants
ASCII_EXCLAMATION: int = ord("!")
ASCII_TILDE: int = ord("~")
ASCII_INVERTED_EXCLAMATION: int = ord("¡")
ASCII_NOT_SIGN: int = ord("¬")
ASCII_REGISTERED: int = ord("®")
ASCII_Y_DIAERESIS: int = ord("ÿ")

# OpenAI GPT-2 model constants
ENCODER_URL: str = "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json"
VOCAB_URL: str = "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe"

# Vocabulary constants
VOCAB_SIZE: int = 50257  # 256 byte tokens + 50,000 merged + 1 special
NUM_BYTE_TOKENS: int = 256
NUM_MERGED_TOKENS: int = 50000
NUM_SPECIAL_TOKENS: int = 1
SPECIAL_TOKEN: str = "<|endoftext|>"

# Cache directory
CACHE_DIR_NAME: str = ".cache"
MINGPT_DIR_NAME: str = "mingpt"
ENCODER_FILENAME: str = "encoder.json"
VOCAB_FILENAME: str = "vocab.bpe"

# Regex pattern for pre-tokenization
# Explanation:
#   - 's|'t|'re|'ve|'m|'ll|'d: Common apostrophe contractions
#   - ?\p{L}+: Optional space + letters
#   - ?\p{N}+: Optional space + numbers
#   - ?[^\s\p{L}\p{N}]+: Optional space + non-space/non-letter/non-number
#   - \s+(?!\S): Whitespace sequences (excluding trailing space)
#   - \s+: Trailing whitespace
PRETOKENIZATION_PATTERN: str = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# -----------------------------------------------------------------------------
# Byte Encoding Utilities
# -----------------------------------------------------------------------------


def bytes_to_unicode() -> Dict[int, str]:
    """Create mapping from bytes to unicode characters.

    Every byte (0-255) gets mapped to a unicode character that represents
    it visually. Some bytes preserve their original appearance, while others
    are shifted to the range 256-511 for nicer display.

    For example:
        - byte 33 ('!') -> '!'
        - byte 32 (' ') -> 'Ġ' (shifted by 256)
        - byte 0 -> 'Ā' (shifted by 256)

    Returns:
        Dictionary mapping byte values (int) to unicode strings.
    """
    # Bytes that render fine in their original form
    nice_bytes: List[int] = (
        list(range(ASCII_EXCLAMATION, ASCII_TILDE + 1))
        + list(range(ASCII_INVERTED_EXCLAMATION, ASCII_NOT_SIGN + 1))
        + list(range(ASCII_REGISTERED, ASCII_Y_DIAERESIS + 1))
    )

    # Start with nice bytes mapped to themselves
    bs: List[int] = nice_bytes[:]
    cs: List[int] = nice_bytes[:]

    # Map remaining bytes to shifted unicode positions
    n: int = 0
    for b in range(BYTE_MAX):
        if b not in bs:
            bs.append(b)
            cs.append(BYTE_SHIFT + n)
            n += 1

    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word: Tuple[str, ...]) -> Set[Tuple[str, str]]:
    """Extract all consecutive character pairs (bigrams) from a word.

    Args:
        word: Tuple of characters representing a word.

    Returns:
        Set of all consecutive character pairs.
    """
    pairs: Set[Tuple[str, str]] = set()
    if len(word) < 2:
        return pairs

    prev_char: str = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char

    return pairs


# -----------------------------------------------------------------------------
# BPE Encoder
# -----------------------------------------------------------------------------


class Encoder:
    """BPE Encoder/Decoder for tokenizing text.

    Attributes:
        byte_encoder: Mapping from bytes to unicode characters.
        byte_decoder: Inverse mapping from unicode to bytes.
        encoder: Mapping from BPE tokens to token IDs.
        decoder: Inverse mapping from token IDs to BPE tokens.
        bpe_ranks: Ranking of BPE merge operations.
        pat: Regex pattern for pre-tokenization.
        cache: Cache for BPE results.
    """

    def __init__(
        self,
        encoder: Dict[str, int],
        bpe_merges: List[Tuple[str, str]],
    ) -> None:
        """Initialize the BPE encoder.

        Args:
            encoder: Dictionary mapping BPE tokens to integer IDs.
            bpe_merges: List of BPE merge operations as character pairs.
        """
        # Byte encoder/decoder
        self.byte_encoder: Dict[int, str] = bytes_to_unicode()
        self.byte_decoder: Dict[str, int] = {
            v: k for k, v in self.byte_encoder.items()
        }

        # BPE token encoder/decoder
        self.encoder: Dict[str, int] = encoder
        self.decoder: Dict[int, str] = {v: k for k, v in self.encoder.items()}

        # BPE merge rankings
        self.bpe_ranks: Dict[Tuple[str, str], int] = dict(
            zip(bpe_merges, range(len(bpe_merges)))
        )

        # Pre-tokenization pattern
        self.pat: re.Pattern = re.compile(PRETOKENIZATION_PATTERN)

        # Cache for BPE results
        self.cache: Dict[str, str] = {}

    def bpe(self, token: str) -> str:
        """Apply BPE encoding to a single token.

        Iteratively merges character pairs according to BPE rankings
        until no more merges are possible.

        Args:
            token: Input token string (after byte encoding).

        Returns:
            Space-separated BPE tokens.
        """
        # Check cache
        if token in self.cache:
            return self.cache[token]

        # Convert to character tuple
        word: Tuple[str, ...] = tuple(token)
        pairs: Set[Tuple[str, str]] = get_pairs(word)

        if not pairs:
            return token

        # Iteratively merge pairs
        while True:
            # Find the pair with lowest rank (highest priority)
            bigram: Tuple[str, str] = min(
                pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf"))
            )

            if bigram not in self.bpe_ranks:
                break  # No more eligible pairs

            first, second = bigram

            # Replace all occurrences of (first, second) with merged token
            new_word: List[str] = []
            i: int = 0
            while i < len(word):
                # Find next occurrence of first
                try:
                    j: int = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                # Merge if followed by second
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)

        # Join with space (safe because byte encoding ensures no spaces in data)
        result: str = " ".join(word)
        self.cache[token] = result
        return result

    def encode(self, text: str) -> List[int]:
        """Encode text to a list of token IDs.

        Args:
            text: Input text string.

        Returns:
            List of integer token IDs.
        """
        bpe_idx: List[int] = []

        # Pre-tokenize into word-like tokens
        tokens: List[str] = re.findall(self.pat, text)

        for token in tokens:
            # Encode token as bytes
            token_bytes: bytes = token.encode("utf-8")

            # Translate bytes to unicode representation
            token_translated: str = "".join(
                self.byte_encoder[b] for b in token_bytes
            )

            # Apply BPE merges
            token_merged: List[str] = self.bpe(token_translated).split(" ")

            # Convert to token IDs
            token_ix: List[int] = [
                self.encoder[bpe_token] for bpe_token in token_merged
            ]

            bpe_idx.extend(token_ix)

        return bpe_idx

    def encode_and_show_work(self, text: str) -> Dict[str, any]:
        """Encode text with detailed intermediate steps for debugging.

        Args:
            text: Input text string.

        Returns:
            Dictionary containing all intermediate processing steps.
        """
        bpe_idx: List[int] = []
        parts: List[Dict[str, any]] = []

        tokens: List[str] = re.findall(self.pat, text)

        for token in tokens:
            token_bytes: bytes = token.encode("utf-8")
            token_translated: str = "".join(
                self.byte_encoder[b] for b in token_bytes
            )
            token_merged: List[str] = self.bpe(token_translated).split(" ")
            token_ix: List[int] = [
                self.encoder[bpe_token] for bpe_token in token_merged
            ]

            bpe_idx.extend(token_ix)
            parts.append({
                "token": token,
                "token_bytes": token_bytes,
                "token_translated": token_translated,
                "token_merged": token_merged,
                "token_ix": token_ix,
            })

        return {
            "bpe_idx": bpe_idx,
            "tokens": tokens,
            "parts": parts,
        }

    def decode(self, bpe_idx: List[int]) -> str:
        """Decode a list of token IDs back to text.

        Args:
            bpe_idx: List of integer token IDs.

        Returns:
            Decoded text string.
        """
        # Map IDs to tokens
        tokens_merged: List[str] = [self.decoder[token] for token in bpe_idx]

        # Join and decode bytes
        tokens_flat: str = "".join(tokens_merged)
        tokens_bytes: bytearray = bytearray(
            self.byte_decoder[c] for c in tokens_flat
        )

        # Decode to UTF-8 string
        text: str = tokens_bytes.decode("utf-8", errors="replace")
        return text


# -----------------------------------------------------------------------------
# File Download and Cache
# -----------------------------------------------------------------------------


def get_file(local_file: Union[str, Path], remote_file: str) -> None:
    """Download a remote file to a local path if it doesn't exist.

    Args:
        local_file: Local file path.
        remote_file: Remote URL to download from.
    """
    local_path = Path(local_file)
    if not local_path.exists():
        print(f"Downloading {remote_file} to {local_file}")
        response = requests.get(remote_file)
        response.raise_for_status()
        local_path.write_bytes(response.content)


def get_cache_dir() -> Path:
    """Get the cache directory for minGPT files.

    Returns:
        Path to the cache directory.
    """
    home_dir = Path.home()
    cache_dir = home_dir / CACHE_DIR_NAME / MINGPT_DIR_NAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_encoder() -> Encoder:
    """Get a BPE encoder with cached model files.

    Downloads encoder.json and vocab.bpe from OpenAI if not cached.

    Returns:
        Configured Encoder instance.
    """
    cache_dir = get_cache_dir()

    # Load encoder.json
    encoder_path = cache_dir / ENCODER_FILENAME
    get_file(encoder_path, ENCODER_URL)
    encoder: Dict[str, int] = json.loads(encoder_path.read_text(encoding="utf-8"))
    assert len(encoder) == VOCAB_SIZE, (
        f"Expected vocab size {VOCAB_SIZE}, got {len(encoder)}"
    )

    # Load vocab.bpe
    vocab_path = cache_dir / VOCAB_FILENAME
    get_file(vocab_path, VOCAB_URL)
    bpe_data: str = vocab_path.read_text(encoding="utf-8")

    # Parse BPE merges (skip version line and blank last line)
    bpe_lines: List[str] = bpe_data.split("\n")
    bpe_merges: List[Tuple[str, str]] = [
        tuple(merge_str.split()) for merge_str in bpe_lines[1:-1]
    ]
    assert len(bpe_merges) == NUM_MERGED_TOKENS, (
        f"Expected {NUM_MERGED_TOKENS} merges, got {len(bpe_merges)}"
    )

    return Encoder(encoder, bpe_merges)


# -----------------------------------------------------------------------------
# PyTorch Tokenizer Wrapper
# -----------------------------------------------------------------------------


class BPETokenizer:
    """PyTorch-aware BPE tokenizer wrapper.

    Provides a HuggingFace-style interface for encoding/decoding text
    with PyTorch tensor outputs.
    """

    def __init__(self) -> None:
        """Initialize the tokenizer."""
        self.encoder: Encoder = get_encoder()

    def __call__(
        self,
        text: str,
        return_tensors: str = "pt",
    ) -> torch.Tensor:
        """Encode text to PyTorch tensor.

        Args:
            text: Input text string.
            return_tensors: Only "pt" (PyTorch) is supported.

        Returns:
            Tensor of token IDs with shape (1, sequence_length).

        Raises:
            AssertionError: If return_tensors is not "pt" or text is not a string.
        """
        assert return_tensors == "pt", f"Only return_tensors='pt' supported, got {return_tensors}"
        assert isinstance(text, str), f"Expected string input, got {type(text)}"

        idx: List[List[int]] = [self.encoder.encode(text)]
        return torch.tensor(idx, dtype=torch.long)

    def decode(self, idx: torch.Tensor) -> str:
        """Decode tensor of token IDs to text.

        Args:
            idx: 1D tensor of token IDs.

        Returns:
            Decoded text string.

        Raises:
            AssertionError: If idx is not a 1D tensor.
        """
        assert idx.ndim == 1, f"Expected 1D tensor, got shape {idx.shape}"
        return self.encoder.decode(idx.tolist())

    def encode(self, text: str) -> List[int]:
        """Encode text to list of token IDs.

        Args:
            text: Input text string.

        Returns:
            List of integer token IDs.
        """
        return self.encoder.encode(text)


# -----------------------------------------------------------------------------
# Main (for testing)
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    # Example encoding
    text = "Hello!! I'm Andrej Karpathy. It's 2022. w00t :D 🤗"
    e = get_encoder()
    r = e.encode_and_show_work(text)

    print("Original text:")
    print(text)
    print("\nPre-tokenization result:")
    print(r["tokens"])
    print("\nIntermediate processing for each token:")
    for part in r["parts"]:
        print(part)
    print("\nFinal token IDs:")
    print(r["bpe_idx"])
    print("\nReady to feed into a Transformer!")
