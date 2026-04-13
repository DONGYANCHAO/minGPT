"""
Byte Pair Encoding (BPE) tokenizer.

Translates arbitrary UTF-8 strings into sequences of integers representing
common character chunks. Based on OpenAI's GPT-2 encoder.py with additional
documentation and type safety.

Reference: https://github.com/openai/gpt-2/blob/master/src/encoder.py
"""

import os
import json
from typing import (
    Any, Dict, List, Mapping,
    Sequence, Set, Tuple, Union, cast
)

import regex as re
import requests
import torch


CACHE_DIR = os.path.expanduser('~/.cache/mingpt')
ENCODER_FILENAME = 'encoder.json'
VOCAB_FILENAME = 'vocab.bpe'
ENCODER_REMOTE = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json'
VOCAB_REMOTE = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe'
EXPECTED_ENCODER_SIZE = 50257
EXPECTED_BPE_MERGES = 50000

BPE_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

BytePair = Tuple[str, str]


def bytes_to_unicode() -> Dict[int, str]:
    """Map bytes 0..255 to visually pleasing unicode characters.

    Bytes that render nicely are preserved. Ugly control characters are
    shifted to a nice unicode range starting at chr(256).

    Returns:
        Dictionary mapping byte values (0-255) to unicode characters.
    """
    bs: List[int] = (
        list(range(ord("!"), ord("~") + 1)) +
        list(range(ord("¡"), ord("¬") + 1)) +
        list(range(ord("®"), ord("ÿ") + 1))
    )
    cs: List[int] = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs_str = [chr(n) for n in cs]
    return dict(zip(bs, cs_str))


def get_pairs(word: Sequence[str]) -> Set[BytePair]:
    """Return all consecutive bigram pairs in a word.

    Args:
        word: Sequence of characters/tokens.

    Returns:
        Set of (prev, current) character tuples.
    """
    pairs: Set[BytePair] = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Encoder:
    """BPE encoder and decoder.

    Attributes:
        byte_encoder: Maps byte values to unicode characters.
        byte_decoder: Reverse mapping of byte_encoder.
        encoder: Maps BPE tokens to integer IDs.
        decoder: Reverse mapping of encoder.
        bpe_ranks: Maps (a, b) pairs to their merge priority.
        pat: Regex pattern for pre-tokenization.
        cache: Memoization cache for BPE results.
    """

    byte_encoder: Dict[int, str]
    byte_decoder: Dict[str, int]
    encoder: Dict[str, int]
    decoder: Dict[int, str]
    bpe_ranks: Dict[BytePair, int]
    pat: re.Pattern[str]
    cache: Dict[str, str]

    def __init__(self, encoder: Dict[str, int], bpe_merges: List[BytePair]) -> None:
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.pat = re.compile(BPE_PATTERN)
        self.cache = {}

    def bpe(self, token: str) -> str:
        """Apply BPE merges to a single token.

        Args:
            token: Single token string after byte encoding.

        Returns:
            Space-separated string of BPE tokens.
        """
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            self.cache[token] = token
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word: List[str] = []
            i = 0

            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            new_word_tuple = tuple(new_word)
            word = new_word_tuple
            if len(word) == 1:
                break
            pairs = get_pairs(word)

        result = ' '.join(word)
        self.cache[token] = result
        return result

    def encode(self, text: str) -> List[int]:
        """Encode text string to BPE token IDs.

        Args:
            text: Input text string.

        Returns:
            List of integer token IDs.
        """
        bpe_idx: List[int] = []
        tokens = re.findall(self.pat, text)

        for token in tokens:
            token_bytes = token.encode('utf-8')
            token_translated = ''.join(self.byte_encoder[b] for b in token_bytes)
            token_merged = self.bpe(token_translated).split(' ')
            token_ix = [self.encoder[bpe_token] for bpe_token in token_merged]
            bpe_idx.extend(token_ix)

        return bpe_idx

    def encode_and_show_work(self, text: str) -> Dict[str, Any]:
        """Encode text with debugging information.

        Args:
            text: Input text string.

        Returns:
            Dictionary with final tokens, intermediate tokens, and details.
        """
        bpe_idx: List[int] = []
        parts: List[Dict[str, Any]] = []
        tokens = re.findall(self.pat, text)

        for token in tokens:
            token_bytes = token.encode('utf-8')
            token_translated = ''.join(self.byte_encoder[b] for b in token_bytes)
            token_merged = self.bpe(token_translated).split(' ')
            token_ix = [self.encoder[bpe_token] for bpe_token in token_merged]
            bpe_idx.extend(token_ix)
            parts.append({
                'token': token,
                'token_bytes': token_bytes,
                'token_translated': token_translated,
                'token_merged': token_merged,
                'token_ix': token_ix,
            })

        return {
            'bpe_idx': bpe_idx,
            'tokens': tokens,
            'parts': parts,
        }

    def decode(self, bpe_idx: Sequence[int]) -> str:
        """Decode BPE token IDs back to text.

        Args:
            bpe_idx: Sequence of integer token IDs.

        Returns:
            Reconstructed text string.
        """
        tokens_merged = [self.decoder[token] for token in bpe_idx]
        tokens_flat = ''.join(tokens_merged)
        tokens_bytes = bytearray([self.byte_decoder[c] for c in tokens_flat])
        text = tokens_bytes.decode('utf-8', errors='replace')
        return text


def get_file(local_file: str, remote_file: str) -> None:
    """Download file if it doesn't exist locally.

    Args:
        local_file: Local file path.
        remote_file: Remote URL to fetch from.
    """
    if not os.path.isfile(local_file):
        print(f"Downloading {remote_file} to {local_file}")
        response = requests.get(remote_file)
        with open(local_file, "wb") as f:
            f.write(response.content)


def get_encoder() -> Encoder:
    """Get or create cached BPE encoder.

    Returns:
        Initialized BPE Encoder instance.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    encoder_local_file = os.path.join(CACHE_DIR, ENCODER_FILENAME)
    get_file(encoder_local_file, ENCODER_REMOTE)

    with open(encoder_local_file, 'r') as f:
        encoder = json.load(f)

    assert len(encoder) == EXPECTED_ENCODER_SIZE, (
        f"Expected {EXPECTED_ENCODER_SIZE} tokens, got {len(encoder)}"
    )

    vocab_local_file = os.path.join(CACHE_DIR, VOCAB_FILENAME)
    get_file(vocab_local_file, VOCAB_REMOTE)

    with open(vocab_local_file, 'r', encoding="utf-8") as f:
        bpe_data = f.read()

    bpe_merges = [
        cast(BytePair, tuple(merge_str.split()))
        for merge_str in bpe_data.split('\n')[1:-1]
    ]

    assert len(bpe_merges) == EXPECTED_BPE_MERGES, (
        f"Expected {EXPECTED_BPE_MERGES} merges, got {len(bpe_merges)}"
    )

    return Encoder(encoder, bpe_merges)


class BPETokenizer:
    """PyTorch-aware BPE tokenizer matching HuggingFace interface.

    Attributes:
        encoder: Underlying BPE encoder.
    """

    encoder: Encoder

    def __init__(self) -> None:
        self.encoder = get_encoder()

    def __call__(self, text: str, return_tensors: str = 'pt') -> torch.Tensor:
        """Encode text to PyTorch tensor.

        Args:
            text: Input text string.
            return_tensors: Must be 'pt' (PyTorch).

        Returns:
            Tensor of token IDs with batch dimension.
        """
        assert return_tensors == 'pt', "Only PyTorch tensors are supported"
        assert isinstance(text, str), "Single string input expected"
        idx = [self.encoder.encode(text)]
        return torch.tensor(idx, dtype=torch.long)

    def decode(self, idx: torch.Tensor) -> str:
        """Decode token tensor back to text.

        Args:
            idx: 1D tensor of token IDs.

        Returns:
            Decoded text string.
        """
        assert idx.ndim == 1, "Expected 1D tensor"
        return self.encoder.decode(idx.tolist())


if __name__ == '__main__':
    text = "Hello!! I'm Andrej Karpathy. It's 2022. w00t :D 🤗"
    e = get_encoder()
    r = e.encode_and_show_work(text)

    print("Original text is:")
    print(text)
    print("First the text gets pre-tokenized, broken up into chunks:")
    print(r['tokens'])
    print("Then we iterate over each chunk...")
    for part in r['parts']:
        print(part)
    print("Final token indices:")
    print(r['bpe_idx'])
    print("Ready to feed into a Transformer!")
