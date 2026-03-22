import math
from typing import Callable
import mlx.core as mx
import mlx.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int):
        super().__init__()

        pe = mx.zeros((seq_len, d_model))
        positions = mx.arange(0, seq_len, 1, dtype=mx.float32)[:, None]

        div_term = mx.exp(
            mx.arange(0, d_model, 2, dtype=mx.float32) * -(math.log(10000.0) / d_model)
        )
        pe[:, ::2] = mx.sin(positions * div_term)
        pe[:, 1::2] = mx.cos(positions * div_term)

        self.pe = pe[None, :, :]

    def __call__(self, X: mx.array):
        return X + self.pe[:, : X.shape[1], :]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.dropout = nn.Dropout(p=0.1)
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

    def __call__(self, X: mx.array):
        batch_size, seq_len, d_model = X.shape

        QKV = self.qkv_proj(X)
        Q, K, V = mx.split(QKV, 3, axis=-1)

        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_head).transpose(
            0, 2, 1, 3
        )
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_head).transpose(
            0, 2, 1, 3
        )
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_head).transpose(
            0, 2, 1, 3
        )

        scores = (Q @ K.transpose(0, 1, 3, 2)) / math.sqrt(self.d_head)

        causal_mask = mx.triu(mx.ones((seq_len, seq_len)), k=1)
        scores = mx.where(causal_mask == 1, -1e9, scores)

        weights = mx.softmax(scores, axis=-1)
        weights = self.dropout(weights)

        context = (
            (weights @ V).transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        )

        return self.out_proj(context)


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        d_ff = d_model * 4
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def __call__(self, X: mx.array):
        return self.ffn(X)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        self.dropout = nn.Dropout(p=0.1)
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model)

    def __call__(self, X: mx.array):
        X_attn = self.mha(self.ln1(X))
        X = X + self.dropout(X_attn)
        X_ffn = self.ffn(self.ln2(X))
        X = X + self.dropout(X_ffn)
        return X


class GPT(nn.Module):
    def __init__(
        self, vocab_size: int, seq_len: int, d_model: int, num_heads: int, n_layers: int
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(seq_len, d_model)
        self.dropout = nn.Dropout(p=0.1)

        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model, num_heads) for _ in range(n_layers)]
        )

        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def __call__(self, X: mx.array):
        X = self.embedding(X)
        X = self.pe(X)
        X = self.dropout(X)
        X = self.blocks(X)
        X = self.head(self.ln(X))
        return X

    def generate(
        self,
        prompt: str,
        encode: Callable[[str], list[int]],
        decode: Callable[[list[int]], str],
        max_len: int,
        temperature: float = 0.0,
    ):
        tokens = encode(prompt)
        input_ids = mx.array(tokens)[None, :]

        for _ in range(max_len):
            logits = self(input_ids)
            last_logits = logits[:, -1, :]

            if temperature == 0.0:
                next_id = mx.argmax(last_logits, axis=-1, keepdims=True)
            else:
                next_id = mx.random.categorical(last_logits / temperature, axis=-1)
                next_id = next_id[:, None]

            input_ids = mx.concatenate([input_ids, next_id], axis=1)

            new_token = decode([next_id.item()])
            yield new_token
