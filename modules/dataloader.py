import mlx.core as mx


def get_batch(
    data: mx.array, batch_size: int, seq_len: int
) -> tuple[mx.array, mx.array]:
    indices = mx.random.randint(0, len(data) - seq_len, (batch_size,))
    indices = indices.tolist()

    X = mx.stack([data[i : i + seq_len] for i in indices])
    y = mx.stack([data[i + 1 : i + seq_len + 1] for i in indices])

    return X, y
