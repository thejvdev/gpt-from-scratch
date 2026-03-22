from typing import Callable


def build_vocab(
    data: str,
) -> tuple[int, Callable[[str], list[int]], Callable[[list[int]], str]]:
    chars = sorted(set(data))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [stoi[ch] for ch in s]
    decode = lambda l: "".join(itos[i] for i in l)

    return len(chars), encode, decode
