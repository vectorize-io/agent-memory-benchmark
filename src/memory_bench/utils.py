import functools

import tiktoken

CHUNK_SIZE = 512  # tokens


@functools.lru_cache(maxsize=1)
def _enc():
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc().encode(text, disallowed_special=()))


def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """Split text into token-sized chunks, decoded back to strings."""
    enc = _enc()
    tokens = enc.encode(text, disallowed_special=())
    if len(tokens) <= size:
        return [text]
    return [enc.decode(tokens[i:i + size]) for i in range(0, len(tokens), size)]
