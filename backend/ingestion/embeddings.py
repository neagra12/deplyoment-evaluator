"""
OpenAI text-embedding-3-small wrapper.
Batches requests to stay within token limits and returns float32 numpy arrays.

Future consideration: swap the OpenAI call for a local model such as
nomic-embed-text via sentence-transformers for zero-cost offline usage.
"""
import os
import struct
import numpy as np
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100  # max texts per API call


def _get_client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def embed_texts(texts: list[str]) -> list[np.ndarray]:
    """Return a list of float32 numpy arrays, one per input text."""
    client = _get_client()
    results: list[np.ndarray] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        for item in response.data:
            arr = np.array(item.embedding, dtype=np.float32)
            results.append(arr)

    return results


def embed_query(text: str) -> np.ndarray:
    """Embed a single query string."""
    return embed_texts([text])[0]


def serialize_embedding(arr: np.ndarray) -> bytes:
    """Pack a float32 numpy array to bytes for SQLite BLOB storage."""
    return arr.tobytes()


def deserialize_embedding(blob: bytes) -> np.ndarray:
    """Unpack bytes from SQLite back to a float32 numpy array."""
    return np.frombuffer(blob, dtype=np.float32)
