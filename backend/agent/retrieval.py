"""
Vector retrieval over stored chunk embeddings.
Uses cosine similarity computed in NumPy; no external vector store required.
"""
import numpy as np
from database import get_connection
from ingestion.embeddings import deserialize_embedding, embed_query


TOP_K = 5


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def retrieve_chunks(query: str, manual_id: int, top_k: int = TOP_K) -> list[dict]:
    """
    Embed the query, then rank all stored chunks for the given manual by
    cosine similarity. Returns the top_k results as a list of dicts.
    """
    query_vec = embed_query(query)

    conn = get_connection()
    rows = conn.execute(
        "SELECT id, page_number, chunk_index, text, embedding, section_header "
        "FROM chunks WHERE manual_id = ? AND embedding IS NOT NULL",
        (manual_id,),
    ).fetchall()
    conn.close()

    scored: list[tuple[float, dict]] = []
    for row in rows:
        chunk_vec = deserialize_embedding(row["embedding"])
        sim = cosine_similarity(query_vec, chunk_vec)
        scored.append((sim, {
            "chunk_id": row["id"],
            "page_number": row["page_number"],
            "text": row["text"],
            "section_header": row["section_header"],
            "similarity": sim,
        }))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:top_k]]
