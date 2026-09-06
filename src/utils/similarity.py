from functools import lru_cache

@lru_cache(maxsize=1)
def _get_sbert_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_sentence_similarity(new_sent, sentences, threshold=0.9):
    if not sentences:
        return 0

    from sentence_transformers import util

    model = _get_sbert_model()
    embeddings = model.encode([new_sent, *sentences], convert_to_tensor=True)
    similarities = util.cos_sim(embeddings[:1], embeddings[1:])
    return int((similarities > threshold).sum().item())