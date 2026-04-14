from .tfidf import TfidfEmbedder

EMBEDDERS = {"tfidf": TfidfEmbedder}


def get_embedder(name: str):
    if name not in EMBEDDERS:
        raise ValueError(f"unknown embedder: {name}. available: {list(EMBEDDERS)}")
    return EMBEDDERS[name]()
