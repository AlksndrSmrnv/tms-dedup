from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbedder:
    """Character + word n-gram TF-IDF. Robust to small wording differences, Russian-friendly."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )

    def fit_transform(self, texts: list[str]):
        return self.vectorizer.fit_transform(texts)
