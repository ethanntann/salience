"""Turn a summary sentence into a vector the taste model can rank on.

Two backends, and the choice is deliberate rather than incidental:

``tfidf``
    sklearn, no download, instant. It matches on shared *words*, so "pistol" and
    "handgun" are unrelated to it. That is a real weakness, and it is exactly
    why it makes a good baseline: any neural encoder has to beat a method this
    naive before its cost is justified.

``sentence-transformers``
    Semantic, ~30MB, CPU, milliseconds per sentence. Used automatically when
    installed.

One property of TF-IDF worth knowing before it surprises you: vectors depend on
the corpus, so adding clips changes every existing vector. Stored vectors are
tagged with the model name and must be rebuilt together, never mixed.
"""

from __future__ import annotations

from importlib.util import find_spec


class TfidfEncoder:
    """Bag-of-words baseline. Fit on the corpus, then transform it."""

    name = "tfidf-lsa"

    def __init__(self) -> None:
        self._vectorizer = None
        self._svd = None

    def fit_transform(self, texts: list[str]) -> list[list[float]]:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Sub-linear term frequency and unigrams+bigrams: summaries are short and
        # repetitive ("securing an elimination" appears constantly), so bigrams
        # carry most of what distinguishes one clip from another.
        self._vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
        )
        if not texts:
            return []
        try:
            matrix = self._vectorizer.fit_transform(texts)
        except ValueError as exc:
            if "empty vocabulary" not in str(exc):
                raise
            return [[0.0] for _ in texts]
        if min(matrix.shape) < 3:
            return matrix.toarray().tolist()

        # Reduce before ranking. Raw TF-IDF gave ~500 dimensions over 24
        # documents -- more dimensions than examples, where a centroid direction
        # fits noise perfectly and generalises to nothing. LSA to a few dozen
        # dense dimensions also lets words that co-occur ("pistol", "handgun")
        # land near each other, which raw term matching cannot do at all.
        components = min(48, matrix.shape[0] - 1, matrix.shape[1] - 1)
        self._svd = TruncatedSVD(n_components=components, random_state=0)
        return self._svd.fit_transform(matrix).tolist()

    def transform(self, texts: list[str]) -> list[list[float]]:
        """Project unseen summaries without fitting on the evaluation set."""
        if self._vectorizer is None or not hasattr(self._vectorizer, "vocabulary_"):
            raise ValueError("Fit a nonempty training vocabulary before transforming")
        matrix = self._vectorizer.transform(texts)
        return (
            self._svd.transform(matrix).tolist()
            if self._svd is not None
            else matrix.toarray().tolist()
        )


class SentenceTransformerEncoder:
    name = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.name)

    def fit_transform(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.encode(texts)]

    def transform(self, texts: list[str]) -> list[list[float]]:
        """Identical to fit_transform: this encoder has no corpus state.

        It exists so both backends satisfy one interface. Without it, anything
        holding out a test set -- `evaluation.rank` does -- worked only because
        it happened to hardcode the TF-IDF baseline.
        """
        return self.fit_transform(texts)


def load_encoder(prefer: str | None = None):
    if prefer == "tfidf":
        return TfidfEncoder()
    if find_spec("sentence_transformers") is not None:
        return SentenceTransformerEncoder()
    return TfidfEncoder()


def cosine(left: list[float], right: list[float]) -> float:
    import numpy as np

    a, b = np.asarray(left), np.asarray(right)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denominator) if denominator else 0.0


class ClipImageEncoder:
    """CLIP image embeddings, mean-pooled over a clip's frame window.

    Motivation, from measurement rather than taste: v3.1-exec showed that adding
    more *prose* does not improve within-fight ranking. Everything the taste
    model sees passes through ~40 words of English, which is a narrow pipe for
    "which of these two near-identical kills is better". This bypasses the text
    entirely and ranks on pixels.

    Mean-pooling the seven frames is the crude choice. It discards ordering, so
    it cannot represent "missed, adjusted, then hit" -- but it is the cheapest
    thing that could work, and if it beats text the next question is worth asking.
    """

    name = "clip-ViT-B-32"

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.name)

    def encode_frames(self, frame_paths: list) -> list[float]:
        from PIL import Image
        import numpy as np

        images = []
        for path in frame_paths:
            try:
                images.append(Image.open(path).convert("RGB"))
            except OSError:
                continue
        if not images:
            return []
        vectors = self._model.encode(images)
        return np.asarray(vectors, dtype=float).mean(axis=0).tolist()
