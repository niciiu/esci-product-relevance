"""
TF-IDF feature extraction for ESCI lexical classification.

This module is responsible only for TF-IDF vectorization.

Pipeline:
    Cleaned text
        |
    TF-IDF Vectorizer
        |
    Sparse feature matrix

Important (no-leakage guarantee):
    - TF-IDF must be fitted on TRAINING data only (fit / fit_transform).
    - Validation and test data must only ever be transformed, never
      used to (re-)fit the vocabulary or IDF weights.
    - This module does NOT train the classifier.
"""

from dataclasses import dataclass

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class TfidfConfig:
    max_features: int | None = None
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int | float = 1
    max_df: int | float = 1.0
    sublinear_tf: bool = True


class TfidfFeatureExtractor:
    """TF-IDF feature extractor for lexical experiments.

    The vectorizer is fitted only once, on training text. Validation and
    test text are transformed using the already-fitted vectorizer.
    """

    def __init__(self, config: TfidfConfig | None = None):
        self.config = config or TfidfConfig()
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            sublinear_tf=self.config.sublinear_tf,
        )
        self._is_fitted = False

    def fit(self, texts) -> "TfidfFeatureExtractor":
        """Fit TF-IDF vocabulary and IDF values on training text only."""
        self.vectorizer.fit(texts)
        self._is_fitted = True
        return self

    def transform(self, texts) -> csr_matrix:
        """Transform text using the already-fitted TF-IDF vectorizer."""
        if not self._is_fitted:
            raise RuntimeError("TF-IDF vectorizer has not been fitted. Call fit() using training data first.")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts) -> csr_matrix:
        """Fit the vectorizer and transform training text. TRAINING DATA ONLY."""
        features = self.vectorizer.fit_transform(texts)
        self._is_fitted = True
        return features

    @property
    def vocabulary_size(self) -> int:
        if not self._is_fitted:
            raise RuntimeError("TF-IDF vectorizer has not been fitted.")
        return len(self.vectorizer.vocabulary_)

    def get_feature_names(self) -> list[str]:
        if not self._is_fitted:
            raise RuntimeError("TF-IDF vectorizer has not been fitted.")
        return self.vectorizer.get_feature_names_out().tolist()
