import math

from src.db.vector import cosine_similarity


class FakeEmbedding:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


def test_cosine_similarity_handles_array_like_embeddings():
    lhs = FakeEmbedding([1.0, 0.0, 0.0])
    rhs = [1.0, 0.0, 0.0]

    score = cosine_similarity(lhs, rhs)

    assert math.isclose(score, 1.0)


def test_cosine_similarity_returns_zero_for_empty_or_mismatched_embeddings():
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([1.0, 2.0], [1.0]) == 0.0
