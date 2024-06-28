import unittest

from rdagent.oai.llm_utils import (
    APIBackend,
    calculate_embedding_distance_between_str_list,
)


class TestEmbedding(unittest.TestCase):
    def test_embedding(self) -> None:
        emb = APIBackend().create_embedding("hello")
        assert emb is not None
        assert isinstance(emb, list)
        assert len(emb) > 0

    def test_embedding_similarity(self) -> None:
        similarity = calculate_embedding_distance_between_str_list(["Hello"], ["Hi"])[0][0]
        assert similarity is not None
        assert isinstance(similarity, float)
        min_similarity_threshold = 0.8
        assert similarity >= min_similarity_threshold


if __name__ == "__main__":
    unittest.main()
