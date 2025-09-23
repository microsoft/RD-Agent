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

    def test_embedding_list(self) -> None:
        emb = APIBackend().create_embedding(["hello", "hi"])
        assert emb is not None
        assert isinstance(emb, list)
        assert len(emb) == 2

    def test_embedding_similarity(self) -> None:
        similarity = calculate_embedding_distance_between_str_list(["Hello"], ["Hi"])[0][0]
        assert similarity is not None
        assert isinstance(similarity, float)
        min_similarity_threshold = 0.8
        assert similarity >= min_similarity_threshold

    def test_embedding_long_text_truncation(self) -> None:
        """Test embedding with very long text that exceeds token limits"""
        # Create a very long text that will definitely exceed embedding token limits
        # Using a repetitive pattern to simulate a real long document
        long_content = (
            """
        This is a very long document that contains a lot of repetitive content to test the embedding truncation functionality.
        We need to make this text long enough to exceed the typical embedding model token limits of around 8192 tokens.
        """
            * 1000
        )  # This should create a text with approximately 50,000+ tokens
        # This should trigger the gradual truncation mechanism
        emb = APIBackend().create_embedding(long_content)

        assert emb is not None
        assert isinstance(emb, list)
        assert len(emb) > 0


if __name__ == "__main__":
    unittest.main()
