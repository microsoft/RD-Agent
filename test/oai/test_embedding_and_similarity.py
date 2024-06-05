import pickle
import unittest
from pathlib import Path
import json
import random

from rdagent.oai.llm_utils import APIBackend, calculate_embedding_distance_between_str_list


class TestEmbedding(unittest.TestCase):
    def test_embedding(self):
        emb = APIBackend().create_embedding("hello")
        assert emb is not None
        assert type(emb) == list
        assert len(emb) > 0

    def test_embedding_similarity(self):
        similarity = calculate_embedding_distance_between_str_list(["Hello"], ["Hi"])[0][0]
        assert similarity is not None
        assert type(similarity) == float
        assert similarity >= 0.8


if __name__ == "__main__":
    unittest.main()
