from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from rdagent.core.log import RDAgentLog
from rdagent.oai.llm_utils import APIBackend
from scipy.spatial.distance import cosine


class KnowledgeMetaData:
    def __init__(self,
                 content: str = "",
                 label: str | None = None,
                 embedding: Any = None,
                 identity: Any = None,
                ) -> None:
        self.label = label
        self.content = content
        self.id = str(uuid.uuid3(uuid.NAMESPACE_DNS, str(self.content))) if identity is None else identity
        self.embedding = embedding
        self.trunks = []
        self.trunks_embedding = []

    def split_into_trunk(self, size: int = 1000) -> None:
        """
        split content into trunks and create embedding by trunk
        Returns
        -------

        """

        def split_string_into_chunks(string: str, chunk_size: int) -> list[str]:
            chunks = []
            for i in range(0, len(string), chunk_size):
                chunk = string[i : i + chunk_size]
                chunks.append(chunk)
            return chunks

        self.trunks = split_string_into_chunks(self.content, chunk_size=size)
        self.trunks_embedding = APIBackend().create_embedding(input_content=self.trunks)

    def create_embedding(self) -> None:
        """
        create content's embedding
        Returns
        -------

        """
        if self.embedding is None:
            self.embedding = APIBackend().create_embedding(input_content=self.content)

    def from_dict(self, data: dict) -> KnowledgeMetaData:
        for key, value in data.items():
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        return f"Document(id={self.id}, label={self.label}, data={self.content})"


Document = KnowledgeMetaData


def contents_to_documents(contents: list[str], label: str | None = None) -> list[Document]:
    # openai create embedding API input's max length is 16
    size = 16
    embedding = []
    for i in range(0, len(contents), size):
        embedding.extend(APIBackend().create_embedding(input_content=contents[i : i + size]))
    return [Document(content=c, label=label, embedding=e) for c, e in zip(contents, embedding)]


class VectorBase:
    """
    This class is used for handling vector storage and query
    """

    def __init__(self, vector_df_path: str | Path | None = None, **kwargs: dict) -> None:
        pass

    def add(self, document: Document | list[Document]) -> None:
        """
        add new node to vector_df
        Parameters
        ----------
        document

        Returns
        -------

        """

    def search(self, content: str, topk_k: int = 5, similarity_threshold: float = 0) -> list[Document]:
        """
        search vector_df by node
        Parameters
        ----------
        similarity_threshold
        content
        topk_k: return topk_k nearest vector_df

        Returns
        -------

        """

    def load(self, **kwargs: dict) -> None:
        """load vector_df"""

    def save(self, **kwargs: dict) -> None:
        """save vector_df"""


class PDVectorBase(VectorBase):
    """
    Implement of VectorBase using Pandas
    """

    def __init__(self, vector_df_path: str | Path | None = None) -> None:
        super().__init__(vector_df_path)

        if vector_df_path:
            try:
                self.vector_df = self.load(vector_df_path)
            except FileNotFoundError:
                self.vector_df = pd.DataFrame(columns=["id", "label", "content", "embedding"])
        else:
            self.vector_df = pd.DataFrame(columns=["id", "label", "content", "embedding"])

        RDAgentLog().info(f"VectorBase loaded, shape={self.vector_df.shape}")

    def shape(self) -> pd.DataFrame.shape:
        return self.vector_df.shape

    def add(self, document: Document | list[Document]) -> None:
        """
        add new node to vector_df
        Parameters
        ----------
        document

        Returns
        -------

        """
        if isinstance(document, Document):
            if document.embedding is None:
                document.create_embedding()
            docs = [
                {
                    "id": document.id,
                    "label": document.label,
                    "content": document.content,
                    "trunk": document.content,
                    "embedding": document.embedding,
                },
            ]
            docs.extend(
                [
                    {
                        "id": document.id,
                        "label": document.label,
                        "content": document.content,
                        "trunk": trunk,
                        "embedding": embedding,
                    }
                    for trunk, embedding in zip(document.trunks, document.trunks_embedding)
                ],
            )
            self.vector_df = pd.concat([self.vector_df, pd.DataFrame(docs)], ignore_index=True)
        else:
            for doc in document:
                self.add(document=doc)

    def search(self,
               content: str,
               topk_k: int = 5,
               similarity_threshold: float = 0,
               ) -> tuple[list[Document], list[float]]:
        """
        search vector by node
        Parameters
        ----------
        similarity_threshold
        content
        topk_k: return topk_k nearest vector

        Returns
        -------

        """
        if not self.vector_df.shape[0]:
            return [], []
        document = Document(content=content)
        document.create_embedding()
        similarities = self.vector_df["embedding"].apply(
            lambda x: 1 - cosine(x, document.embedding),
        )  # cosine is cosine distance, 1-similarity
        searched_similarities = similarities[similarities > similarity_threshold].nlargest(topk_k)
        most_similar_docs = self.vector_df.loc[searched_similarities.index]
        docs = []
        for _, similar_docs in most_similar_docs.iterrows():
            docs.append(Document().from_dict(similar_docs.to_dict()))
        return docs, searched_similarities.to_list()

    def load(self, vector_df_path: str | Path) -> pd.DataFrame:
        return pd.read_pickle(vector_df_path)


    def save(self, vector_df_path: str | Path) -> None:
        self.vector_df.to_pickle(vector_df_path)
        RDAgentLog().info(f"Save vectorBase vector_df to: {vector_df_path}")
