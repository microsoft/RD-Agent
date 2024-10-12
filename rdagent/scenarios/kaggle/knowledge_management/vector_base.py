import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Union

import pandas as pd
from jinja2 import Environment, StrictUndefined

from rdagent.components.knowledge_management.vector_base import Document, PDVectorBase
from rdagent.core.prompts import Prompts
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.kaggle.knowledge_management.extract_knowledge import (
    extract_knowledge_from_feedback,
)


class KGKnowledgeDocument(Document):
    """
    Class for handling Kaggle competition specific metadata
    """

    def __init__(
        self,
        content: str = "",
        label: str = None,
        embedding=None,
        identity=None,
        competition_name=None,
        task_category=None,
        field=None,
        ranking=None,
        score=None,
        entities=None,
        relations=None,
    ):
        """
        Initialize KGKnowledgeMetaData for Kaggle competition posts

        Parameters:
        ----------
        competition_name: str, optional
            The name of the Kaggle competition.
        task_category: str, required
            The type of task (e.g., classification, regression).
        field: str, optional
            The specific field of knowledge (e.g., feature engineering, modeling).
        ranking: str or int, optional
            The ranking achieved in the competition.
        score: float, optional
            The score or metric achieved in the competition.
        entities: list, optional
            Entities related to the content (for knowledge graph integration).
        relations: list, optional
            Relations between entities (for knowledge graph integration).
        """
        super().__init__(content, label, embedding, identity)
        self.competition_name = competition_name
        self.task_category = task_category  # Task type is required
        self.field = field  # Knowledge field, optional (model/data/others/overall)
        self.ranking = ranking  # Ranking
        # TODO ranking and score might be unified
        self.score = score  # Competition score
        # TODO Perhaps this shouldn't be here?
        self.entities = entities or []  # Entities in the knowledge graph
        self.relations = relations or []  # Relations in the knowledge graph

    def split_into_trunk(self, size: int = 1000, overlap: int = 0):
        """
        Split content into trunks and create embeddings by trunk
        #TODO let GPT do the split based on the field of knowledge(data/model/others)
        """

        def split_string_into_chunks(string: str, chunk_size: int):
            chunks = []
            for i in range(0, len(string), chunk_size):
                chunk = string[i : i + chunk_size]
                chunks.append(chunk)
            return chunks

        self.trunks = split_string_into_chunks(self.content, chunk_size=size)
        self.trunks_embedding = APIBackend().create_embedding(input_content=self.trunks)

    def from_dict(self, data: dict):
        """
        Load Kaggle post data from a dictionary
        """
        super().from_dict(data)
        self.competition_name = data.get("competition_name", None)
        self.task_category = data.get("task_category", None)
        self.field = data.get("field", None)
        self.ranking = data.get("ranking", None)
        self.score = data.get("score", None)
        self.entities = data.get("entities", [])
        self.relations = data.get("relations", [])
        return self

    def __repr__(self):
        return (
            f"KGKnowledgeMetaData(id={self.id}, label={self.label}, competition={self.competition_name}, "
            f"task_category={self.task_category}, field={self.field}, ranking={self.ranking}, score={self.score})"
        )


KGDocument = KGKnowledgeDocument


class KaggleExperienceBase(PDVectorBase):
    """
    Class for handling Kaggle competition experience posts and organizing them for reference
    """

    def __init__(self, vector_df_path: Union[str, Path] = None, kaggle_experience_path: Union[str, Path] = None):
        """
        Initialize the KaggleExperienceBase class

        Parameters:
        ----------
        vector_df_path: str or Path, optional
            Path to the vector DataFrame for embedding management.
        kaggle_experience_path: str or Path, optional
            Path to the Kaggle experience post data.
        """
        super().__init__(vector_df_path)
        self.kaggle_experience_path = kaggle_experience_path
        self.kaggle_experience_data = []
        if kaggle_experience_path:
            self.load_kaggle_experience(kaggle_experience_path)

    def add(self, document: Union[KGDocument, List[KGDocument]]):
        document.split_into_trunk()
        docs = [
            {
                "id": document.id,
                "label": document.label,
                "content": document.content,
                "competition_name": document.competition_name,
                "task_category": document.task_category,
                "field": document.field,
                "ranking": document.ranking,
                "score": document.score,
                "embedding": document.embedding,
            }
        ]
        if len(document.trunks) > 1:
            docs.extend(
                [
                    {
                        "id": document.id,
                        "label": document.label,
                        "content": document.content,
                        "competition_name": document.competition_name,
                        "task_category": document.task_category,
                        "field": document.field,
                        "ranking": document.ranking,
                        "score": document.score,
                        "embedding": trunk_embedding,
                    }
                    for trunk, trunk_embedding in zip(document.trunks, document.trunks_embedding)
                ]
            )
        self.vector_df = pd.concat([self.vector_df, pd.DataFrame(docs)], ignore_index=True)

    def load_kaggle_experience(self, kaggle_experience_path: Union[str, Path]):
        """
        Load Kaggle experience posts from a JSON or text file

        Parameters:
        ----------
        kaggle_experience_path: str or Path
            Path to the Kaggle experience post data.
        """
        try:
            with open(kaggle_experience_path, "r", encoding="utf-8") as file:
                self.kaggle_experience_data = json.load(file)
            logger.info(f"Kaggle experience data loaded from {kaggle_experience_path}")
        except FileNotFoundError:
            logger.error(f"Kaggle experience data not found at {kaggle_experience_path}")
            self.kaggle_experience_data = []

    def add_experience_to_vector_base(self, experiment_feedback=None):
        """
        Process Kaggle experience data or experiment feedback and add relevant information to the vector base.

        Args:
            experiment_feedback (dict, optional): A dictionary containing experiment feedback.
                                                If provided, this feedback will be processed and added to the vector base.
        """
        # If experiment feedback is provided, extract relevant knowledge and add it to the vector base
        if experiment_feedback:
            extracted_knowledge = extract_knowledge_from_feedback(experiment_feedback)

            document = KGKnowledgeDocument(
                content=experiment_feedback.get("hypothesis_text", ""),
                label="Experiment Feedback",
                competition_name="Experiment Result",
                task_category=experiment_feedback.get("tasks_factors", "General Task"),
                field="Research Feedback",
                ranking=None,
                score=experiment_feedback.get("current_result", None),
            )
            document.create_embedding()
            self.add(document)
            return

        # Process Kaggle experience data
        logger.info(f"Processing {len(self.kaggle_experience_data)} Kaggle experience posts")
        for experience in self.kaggle_experience_data:
            logger.info(f"Processing experience index: {self.kaggle_experience_data.index(experience)}")
            content = experience.get("content", "")
            label = experience.get("title", "Kaggle Experience")
            competition_name = experience.get("competition_name", "Unknown Competition")
            task_category = experience.get("task_category", "General Task")
            field = experience.get("field", None)
            ranking = experience.get("ranking", None)
            score = experience.get("score", None)

            document = KGKnowledgeDocument(
                content=content,
                label=label,
                competition_name=competition_name,
                task_category=task_category,
                field=field,
                ranking=ranking,
                score=score,
            )
            document.create_embedding()
            self.add(document)

    def search_experience(self, target: str, query: str, topk_k: int = 5, similarity_threshold: float = 0.1):
        """
        Search for Kaggle experience posts related to the query, initially filtered by the target.

        Parameters:
        ----------
        target: str
            The target context to refine the search query.
        query: str
            The search query to find relevant experience posts.
        topk_k: int, optional
            Number of top similar results to return (default is 5).
        similarity_threshold: float, optional
            The similarity threshold for filtering results (default is 0.1).

        Returns:
        -------
        List[KGKnowledgeMetaData], List[float]:
            A list of the most relevant documents and their similarities.
        """

        # Modify the query to include the target
        modified_query = f"The target is {target}. And I need you to query {query} based on the {target}."

        # First, search based on the modified query
        search_results, similarities = super().search(
            modified_query, topk_k=topk_k, similarity_threshold=similarity_threshold
        )

        # If the results do not match the target well, refine the search using LLM or further adjustment
        kaggle_docs = []
        for result in search_results:
            kg_doc = KGKnowledgeDocument().from_dict(result.__dict__)

            gpt_feedback = self.refine_with_LLM(target, kg_doc)
            if gpt_feedback:
                kg_doc.content = gpt_feedback

            kaggle_docs.append(kg_doc)

        return kaggle_docs, similarities

    def refine_with_LLM(self, target: str, text: str) -> str:
        prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")

        sys_prompt = (
            Environment(undefined=StrictUndefined).from_string(prompt_dict["refine_with_LLM"]["system"]).render()
        )

        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["refine_with_LLM"]["user"])
            .render(target=target, text=text)
        )

        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=False,
        )

        return response

    def save(self, vector_df_path: Union[str, Path]):
        """
        Save the vector DataFrame to a file

        Parameters:
        ----------
        vector_df_path: str or Path
            Path to save the vector DataFrame.
        """
        self.vector_df.to_pickle(vector_df_path)
        logger.info(f"Vector DataFrame saved to {vector_df_path}")


if __name__ == "__main__":
    kaggle_base = KaggleExperienceBase(
        kaggle_experience_path="git_ignore_folder/data_minicase/kaggle_experience_results.json"
    )

    kaggle_base.add_experience_to_vector_base()

    kaggle_base.save("git_ignore_folder/vector_base/kaggle_vector_base.pkl")

    print(f"There are {kaggle_base.shape()[0]} records in the vector base.")

    search_results, similarities = kaggle_base.search_experience(query="image classification", topk_k=3)

    for result, similarity in zip(search_results, similarities):
        print(
            f"Competition name: {result.competition_name}, task_category: {result.task_category}, score: {result.score}, similarity: {similarity}"
        )
