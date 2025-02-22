from __future__ import annotations

import pickle
import random
from collections import deque
from pathlib import Path
from typing import Any, NoReturn

from rdagent.components.knowledge_management.vector_base import (
    KnowledgeMetaData,
    PDVectorBase,
    VectorBase,
    cosine,
)
from rdagent.core.knowledge_base import KnowledgeBase
from rdagent.oai.llm_utils import APIBackend

Node = KnowledgeMetaData


class UndirectedNode(Node):
    def __init__(self, content: str = "", label: str = "", embedding: Any = None) -> None:
        super().__init__(content, label, embedding)
        self.neighbors: set[UndirectedNode] = set()
        assert isinstance(content, str), "content must be a string"

    def add_neighbor(self, node: UndirectedNode) -> None:
        self.neighbors.add(node)
        node.neighbors.add(self)

    def remove_neighbor(self, node: UndirectedNode) -> None:
        if node in self.neighbors:
            self.neighbors.remove(node)
            node.neighbors.remove(self)

    def get_neighbors(self) -> set[UndirectedNode]:
        return self.neighbors

    def __str__(self) -> str:
        return (
            f"UndirectedNode(id={self.id}, label={self.label}, content={self.content[:100]}, "
            f"neighbors={self.neighbors})"
        )

    def __repr__(self) -> str:
        return (
            f"UndirectedNode(id={self.id}, label={self.label}, content={self.content[:100]}, "
            f"neighbors={self.neighbors})"
        )


class Graph(KnowledgeBase):
    """
    base Graph class for Knowledge Graph Search
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.nodes = {}
        super().__init__(path=path)

    def size(self) -> int:
        return len(self.nodes)

    def get_node(self, node_id: str) -> Node | None:
        return self.nodes.get(node_id)

    def add_node(self, **kwargs: Any) -> NoReturn:
        raise NotImplementedError

    def get_all_nodes(self) -> list[Node]:
        return list(self.nodes.values())

    def get_all_nodes_by_label_list(self, label_list: list[str]) -> list[Node]:
        return [node for node in self.nodes.values() if node.label in label_list]

    def find_node(self, content: str, label: str) -> Node | None:
        for node in self.nodes.values():
            if node.content == content and node.label == label:
                return node
        return None

    @staticmethod
    def batch_embedding(nodes: list[Node]) -> list[Node]:
        contents = [node.content for node in nodes]
        # openai create embedding API input's max length is 16
        size = 16
        embeddings = []
        for i in range(0, len(contents), size):
            embeddings.extend(
                APIBackend().create_embedding(input_content=contents[i : i + size]),
            )

        assert len(nodes) == len(embeddings), "nodes' length must equals embeddings' length"
        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding
        return nodes

    def __str__(self) -> str:
        return f"Graph(nodes={self.nodes})"


class UndirectedGraph(Graph):
    """
    Undirected Graph which edges have no relationship
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.vector_base: VectorBase = PDVectorBase()
        super().__init__(path=path)

    def __str__(self) -> str:
        return f"UndirectedGraph(nodes={self.nodes})"

    def add_node(
        self,
        node: UndirectedNode,
        neighbor: UndirectedNode = None,
        same_node_threshold: float = 0.95,  # noqa: ARG002
    ) -> None:
        """
        add node and neighbor to the Graph
        Parameters
        ----------
        same_node_threshold: 0.95 is an empirical value. When two strings only differ in case, the similarity is greater
         than 0.95.
        node
        neighbor

        Returns
        -------

        """
        if self.get_node(node.id):
            node = self.get_node(node.id)
        elif self.find_node(content=node.content, label=node.label):
            node = self.find_node(content=node.content, label=node.label)
        else:
            # same_node = self.semantic_search(node=node.content, similarity_threshold=same_node_threshold, topk_k=1)
            # if len(same_node):
            #     node = same_node[0]
            # else:
            node.create_embedding()
            self.vector_base.add(document=node)
            self.nodes.update({node.id: node})

        if neighbor is not None:
            if self.get_node(neighbor.id):
                neighbor = self.get_node(neighbor.id)
            elif self.find_node(content=neighbor.content, label=node.label):
                neighbor = self.find_node(content=neighbor.content, label=node.label)
            else:
                # same_node = self.semantic_search(node=neighbor.content,
                #                                  similarity_threshold=same_node_threshold, topk_k=1)
                # if len(same_node):
                #     neighbor = same_node[0]
                # else:
                neighbor.create_embedding()
                self.vector_base.add(document=neighbor)
                self.nodes.update({neighbor.id: neighbor})

            node.add_neighbor(neighbor)

    def add_nodes(self, node: UndirectedNode, neighbors: list[UndirectedNode]) -> None:
        if not neighbors:
            self.add_node(node)
        else:
            for neighbor in neighbors:
                self.add_node(node, neighbor=neighbor)

    def get_node(self, node_id: str) -> UndirectedNode:
        return self.nodes.get(node_id)

    def get_node_by_content(self, content: str) -> UndirectedNode | None:
        """
        Get node by semantic distance
        Parameters
        ----------
        content

        Returns
        -------

        """
        if content == "Model":
            pass
        match = self.semantic_search(node=content, similarity_threshold=0.999)
        if match:
            return match[0]
        return None

    def get_nodes_within_steps(
        self,
        start_node: UndirectedNode,
        steps: int = 1,
        constraint_labels: list[str] | None = None,
        *,
        block: bool = False,
    ) -> list[UndirectedNode]:
        """
        Returns the nodes in the graph whose distance from node is less than or equal to step
        """
        visited = set()
        queue = deque([(start_node, 0)])
        result = []

        while queue:
            node, current_steps = queue.popleft()

            if current_steps > steps:
                break

            if node not in visited:
                visited.add(node)
                result.append(node)

                for neighbor in sorted(
                    self.get_node(node.id).neighbors,
                    key=lambda x: x.content,
                ):  # to make sure the result is deterministic
                    if neighbor not in visited and not (block and neighbor.label not in constraint_labels):
                        queue.append((neighbor, current_steps + 1))

        if constraint_labels:
            result = [node for node in result if node.label in constraint_labels]
        if start_node in result:
            result.remove(start_node)
        return result

    def get_nodes_intersection(
        self,
        nodes: list[UndirectedNode],
        steps: int = 1,
        constraint_labels: list[str] | None = None,
    ) -> list[UndirectedNode]:
        """
        Get the intersection with nodes connected within n steps of nodes

        Parameters
        ----------
        nodes
        steps
        constraint_labels

        Returns
        -------

        """
        min_nodes_count = 2
        assert len(nodes) >= min_nodes_count, "nodes length must >=2"
        intersection = None

        for node in nodes:
            if intersection is None:
                intersection = self.get_nodes_within_steps(
                    node,
                    steps=steps,
                    constraint_labels=constraint_labels,
                )
            intersection = self.intersection(
                nodes1=intersection,
                nodes2=self.get_nodes_within_steps(
                    node,
                    steps=steps,
                    constraint_labels=constraint_labels,
                ),
            )
        return intersection

    def semantic_search(
        self,
        node: UndirectedNode | str,
        similarity_threshold: float = 0.0,
        topk_k: int = 5,
    ) -> list[UndirectedNode]:
        """
        semantic search by node's embedding

        Parameters
        ----------
        topk_k
        node
        similarity_threshold: Returns nodes whose distance score from the input
            node is greater than similarity_threshold

        Returns
        -------

        """
        if isinstance(node, str):
            node = UndirectedNode(content=node)
        docs, scores = self.vector_base.search(
            content=node.content,
            topk_k=topk_k,
            similarity_threshold=similarity_threshold,
        )
        return [self.get_node(doc.id) for doc in docs]

    def clear(self) -> None:
        self.nodes.clear()
        self.vector_base: VectorBase = PDVectorBase()

    def query_by_node(
        self,
        node: UndirectedNode,
        step: int = 1,
        constraint_labels: list[str] | None = None,
        constraint_node: UndirectedNode | None = None,
        constraint_distance: float = 0,
        *,
        block: bool = False,
    ) -> list[UndirectedNode]:
        """
        search graph by connection, return empty list if nodes' chain without node near to constraint_node
        Parameters
        ----------
        node
        step
        constraint_labels
        constraint_node
        constraint_distance
        block: despite the start node, the search can only flow through the constraint_label type nodes

        Returns
        -------

        """
        nodes = self.get_nodes_within_steps(
            start_node=node,
            steps=step,
            constraint_labels=constraint_labels,
            block=block,
        )
        if constraint_node is not None:
            for n in nodes:
                if self.cal_distance(n, constraint_node) > constraint_distance:
                    return nodes
            return []
        return nodes

    def query_by_content(
        self,
        content: str | list[str],
        topk_k: int = 5,
        step: int = 1,
        constraint_labels: list[str] | None = None,
        constraint_node: UndirectedNode | None = None,
        similarity_threshold: float = 0.0,
        constraint_distance: float = 0,
        *,
        block: bool = False,
    ) -> list[UndirectedNode]:
        """
        Search graph by content similarity and connection relationship, return empty
        list if nodes' chain without node near to constraint_node.

        Parameters
        ----------
        constraint_distance : float
            The distance between the node and the constraint_node.
        content : Union[str, List[str]]
            Content to search for.
        topk_k: int
            The upper number of output for each query. If the number of fit nodes is
            less than topk_k, returns all fit nodes' content.
        step : int
            The maximum distance between the start node and the result node.
        constraint_labels : List[str]
            The type of nodes that the search can only flow through.
        constraint_node : UndirectedNode, optional
            The node that the search can only flow through.
        similarity_threshold : float
            The similarity threshold of the content.
        block: bool
            Despite the start node, the search can only flow through the constraint_label type nodes.

        Returns
        -------

        """

        if isinstance(content, str):
            content = [content]

        res_list = []
        for query in content:
            similar_nodes = self.semantic_search(
                content=query,
                topk_k=topk_k,
                similarity_threshold=similarity_threshold,
            )

            connected_nodes = []
            for node in similar_nodes:
                graph_query_node_res = self.query_by_node(
                    node,
                    step=step,
                    constraint_labels=constraint_labels,
                    constraint_node=constraint_node,
                    constraint_distance=constraint_distance,
                    block=block,
                )
                connected_nodes.extend(
                    [node for node in graph_query_node_res if node not in connected_nodes],
                )
                if len(connected_nodes) >= topk_k:
                    break

            res_list.extend(
                [node for node in connected_nodes[:topk_k] if node not in res_list],
            )
        return res_list

    @staticmethod
    def intersection(nodes1: list[UndirectedNode], nodes2: list[UndirectedNode]) -> list[UndirectedNode]:
        return [node for node in nodes1 if node in nodes2]

    @staticmethod
    def different(nodes1: list[UndirectedNode], nodes2: list[UndirectedNode]) -> list[UndirectedNode]:
        return list(set(nodes1).symmetric_difference(set(nodes2)))

    @staticmethod
    def cal_distance(node1: UndirectedNode, node2: UndirectedNode) -> float:
        return cosine(node1.embedding, node2.embedding)

    @staticmethod
    def filter_label(nodes: list[UndirectedNode], labels: list[str]) -> list[UndirectedNode]:
        return [node for node in nodes if node.label in labels]


def graph_to_edges(graph: dict[str, list[str]]) -> list[tuple[str, str]]:
    edges = []

    for node, neighbors in graph.items():
        for neighbor in neighbors:
            if (node, neighbor) in edges or (neighbor, node) in edges:
                continue
            edges.append((node, neighbor))

    return edges


def assign_random_coordinate_to_node(
    nodes: list[str],
    scope: float = 1.0,
    origin: tuple[float, float] = (0.0, 0.0),
) -> dict[str, tuple[float, float]]:
    coordinates = {}
    for node in nodes:
        x = random.SystemRandom().uniform(0, scope) + origin[0]
        y = random.SystemRandom().uniform(0, scope) + origin[1]
        coordinates[node] = (x, y)

    return coordinates


def assign_isometric_coordinate_to_node(
    nodes: list,
    x_step: float = 1.0,
    x_origin: float = 0.0,
    y_origin: float = 0.0,
) -> dict:
    coordinates = {}

    for i, node in enumerate(nodes):
        x = x_origin + i * x_step
        y = y_origin
        coordinates[node] = (x, y)

    return coordinates


def curly_node_coordinate(
    coordinates: dict,
    center_y: float = 1.0,
    r: float = 1.0,
) -> dict:
    # noto: this method can only curly < 90 degree, and the curl line is circle.
    # the original function is: x**2 + (y-m)**2 = r**2
    for node, coordinate in coordinates.items():
        coordinates[node] = (coordinate[0], center_y + (r**2 - coordinate[0] ** 2) ** 0.5)
    return coordinates
