from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, TypedDict


NodeType = Literal["document", "section", "paragraph", "summary", "chunk"]


class NodeMetadata(TypedDict, total=False):
    pdf_name: str
    page: int
    doc_name: str
    original_parent_id: str
    original_chunk_id: str


@dataclass
class Node:
    """
    Unified document node representation used by all retrieval layers.
    """

    id: str
    content: str
    type: NodeType
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    keywords: List[str] = field(default_factory=list)
    metadata: NodeMetadata = field(default_factory=dict)


class ScoreComponents(TypedDict, total=False):
    semantic: float
    structural: float
    keyword: float


class ScoredNode(TypedDict):
    node: Node
    source: str  # e.g. "vector", "keyword", "tree"
    scores: ScoreComponents
    final_score: float


def legacy_tree_to_nodes(tree: Dict, pdf_name: Optional[str] = None) -> Dict[str, Node]:
    """
    Convert the existing chunk tree JSON structure into a Node graph.

    This is a lightweight adapter so we can start using the unified Node model
    without changing the on-disk JSON format yet.
    """
    if not tree:
        return {}

    pdf_name = pdf_name or tree.get("doc_name") or ""

    nodes: Dict[str, Node] = {}

    # Root node derived from root_summary
    root_summary = tree.get("root_summary") or ""
    root_id = f"{pdf_name}::root"
    nodes[root_id] = Node(
        id=root_id,
        content=root_summary,
        type="document",
        parent=None,
        children=[],
        metadata={"pdf_name": pdf_name, "doc_name": tree.get("doc_name", pdf_name)},
    )

    # Parent summary nodes
    for parent in tree.get("parents", []):
        parent_id = str(parent.get("parent_id"))
        if not parent_id:
            continue

        child_chunk_ids = [str(cid) for cid in parent.get("child_chunk_ids", [])]

        node = Node(
            id=parent_id,
            content=parent.get("summary") or "",
            type="section",
            parent=root_id,
            children=child_chunk_ids,
            metadata={
                "pdf_name": pdf_name,
                "doc_name": tree.get("doc_name", pdf_name),
                "original_parent_id": parent_id,
            },
        )
        nodes[parent_id] = node

        # Attach parent under root
        nodes[root_id].children.append(parent_id)

    # Chunk / paragraph nodes
    for chunk in tree.get("chunks", []):
        chunk_id = str(chunk.get("chunk_id"))
        if not chunk_id:
            continue

        page = int(chunk.get("page", 0) or 0)

        # If the chunk already exists as a parent child, keep relationships;
        # otherwise relationships will be filled from the parents above.
        if chunk_id in nodes:
            node = nodes[chunk_id]
            node.content = chunk.get("text") or node.content
            node.metadata["page"] = page
            node.metadata["original_chunk_id"] = chunk_id
            continue

        nodes[chunk_id] = Node(
            id=chunk_id,
            content=chunk.get("text") or "",
            type="paragraph",
            parent=None,  # Will be inferred via parents' child lists
            children=[],
            metadata={
                "pdf_name": pdf_name,
                "doc_name": tree.get("doc_name", pdf_name),
                "page": page,
                "original_chunk_id": chunk_id,
            },
        )

    # Backfill parent links for chunk nodes using parents' child lists
    for parent in tree.get("parents", []):
        parent_id = str(parent.get("parent_id"))
        if not parent_id or parent_id not in nodes:
            continue
        for cid in parent.get("child_chunk_ids", []):
            cid_str = str(cid)
            child_node = nodes.get(cid_str)
            if not child_node:
                continue
            # Only set parent if it's not already set
            if child_node.parent is None:
                child_node.parent = parent_id

    return nodes

