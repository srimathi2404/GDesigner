"""Communication topologies as DAGs over agent indices 0..N-1.

An edge (i, j) means agent i's message is forwarded to agent j within a round
("spatial" edge in G-Designer terms). Every agent also feeds the sink.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

Edge = Tuple[int, int]


def chain(n: int) -> List[Edge]:
    return [(i, i + 1) for i in range(n - 1)]


def mesh(n: int) -> List[Edge]:
    """Complete DAG: every agent hears all earlier agents."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def star(n: int) -> List[Edge]:
    return [(0, j) for j in range(1, n)]


def empty(n: int) -> List[Edge]:
    """No within-round edges (pure debate: agents only see the previous round)."""
    return []


TOPOLOGIES: Dict[str, Callable[[int], List[Edge]]] = {
    "chain": chain, "mesh": mesh, "star": star, "empty": empty}


def predecessors(edges: List[Edge], n: int) -> List[List[int]]:
    preds: List[List[int]] = [[] for _ in range(n)]
    for i, j in edges:
        preds[j].append(i)
    return preds


def topological_order(edges: List[Edge], n: int) -> List[int]:
    preds = predecessors(edges, n)
    indeg = [len(p) for p in preds]
    order, queue = [], [i for i in range(n) if indeg[i] == 0]
    while queue:
        u = queue.pop(0)
        order.append(u)
        for i, j in edges:
            if i == u:
                indeg[j] -= 1
                if indeg[j] == 0:
                    queue.append(j)
    if len(order) != n:
        raise ValueError("topology has a cycle")
    return order
