# utils/types.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import time


@dataclass
class Subtask:
    id: str
    description: str
    task_type: str
    dependencies: List[str] = field(default_factory=list)
    required_resources: Dict[str, Any] = field(default_factory=dict)
    expected_output: str = ""
    difficulty: float = 3.0
    estimated_time: float = 60.0
    applied_constraints: List[str] = field(default_factory=list)
    rule_violations: List[str] = field(default_factory=list)
    actual_start_time: Optional[float] = None
    actual_end_time: Optional[float] = None
    status: str = "pending"
    result: Optional[Any] = None


@dataclass
class SubtaskDAG:
    nodes: Dict[str, Subtask] = field(default_factory=dict)
    entry_points: List[str] = field(default_factory=list)
    exit_points: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)  # 修复：添加冒号

    def add_subtask(self, subtask: Subtask) -> None:
        self.nodes[subtask.id] = subtask
        if not subtask.dependencies:
            if subtask.id not in self.entry_points:
                self.entry_points.append(subtask.id)
        else:
            if subtask.id in self.entry_points:
                self.entry_points.remove(subtask.id)

        is_exit = True
        for node_id, node in self.nodes.items():
            if subtask.id in node.dependencies and node_id != subtask.id:
                is_exit = False
                break
        if is_exit and subtask.id not in self.exit_points:
            self.exit_points.append(subtask.id)

    def validate_dag(self) -> Tuple[bool, str, List[Dict]]:
        try:
            import networkx as nx
            G = nx.DiGraph()
            for node_id, node in self.nodes.items():
                G.add_node(node_id)
                for dep in node.dependencies:
                    if dep != node_id:
                        G.add_edge(dep, node_id)

            if not nx.is_directed_acyclic_graph(G):
                cycles = list(nx.simple_cycles(G))
                return False, f"DAG contains cycles: {cycles}", []

            if not self.entry_points:
                return False, "No entry points found", []

            unreachable = []
            for node_id in self.nodes:
                reachable = False
                for entry in self.entry_points:
                    if entry == node_id or (entry in G and node_id in G and nx.has_path(G, entry, node_id)):
                        reachable = True
                        break
                if not reachable:
                    unreachable.append(node_id)

            if unreachable:
                return False, f"Nodes {unreachable} are not reachable from entry points", []

            return True, "DAG is valid", []
        except ImportError:
            return True, "DAG validation skipped (networkx not available)", []
        except Exception as e:
            return False, f"DAG validation error: {str(e)}", []