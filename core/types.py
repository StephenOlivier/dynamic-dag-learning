"""
核心类型定义，用于解决循环依赖问题
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
import time
import numpy as np
import difflib
import random
import string

@dataclass
class Experience:
    """经验数据类，存储任务执行的历史信息"""
    task_id: int
    task_type: str
    prompt: str
    execution_time: float
    success: bool
    result: str
    timestamp: float
    difficulty: float
    subtask_dag: Optional['SubtaskDAG'] = None
    embeddings: Optional[np.ndarray] = None
    related_tasks: Optional[List[int]] = None
    failure_reason: Optional[str] = None
    applied_rules: List[str] = field(default_factory=list)
    violated_rules: List[str] = field(default_factory=list)
    satisfied_constraints: List[str] = field(default_factory=list)
    broken_constraints: List[str] = field(default_factory=list)

@dataclass
class Subtask:
    """
    子任务数据类，表示DAG中的一个节点

    Attributes:
        id: 子任务的唯一标识符
        description: 子任务的详细描述
        task_type: 子任务的类型（如cleaning, analysis, implementation等）
        dependencies: 依赖的其他子任务ID列表
        required_resources: 执行此子任务所需的资源
        expected_output: 预期的输出描述
        difficulty: 任务难度（1-5）
        estimated_time: 预估执行时间（秒）
        applied_constraints: 应用的约束ID列表
        rule_violations: 违反的规则ID列表
        actual_start_time: 实际开始时间戳
        actual_end_time: 实际结束时间戳
        status: 任务状态（pending, executing, completed, failed）
        result: 执行结果
    """
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
    status: str = "pending"  # pending, executing, completed, failed
    result: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """将子任务转换为字典格式"""
        return {
            "id": self.id,
            "description": self.description,
            "task_type": self.task_type,
            "dependencies": self.dependencies,
            "required_resources": self.required_resources,
            "expected_output": self.expected_output,
            "difficulty": self.difficulty,
            "estimated_time": self.estimated_time,
            "applied_constraints": self.applied_constraints,
            "rule_violations": self.rule_violations,
            "actual_start_time": self.actual_start_time,
            "actual_end_time": self.actual_end_time,
            "status": self.status,
            "result": str(self.result) if self.result is not None else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Subtask":
        """从字典数据创建子任务实例"""
        return cls(
            id=data["id"],
            description=data["description"],
            task_type=data["task_type"],
            dependencies=data.get("dependencies", []),
            required_resources=data.get("required_resources", {}),
            expected_output=data.get("expected_output", ""),
            difficulty=data.get("difficulty", 3.0),
            estimated_time=data.get("estimated_time", 60.0),
            applied_constraints=data.get("applied_constraints", []),
            rule_violations=data.get("rule_violations", []),
            actual_start_time=data.get("actual_start_time"),
            actual_end_time=data.get("actual_end_time"),
            status=data.get("status", "pending"),
            result=data.get("result")
        )


@dataclass
class SubtaskDAG:
    """
    子任务有向无环图，表示任务的执行结构

    Attributes:
        nodes: 子任务节点字典，键为子任务ID
        entry_points: 入口节点ID列表（没有依赖的节点）
        exit_points: 出口节点ID列表（不被其他节点依赖的节点）
        constraints: 约束条件字典
        meta: 元数据字典
    """
    nodes: Dict[str, Subtask] = field(default_factory=dict)
    entry_points: List[str] = field(default_factory=list)
    exit_points: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def add_subtask(self, subtask: Subtask) -> None:
        """
        添加子任务到DAG

        Args:
            subtask: 要添加的子任务
        """
        self.nodes[subtask.id] = subtask

        # 更新入口点
        if not subtask.dependencies:
            if subtask.id not in self.entry_points:
                self.entry_points.append(subtask.id)
        else:
            if subtask.id in self.entry_points:
                self.entry_points.remove(subtask.id)

        # 更新出口点
        is_exit = True
        for node_id, node in self.nodes.items():
            if subtask.id in node.dependencies and node_id != subtask.id:
                is_exit = False
                break
        if is_exit and subtask.id not in self.exit_points:
            self.exit_points.append(subtask.id)

    def validate_dag(self) -> Tuple[bool, str, List[Dict]]:
        """验证DAG结构的有效性 - 支持多依赖关系"""
        issues = []

        try:
            # 检查是否有重复ID
            if len(self.nodes) != len(set(self.nodes.keys())):
                duplicate_ids = [id for id in self.nodes.keys()
                                 if list(self.nodes.keys()).count(id) > 1]
                issues.append({
                    "severity": "critical",
                    "message": f"Duplicate node IDs found: {duplicate_ids}",
                    "type": "duplicate_id"
                })
                return False, f"Duplicate node IDs found: {duplicate_ids}", issues

            # 检查所有依赖是否有效
            for node_id, node in self.nodes.items():
                for dep in node.dependencies:
                    if dep not in self.nodes and dep != node_id:
                        issues.append({
                            "severity": "critical",
                            "message": f"Dependency '{dep}' for node '{node_id}' does not exist",
                            "type": "invalid_dependency"
                        })
                        return False, f"Dependency '{dep}' for node '{node_id}' does not exist", issues

            # 改进的Kahn算法检测循环（支持多依赖）
            in_degree = {node_id: 0 for node_id in self.nodes}
            for node_id, node in self.nodes.items():
                for dep in node.dependencies:
                    if dep in in_degree and dep != node_id:
                        in_degree[node_id] += 1  # 修正：增加被依赖节点的入度

            # 创建没有入边的节点队列
            queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
            visited_count = 0
            visited_nodes = []

            while queue:
                node_id = queue.pop(0)
                visited_count += 1
                visited_nodes.append(node_id)

                # 减少依赖此节点的所有节点的入度
                for current_id, current_node in self.nodes.items():
                    if node_id in current_node.dependencies:
                        in_degree[current_id] -= 1
                        if in_degree[current_id] == 0:
                            queue.append(current_id)

            # 如果访问的节点数少于总节点数，说明有循环
            if visited_count != len(self.nodes):
                # 尝试找出具体循环
                cycles = []
                unvisited = set(self.nodes.keys()) - set(visited_nodes)

                if unvisited:
                    for start_node in unvisited:
                        path = []
                        visited = set()

                        def find_cycle(node):
                            if node in visited:
                                if node in path:
                                    idx = path.index(node)
                                    cycle = path[idx:] + [node]
                                    if cycle not in cycles:
                                        cycles.append(cycle)
                                return

                            visited.add(node)
                            path.append(node)

                            for next_node, n in self.nodes.items():
                                if node in n.dependencies:
                                    find_cycle(next_node)

                            path.pop()
                            visited.remove(node)

                        try:
                            find_cycle(start_node)
                        except RecursionError:
                            cycles.append(["...循环路径过长..."])

                cycle_str = "detected cycles" if not cycles else f"cycles: {cycles}"
                issues.append({
                    "severity": "critical",
                    "message": f"DAG contains {cycle_str}",
                    "type": "cycle"
                })
                return False, f"DAG contains cycles", issues

            # 检查入口点
            if not self.entry_points:
                issues.append({
                    "severity": "high",
                    "message": "No entry points found in DAG",
                    "type": "entry_point"
                })
                return False, "No entry points found", issues

            # 检查是否存在不可达节点
            unreachable = []
            for node_id in self.nodes:
                if node_id in self.entry_points:
                    continue

                # 检查是否有路径到达此节点
                reachable = False
                for entry in self.entry_points:
                    if self._has_path(entry, node_id):
                        reachable = True
                        break

                if not reachable:
                    unreachable.append(node_id)

            if unreachable:
                issues.append({
                    "severity": "medium",
                    "message": f"Nodes {unreachable} are not reachable from entry points",
                    "type": "unreachable"
                })
                return False, f"Nodes {unreachable} are not reachable from entry points", issues

            return True, "DAG is valid", issues

        except Exception as e:
            error_msg = f"DAG validation error: {str(e)}"
            issues.append({
                "severity": "critical",
                "message": error_msg,
                "type": "exception"
            })
            return False, error_msg, issues

    def _has_path(self, source: str, target: str) -> bool:
        """检查是否存在从source到target的路径 - 支持多依赖"""
        if source == target:
            return True

        visited = set()
        queue = [source]

        while queue:
            node = queue.pop(0)
            if node == target:
                return True

            if node in visited:
                continue

            visited.add(node)
            # 找出所有当前节点指向的节点（即依赖当前节点的节点）
            for next_node, n in self.nodes.items():
                if node in n.dependencies and next_node not in visited:
                    queue.append(next_node)

        return False

    def get_execution_order(self) -> List[str]:
        """获取DAG的执行顺序（拓扑排序） - 改进版"""
        try:
            # 改进的Kahn算法实现拓扑排序
            in_degree = {node_id: 0 for node_id in self.nodes}
            for node_id, node in self.nodes.items():
                for dep in node.dependencies:
                    if dep in in_degree and dep != node_id:
                        in_degree[node_id] += 1

            # 创建没有入边的节点队列
            queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
            result = []

            while queue:
                # 随机选择一个节点（增加多样性）
                node_id = queue.pop(random.randint(0, len(queue) - 1))
                result.append(node_id)

                # 减少依赖此节点的所有节点的入度
                for current_id, current_node in self.nodes.items():
                    if node_id in current_node.dependencies:
                        in_degree[current_id] -= 1
                        if in_degree[current_id] == 0:
                            queue.append(current_id)

            # 如果无法排序所有节点，添加剩余节点
            if len(result) < len(self.nodes):
                remaining = [node_id for node_id in self.nodes if node_id not in result]
                result.extend(remaining)

            return result

        except Exception as e:
            print(f"拓扑排序错误: {e}")
            return list(self.nodes.keys())


    def to_dict(self) -> Dict[str, Any]:
        """
        将DAG转换为字典格式

        Returns:
            Dict[str, Any]: 序列化后的DAG数据
        """
        nodes_data = {}
        for node_id, node in self.nodes.items():
            nodes_data[node_id] = node.to_dict()

        return {
            "nodes": nodes_data,
            "entry_points": self.entry_points,
            "exit_points": self.exit_points,
            "constraints": self.constraints,
            "meta": self.meta
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubtaskDAG":
        """
        从字典数据创建DAG实例

        Args:
            data: 序列化后的DAG数据

        Returns:
            SubtaskDAG: 反序列化后的DAG实例
        """
        nodes = {}
        for node_id, node_data in data["nodes"].items():
            nodes[node_id] = Subtask.from_dict(node_data)

        return cls(
            nodes=nodes,
            entry_points=data.get("entry_points", []),
            exit_points=data.get("exit_points", []),
            constraints=data.get("constraints", {}),
            meta=data.get("meta", {})
        )

    def compute_critical_path(self) -> Tuple[List[str], float]:
        """
        计算DAG的关键路径（最长路径）

        Returns:
            Tuple[List[str], float]: (关键路径节点列表, 路径总时间)
        """
        # 简化实现：按拓扑顺序计算最长路径
        order = self.get_execution_order()
        longest_path = {node_id: ([node_id], self.nodes[node_id].estimated_time) for node_id in self.entry_points}

        for node_id in order:
            if node_id in self.entry_points:
                continue

            max_path = []
            max_time = 0

            for dep in self.nodes[node_id].dependencies:
                if dep in longest_path:
                    path, time = longest_path[dep]
                    total_time = time + self.nodes[node_id].estimated_time
                    if total_time > max_time:
                        max_time = total_time
                        max_path = path + [node_id]

            if max_path:
                longest_path[node_id] = (max_path, max_time)

        # 找到总时间最长的路径
        critical_path = []
        max_time = 0
        for node_id, (path, time) in longest_path.items():
            if node_id in self.exit_points and time > max_time:
                critical_path = path
                max_time = time

        return critical_path, max_time

    def generate_subtask_id(self) -> str:
        """生成唯一的子任务ID"""
        base_id = f"subtask_{int(time.time())}"
        counter = 1
        while f"{base_id}_{counter}" in self.nodes:
            counter += 1
        return f"{base_id}_{counter}"