import json
import time
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import networkx as nx
import matplotlib.pyplot as plt
from enum import Enum
import random
import numpy as np
from collections import defaultdict


class SeverityLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class Subtask:
    """子任务数据类，表示DAG中的一个节点"""
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


@dataclass
class SubtaskDAG:
    """子任务有向无环图，表示任务的执行结构"""
    nodes: Dict[str, Subtask] = field(default_factory=dict)
    entry_points: List[str] = field(default_factory=list)
    exit_points: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_subtask(self, subtask: Subtask) -> None:
        """添加子任务到DAG"""
        self.nodes[subtask.id] = subtask
        # 更新入口/出口点
        if not subtask.dependencies:
            if subtask.id not in self.entry_points:
                self.entry_points.append(subtask.id)
        else:
            if subtask.id in self.entry_points:
                self.entry_points.remove(subtask.id)

        # 检查是否是出口点
        is_exit = True
        for node_id, node in self.nodes.items():
            if subtask.id in node.dependencies and node_id != subtask.id:
                is_exit = False
                break
        if is_exit and subtask.id not in self.exit_points:
            self.exit_points.append(subtask.id)

    def validate_dag(self) -> Tuple[bool, str, List[Dict]]:
        """验证DAG结构的有效性"""
        return validate_dag_structure(self)

    def get_execution_order(self) -> List[str]:
        """获取DAG的执行顺序（拓扑排序）"""
        return get_execution_order(self)

    def compute_critical_path(self) -> Tuple[List[str], float]:
        """计算DAG的关键路径"""
        return find_critical_path(self)

    def visualize(self, filename: str = None) -> None:
        """可视化DAG结构"""
        visualize_dag(self, filename)

    def to_dict(self) -> Dict[str, Any]:
        """将DAG转换为字典格式，用于序列化"""
        return serialize_dag(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubtaskDAG":
        """从字典数据创建DAG实例"""
        return deserialize_dag(data)


def validate_dag_structure(dag: SubtaskDAG) -> Tuple[bool, str, List[Dict]]:
    """
    验证DAG结构的有效性

    Args:
        dag: 要验证的SubtaskDAG实例

    Returns:
        Tuple[bool, str, List[Dict]]: (是否有效, 错误/成功消息, 问题列表)
    """
    issues = []

    try:
        # 创建NetworkX图
        G = nx.DiGraph()
        for node_id, node in dag.nodes.items():
            G.add_node(node_id)
            for dep in node.dependencies:
                if dep != node_id:  # 避免自环
                    G.add_edge(dep, node_id)

        # 检查是否有环
        if not nx.is_directed_acyclic_graph(G):
            cycles = list(nx.simple_cycles(G))
            issues.append({
                "severity": SeverityLevel.CRITICAL.value,
                "message": f"DAG contains cycles: {cycles}",
                "type": "cycle"
            })
            return False, f"DAG contains cycles: {cycles}", issues

        # 检查入口点
        if not dag.entry_points:
            issues.append({
                "severity": SeverityLevel.HIGH.value,
                "message": "No entry points found in DAG",
                "type": "entry_point"
            })
            return False, "No entry points found", issues

        # 检查是否存在不可达节点
        unreachable = []
        for node_id in dag.nodes:
            reachable = False
            for entry in dag.entry_points:
                if entry == node_id or (entry in G and node_id in G and nx.has_path(G, entry, node_id)):
                    reachable = True
                    break
            if not reachable:
                unreachable.append(node_id)

        if unreachable:
            issues.append({
                "severity": SeverityLevel.MEDIUM.value,
                "message": f"Nodes {unreachable} are not reachable from entry points",
                "type": "unreachable"
            })
            return False, f"Nodes {unreachable} are not reachable from entry points", issues

        # 检查是否存在孤立节点（没有依赖也没有被依赖）
        isolated_nodes = []
        for node_id, node in dag.nodes.items():
            has_incoming = any(
                node_id in other.dependencies for other_id, other in dag.nodes.items() if other_id != node_id)
            has_outgoing = len(node.dependencies) > 0
            if not has_incoming and not has_outgoing and len(dag.nodes) > 1:
                isolated_nodes.append(node_id)

        if isolated_nodes:
            issues.append({
                "severity": SeverityLevel.LOW.value,
                "message": f"Isolated nodes detected: {isolated_nodes}",
                "type": "isolated"
            })

        return True, "DAG is valid", issues

    except Exception as e:
        error_msg = f"DAG validation error: {str(e)}"
        issues.append({
            "severity": SeverityLevel.CRITICAL.value,
            "message": error_msg,
            "type": "exception"
        })
        return False, error_msg, issues


def get_execution_order(dag: SubtaskDAG) -> List[str]:
    """
    获取DAG的执行顺序（拓扑排序）

    Args:
        dag: SubtaskDAG实例

    Returns:
        List[str]: 按执行顺序排列的节点ID列表
    """
    try:
        G = nx.DiGraph()
        for node_id, node in dag.nodes.items():
            G.add_node(node_id)
            for dep in node.dependencies:
                if dep in dag.nodes and dep != node_id:
                    G.add_edge(dep, node_id)

        if not nx.is_directed_acyclic_graph(G):
            # 如果不是DAG，使用节点添加顺序
            return list(dag.nodes.keys())

        return list(nx.topological_sort(G))
    except Exception as e:
        print(f"Topological sort error: {e}")
        return list(dag.nodes.keys())


def find_cycles(dag: SubtaskDAG) -> List[List[str]]:
    """
    在DAG中查找循环依赖

    Args:
        dag: SubtaskDAG实例

    Returns:
        List[List[str]]: 循环路径列表
    """
    G = nx.DiGraph()
    for node_id, node in dag.nodes.items():
        G.add_node(node_id)
        for dep in node.dependencies:
            if dep in dag.nodes and dep != node_id:
                G.add_edge(dep, node_id)

    return list(nx.simple_cycles(G))


def find_critical_path(dag: SubtaskDAG) -> Tuple[List[str], float]:
    """
    找到DAG中的关键路径（最长路径）

    Args:
        dag: SubtaskDAG实例

    Returns:
        Tuple[List[str], float]: (关键路径节点列表, 路径总时间)
    """
    try:
        G = nx.DiGraph()
        for node_id, node in dag.nodes.items():
            G.add_node(node_id, weight=node.estimated_time)
            for dep in node.dependencies:
                if dep in dag.nodes and dep != node_id:
                    G.add_edge(dep, node_id, weight=0)  # 边权重为0，节点有权重

        # 使用节点权重计算最长路径
        longest_path = None
        max_weight = 0

        for source in dag.entry_points:
            for target in dag.exit_points:
                if source in G and target in G:
                    try:
                        paths = list(nx.all_simple_paths(G, source, target))
                        for path in paths:
                            path_weight = sum(G.nodes[node_id]['weight'] for node_id in path)
                            if path_weight > max_weight:
                                max_weight = path_weight
                                longest_path = path
                    except nx.NetworkXNoPath:
                        continue

        return longest_path if longest_path else [], max_weight
    except Exception as e:
        print(f"Critical path calculation error: {e}")
        return [], 0


def visualize_dag(dag: SubtaskDAG, filename: str = None) -> None:
    """
    可视化DAG结构

    Args:
        dag: SubtaskDAG实例
        filename: 保存图像的文件名，如果为None则显示图像
    """
    G = nx.DiGraph()

    # 添加节点和边
    for node_id, node in dag.nodes.items():
        G.add_node(node_id, label=f"{node.task_type}\n{node.id}",
                   size=node.difficulty, time=node.estimated_time)
        for dep in node.dependencies:
            if dep in dag.nodes and dep != node_id:
                G.add_edge(dep, node_id)

    # 创建图形
    plt.figure(figsize=(12, 8))
    pos = nx.kamada_kawai_layout(G)  # 使用更好的布局算法

    # 绘制节点
    node_sizes = [G.nodes[n]['size'] * 100 + 100 for n in G.nodes()]
    node_colors = [G.nodes[n]['time'] for n in G.nodes()]

    nx.draw_networkx_edges(G, pos, alpha=0.5, arrows=True)
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                                   node_color=node_colors, cmap=plt.cm.viridis, alpha=0.8)
    nx.draw_networkx_labels(G, pos, {n: G.nodes[n]['label'] for n in G.nodes()}, font_size=8)

    # 添加颜色条
    if nodes:
        plt.colorbar(nodes, label='Estimated Time (seconds)')

    plt.title('Task DAG Structure')
    plt.axis('off')

    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def serialize_dag(dag: SubtaskDAG) -> Dict[str, Any]:
    """
    将DAG序列化为字典

    Args:
        dag: SubtaskDAG实例

    Returns:
        Dict[str, Any]: 序列化后的DAG数据
    """
    nodes_data = {}
    for node_id, node in dag.nodes.items():
        nodes_data[node_id] = {
            'id': node.id,
            'description': node.description,
            'task_type': node.task_type,
            'dependencies': node.dependencies,
            'required_resources': node.required_resources,
            'expected_output': node.expected_output,
            'difficulty': node.difficulty,
            'estimated_time': node.estimated_time,
            'applied_constraints': node.applied_constraints,
            'rule_violations': node.rule_violations,
            'actual_start_time': node.actual_start_time,
            'actual_end_time': node.actual_end_time,
            'status': node.status,
            'result': str(node.result) if node.result is not None else None
        }

    return {
        'nodes': nodes_data,
        'entry_points': dag.entry_points,
        'exit_points': dag.exit_points,
        'constraints': dag.constraints,
        'metadata': dag.metadata
    }


def deserialize_dag(data: Dict[str, Any]) -> SubtaskDAG:
    """
    从字典数据反序列化DAG

    Args:
        data: 序列化后的DAG数据

    Returns:
        SubtaskDAG: 反序列化后的DAG实例
    """
    nodes = {}
    for node_id, node_data in data['nodes'].items():
        nodes[node_id] = Subtask(
            id=node_data['id'],
            description=node_data['description'],
            task_type=node_data['task_type'],
            dependencies=node_data['dependencies'],
            required_resources=node_data['required_resources'],
            expected_output=node_data['expected_output'],
            difficulty=node_data['difficulty'],
            estimated_time=node_data['estimated_time'],
            applied_constraints=node_data.get('applied_constraints', []),
            rule_violations=node_data.get('rule_violations', []),
            actual_start_time=node_data.get('actual_start_time'),
            actual_end_time=node_data.get('actual_end_time'),
            status=node_data.get('status', 'pending'),
            result=node_data.get('result')
        )

    return SubtaskDAG(
        nodes=nodes,
        entry_points=data.get('entry_points', []),
        exit_points=data.get('exit_points', []),
        constraints=data.get('constraints', {}),
        metadata=data.get('metadata', {})
    )


def add_subtask_to_dag(dag: SubtaskDAG, subtask: Subtask) -> bool:
    """
    向DAG添加子任务，并验证结构

    Args:
        dag: 目标DAG
        subtask: 要添加的子任务

    Returns:
        bool: 是否成功添加
    """
    # 检查依赖是否存在
    for dep in subtask.dependencies:
        if dep not in dag.nodes and dep != subtask.id:
            return False

    # 临时添加节点用于验证
    test_nodes = dag.nodes.copy()
    test_nodes[subtask.id] = subtask

    test_entry_points = dag.entry_points.copy()
    test_exit_points = dag.exit_points.copy()

    if not subtask.dependencies:
        if subtask.id not in test_entry_points:
            test_entry_points.append(subtask.id)
    else:
        if subtask.id in test_entry_points:
            test_entry_points.remove(subtask.id)

    # 检查是否是出口点
    is_exit = True
    for node_id, node in test_nodes.items():
        if subtask.id in node.dependencies and node_id != subtask.id:
            is_exit = False
            break
    if is_exit and subtask.id not in test_exit_points:
        test_exit_points.append(subtask.id)

    # 验证DAG结构
    test_dag = SubtaskDAG(
        nodes=test_nodes,
        entry_points=test_entry_points,
        exit_points=test_exit_points,
        constraints=dag.constraints.copy()
    )

    is_valid, _, _ = validate_dag_structure(test_dag)
    if is_valid:
        dag.add_subtask(subtask)
        return True

    return False


def remove_subtask_from_dag(dag: SubtaskDAG, subtask_id: str) -> bool:
    """
    从DAG中移除子任务

    Args:
        dag: 目标DAG
        subtask_id: 要移除的子任务ID

    Returns:
        bool: 是否成功移除
    """
    if subtask_id not in dag.nodes:
        return False

    # 检查是否有其他节点依赖此节点
    dependent_nodes = []
    for node_id, node in dag.nodes.items():
        if node_id != subtask_id and subtask_id in node.dependencies:
            dependent_nodes.append(node_id)

    if dependent_nodes:
        return False  # 不能移除被依赖的节点

    # 移除节点
    removed_node = dag.nodes.pop(subtask_id)

    # 更新入口/出口点
    if subtask_id in dag.entry_points:
        dag.entry_points.remove(subtask_id)

    if subtask_id in dag.exit_points:
        dag.exit_points.remove(subtask_id)

    # 检查移除后是否需要更新其他节点的入口/出口状态
    for node_id, node in dag.nodes.items():
        if not node.dependencies and node_id not in dag.entry_points:
            dag.entry_points.append(node_id)

        # 检查是否应为出口点
        is_exit = True
        for other_id, other in dag.nodes.items():
            if node_id in other.dependencies and other_id != node_id:
                is_exit = False
                break
        if is_exit and node_id not in dag.exit_points:
            dag.exit_points.append(node_id)

    return True


def update_subtask_dependencies(dag: SubtaskDAG, subtask_id: str, new_dependencies: List[str]) -> bool:
    """
    更新子任务的依赖关系

    Args:
        dag: 目标DAG
        subtask_id: 要更新的子任务ID
        new_dependencies: 新的依赖列表

    Returns:
        bool: 是否成功更新
    """
    if subtask_id not in dag.nodes:
        return False

    # 保存原始依赖
    original_dependencies = dag.nodes[subtask_id].dependencies.copy()
    dag.nodes[subtask_id].dependencies = new_dependencies

    # 验证DAG结构
    is_valid, _, _ = validate_dag_structure(dag)

    if not is_valid:
        # 恢复原始依赖
        dag.nodes[subtask_id].dependencies = original_dependencies
        return False

    # 更新入口/出口点
    if not new_dependencies:
        if subtask_id not in dag.entry_points:
            dag.entry_points.append(subtask_id)
    else:
        if subtask_id in dag.entry_points:
            dag.entry_points.remove(subtask_id)

    # 检查是否是出口点
    is_exit = True
    for node_id, node in dag.nodes.items():
        if subtask_id in node.dependencies and node_id != subtask_id:
            is_exit = False
            break

    if is_exit:
        if subtask_id not in dag.exit_points:
            dag.exit_points.append(subtask_id)
    else:
        if subtask_id in dag.exit_points:
            dag.exit_points.remove(subtask_id)

    return True


def analyze_dag_parallelism(dag: SubtaskDAG) -> Dict[str, Any]:
    """分析DAG的并行执行潜力"""
    # 计算关键路径
    critical_path, critical_time = dag.compute_critical_path()

    # 计算总时间（假设串行执行）
    serial_time = sum(node.estimated_time for node in dag.nodes.values())

    # 识别可以并行执行的任务组
    parallel_groups = defaultdict(list)
    for node_id, node in dag.nodes.items():
        # 基于任务类型分组
        parallel_groups[node.task_type].append(node_id)

        # 基于依赖关系分组
        if len(node.dependencies) > 1:
            dep_key = "_and_".join(sorted(node.dependencies))
            parallel_groups[dep_key].append(node_id)

    # 识别独立任务（可以并行执行）
    independent_tasks = []
    for node_id, node in dag.nodes.items():
        if not node.dependencies or all(dep in dag.entry_points for dep in node.dependencies):
            independent_tasks.append(node_id)

    # 计算理论最大并行度
    max_parallelism = 0
    for node_id in dag.nodes:
        # 找出可以同时执行的任务数量
        parallel_count = 1
        for other_id, other in dag.nodes.items():
            if node_id != other_id and not _tasks_dependent(node_id, other_id, dag):
                parallel_count += 1
        max_parallelism = max(max_parallelism, parallel_count)

    return {
        "critical_path": critical_path,
        "critical_time": critical_time,
        "serial_time": serial_time,
        "potential_parallel_groups": dict(parallel_groups),
        "independent_tasks": independent_tasks,
        "max_parallelism": max_parallelism,
        "potential_speedup": serial_time / critical_time if critical_time > 0 else 1
    }


def _tasks_dependent(task1: str, task2: str, dag: SubtaskDAG) -> bool:
    """检查两个任务是否有依赖关系"""
    # 检查task1是否依赖task2
    if task2 in dag.nodes[task1].dependencies:
        return True

    # 检查task2是否依赖task1
    if task1 in dag.nodes[task2].dependencies:
        return True

    # 检查是否有间接依赖
    return _has_path(dag, task1, task2) or _has_path(dag, task2, task1)


def _has_path(dag: SubtaskDAG, source: str, target: str) -> bool:
    """检查DAG中是否存在从source到target的路径"""
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
        for next_node, n in dag.nodes.items():
            if node in n.dependencies and next_node not in visited:
                queue.append(next_node)

    return False


def visualize_dag_with_parallelism(dag: SubtaskDAG, filename: str = None) -> None:
    """可视化DAG结构，突出显示并行执行机会"""
    try:
        import matplotlib.pyplot as plt
        import networkx as nx

        G = nx.DiGraph()

        # 添加节点
        for node_id, node in dag.nodes.items():
            G.add_node(node_id,
                       label=f"{node_id}\n{node.task_type}",
                       color="lightblue")

        # 添加边
        for node_id, node in dag.nodes.items():
            for dep in node.dependencies:
                if dep != node_id:  # 避免自环
                    G.add_edge(dep, node_id)

        # 分析并行性
        parallelism = analyze_dag_parallelism(dag)

        # 标记关键路径
        for i in range(len(parallelism["critical_path"]) - 1):
            G.edges[parallelism["critical_path"][i], parallelism["critical_path"][i + 1]]['color'] = 'red'
            G.edges[parallelism["critical_path"][i], parallelism["critical_path"][i + 1]]['width'] = 2.0

        # 设置布局
        pos = nx.spring_layout(G, k=0.5, iterations=50)

        # 绘制节点
        node_colors = ['lightblue' for _ in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.8)

        # 绘制边
        edge_colors = [G[u][v].get('color', 'black') for u, v in G.edges()]
        edge_widths = [G[u][v].get('width', 1.0) for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths,
                               arrowsize=20, alpha=0.7, connectionstyle='arc3,rad=0.1')

        # 添加标签
        labels = {node: f"{node}\n{dag.nodes[node].task_type}" for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)

        plt.title(
            f"DAG结构 (关键路径时间: {parallelism['critical_time']:.1f}s, 潜在加速比: {parallelism['potential_speedup']:.2f}x)")
        plt.axis('off')

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"DAG可视化已保存至: {filename}")
        else:
            plt.show()

    except ImportError:
        print("警告: 无法可视化DAG，缺少matplotlib或networkx库")