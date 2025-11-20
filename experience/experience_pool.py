from collections import defaultdict
import heapq
import time
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
import difflib

# from knowledge.validation_rules import ValidationRule, TaskConstraint, SeverityLevel, RuleEvidence, ConstraintEvidence
# from utils.types import Subtask, SubtaskDAG
# from models.llm_manager import LLMManager
# _calculate_similarity更改

if TYPE_CHECKING:
    from knowledge.validation_rules import ValidationRule, TaskConstraint
    from core.types import Subtask, SubtaskDAG, Experience
    from models.llm_manager import LLMManager


class SeverityLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class RuleEvidence:
    experience_id: int
    task_type: str
    validation_result: bool
    context_similarity: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConstraintEvidence:
    experience_id: int
    constraint_id: str
    was_satisfied: bool
    value: Any
    timestamp: float = field(default_factory=time.time)


@dataclass
class ValidationRule:
    id: str
    description: str
    category: str
    domain: Optional[str] = None
    severity: SeverityLevel = SeverityLevel.MEDIUM
    confidence: float = 1.0
    last_updated: float = field(default_factory=time.time)
    usage_count: int = 0
    success_count: int = 0
    supporting_evidence: List[RuleEvidence] = field(default_factory=list)
    conflicting_evidence: List[RuleEvidence] = field(default_factory=list)
    required_constraints: List[str] = field(default_factory=list)

    def update_confidence(self, is_valid: bool, context_similarity: float, experience_id: int, task_type: str) -> None:
        self.usage_count += 1
        if is_valid:
            self.success_count += 1
            self.supporting_evidence.append(
                RuleEvidence(experience_id, task_type, True, context_similarity)
            )
        else:
            self.conflicting_evidence.append(
                RuleEvidence(experience_id, task_type, False, context_similarity)
            )
        # 置信度计算
        total_weight = 0.0
        weighted_success = 0.0
        current_time = time.time()
        for evidence in self.supporting_evidence:
            time_factor = np.exp(-(current_time - evidence.timestamp) / (7 * 86400))
            weighted_success += evidence.context_similarity * time_factor
            total_weight += time_factor
        for evidence in self.conflicting_evidence:
            time_factor = np.exp(-(current_time - evidence.timestamp) / (7 * 86400))
            total_weight += time_factor
        self.confidence = weighted_success / max(total_weight, 1e-5)
        self.last_updated = time.time()
        # 确保置信度在合理范围内
        self.confidence = max(0.1, min(1.0, self.confidence))


@dataclass
class TaskConstraint:
    id: str
    description: str
    type: str
    value: Any
    source: str
    confidence: float = 1.0
    propagation_level: int = 0
    related_rules: List[str] = field(default_factory=list)
    evidence: List[ConstraintEvidence] = field(default_factory=list)

    def update_confidence(self, was_satisfied: bool, value: Any, experience_id: int) -> None:
        self.evidence.append(ConstraintEvidence(experience_id, self.id, was_satisfied, value))
        # 计算满足率
        total_weight = 0.0
        satisfied_weight = 0.0
        current_time = time.time()
        for ev in self.evidence:
            time_factor = np.exp(-(current_time - ev.timestamp) / (14 * 86400))
            if ev.was_satisfied:
                satisfied_weight += time_factor
            total_weight += time_factor
        self.confidence = satisfied_weight / max(total_weight, 1e-5)
        # 确保置信度在合理范围内
        self.confidence = max(0.1, min(1.0, self.confidence))


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
    status: str = "pending"  # pending, executing, completed, failed
    result: Optional[Any] = None


@dataclass
class SubtaskDAG:
    nodes: Dict[str, Subtask] = field(default_factory=dict)
    entry_points: List[str] = field(default_factory=list)
    exit_points: List[str] = field(default_factory=list)
    constraints: Dict[str, TaskConstraint] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_subtask(self, subtask: Subtask) -> None:
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
        try:
            import networkx as nx
            G = nx.DiGraph()
            for node_id, node in self.nodes.items():
                G.add_node(node_id)
                for dep in node.dependencies:
                    if dep != node_id:  # 避免自环
                        G.add_edge(dep, node_id)

            if not nx.is_directed_acyclic_graph(G):
                cycles = list(nx.simple_cycles(G))
                return False, f"DAG contains cycles: {cycles}", []

            if not self.entry_points:
                return False, "No entry points found", []

            # 检查是否存在不可达节点
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
            # 没有networkx库，跳过DAG验证
            return True, "DAG validation skipped (networkx not available)", []
        except Exception as e:
            return False, f"DAG validation error: {str(e)}", []


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
    subtask_dag: Optional[SubtaskDAG] = None
    embeddings: np.ndarray = None
    related_tasks: List[int] = None
    failure_reason: Optional[str] = None
    applied_rules: List[str] = field(default_factory=list)
    violated_rules: List[str] = field(default_factory=list)
    satisfied_constraints: List[str] = field(default_factory=list)
    broken_constraints: List[str] = field(default_factory=list)


class ExperiencePool:
    """支持OpenAI和本地模型的经验池实现"""

    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self.experiences: Dict[str, List[Experience]] = defaultdict(list)
        self.task_vectors = {}
        self.rule_evidence: Dict[str, List[RuleEvidence]] = defaultdict(list)
        self.constraint_evidence: Dict[str, List[ConstraintEvidence]] = defaultdict(list)
        # 初始化LLM管理器
        self._llm_manager = None
        self._embedding_initialized = False
        self._tfidf_fitted = False

    @property
    def llm_manager(self):
        if self._llm_manager is None:
            from models.llm_manager import LLMManager
            self._llm_manager = LLMManager()
        return self._llm_manager

    def _fit_tfidf_if_needed(self):
        """如果需要，训练TF-IDF向量化器"""
        if self._tfidf_fitted:
            return

        all_task_texts = []
        for exp_list in self.experiences.values():
            for exp in exp_list:
                if hasattr(exp, 'prompt') and exp.prompt:
                    all_task_texts.append(exp.prompt)

        # 需要至少5个任务才能有效训练TF-IDF
        if len(all_task_texts) >= 5:
            success = self.llm_manager.embedding_manager.fit_tfidf_vectorizer(all_task_texts)
            if success:
                self._tfidf_fitted = True
                print(f"✓ TF-IDF模型已成功拟合，使用 {len(all_task_texts)} 个任务")
            else:
                print("✗ TF-IDF模型拟合失败")
        elif all_task_texts:
            print(f"⚠️ 任务数量不足({len(all_task_texts)}/5)，跳过TF-IDF拟合")
        else:
            print("⚠️ 没有任务文本可用于训练TF-IDF模型")

    def _calculate_similarity(self, exp1: 'Experience', exp2: 'Experience') -> float:
        """计算经验相似度 - 健壮实现"""
        # 确保TF-IDF已拟合（如果可能）
        self._fit_tfidf_if_needed()

        # 1. 文本相似度 (总是使用，作为基础)
        try:
            text_similarity = difflib.SequenceMatcher(
                None,
                exp1.prompt.lower() if hasattr(exp1, 'prompt') and exp1.prompt else "",
                exp2.prompt.lower() if hasattr(exp2, 'prompt') and exp2.prompt else ""
            ).ratio()
        except Exception as e:
            print(f"文本相似度计算错误: {e}")
            text_similarity = 0.0

        # 2. 向量相似度 (如果可用)
        vector_similarity = 0.0
        if hasattr(exp1, 'embeddings') and hasattr(exp2, 'embeddings') and \
                exp1.embeddings is not None and exp2.embeddings is not None:
            try:
                min_dim = min(len(exp1.embeddings), len(exp2.embeddings))
                vec1 = exp1.embeddings[:min_dim]
                vec2 = exp2.embeddings[:min_dim]
                vector_similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
            except Exception as e:
                print(f"向量相似度计算错误: {e}")
                vector_similarity = 0.0

        # 3. TF-IDF 相似度 (如果已拟合)
        tfidf_similarity = 0.0
        if self.llm_manager.embedding_manager.is_tfidf_ready():
            try:
                tfidf_matrix = self.llm_manager.embedding_manager.tfidf_vectorizer.transform([
                    exp1.prompt if hasattr(exp1, 'prompt') and exp1.prompt else "",
                    exp2.prompt if hasattr(exp2, 'prompt') and exp2.prompt else ""
                ])
                if cosine_similarity is not None:
                    tfidf_similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
            except Exception as e:
                print(f"TF-IDF相似度计算错误: {e}")
                tfidf_similarity = 0.0

        # 4. 任务类型相似度
        type_similarity = 1.0
        if hasattr(exp1, 'task_type') and hasattr(exp2, 'task_type'):
            type_similarity = 1.0 if exp1.task_type == exp2.task_type else 0.5

        # 5. DAG结构相似度 (简化版)
        dag_similarity = 0.0
        if hasattr(exp1, 'subtask_dag') and exp1.subtask_dag and \
                hasattr(exp2, 'subtask_dag') and exp2.subtask_dag:
            # 比较任务类型分布
            type_count1 = defaultdict(int)
            type_count2 = defaultdict(int)

            for node in exp1.subtask_dag.nodes.values():
                type_count1[node.task_type] += 1
            for node in exp2.subtask_dag.nodes.values():
                type_count2[node.task_type] += 1

            common_types = set(type_count1.keys()) & set(type_count2.keys())
            if common_types:
                type_similarity_val = sum(min(type_count1[t], type_count2[t]) for t in common_types) / \
                                      max(sum(type_count1.values()), sum(type_count2.values()), 1)
                dag_similarity = type_similarity_val

        # 综合相似度 (权重根据可用性调整)
        weights = {
            'text': 0.25,
            'vector': 0.20 if vector_similarity > 0 else 0,
            'tfidf': 0.25 if self.llm_manager.embedding_manager.is_tfidf_ready() else 0,
            'type': 0.15,
            'dag': 0.15
        }

        # 调整权重使总和为1
        total_weight = sum(weights.values())
        if total_weight > 0:
            for key in weights:
                weights[key] /= total_weight

        # 计算加权相似度
        weighted_similarity = (
                weights['text'] * text_similarity +
                weights['vector'] * vector_similarity +
                weights['tfidf'] * tfidf_similarity +
                weights['type'] * type_similarity +
                weights['dag'] * dag_similarity
        )

        return weighted_similarity

    def _compute_embeddings(self, prompt: str) -> np.ndarray:
        """统一的嵌入计算接口"""
        if not self._embedding_initialized:
            # 确保TF-IDF向量化器已初始化
            all_task_texts = []
            for exp_list in self.experiences.values():
                for exp in exp_list:
                    if hasattr(exp, 'prompt') and exp.prompt:
                        all_task_texts.append(exp.prompt)

            if all_task_texts and hasattr(self.llm_manager.embedding_manager, 'fit_tfidf_vectorizer'):
                self.llm_manager.embedding_manager.fit_tfidf_vectorizer(all_task_texts)
            self._embedding_initialized = True

    def add_rule_evidence(self, rule_id: str, evidence: RuleEvidence) -> None:
        self.rule_evidence[rule_id].append(evidence)
        if len(self.rule_evidence[rule_id]) > 20:
            self.rule_evidence[rule_id] = self.rule_evidence[rule_id][-20:]

    def add_constraint_evidence(self, constraint_id: str, evidence: ConstraintEvidence) -> None:
        self.constraint_evidence[constraint_id].append(evidence)
        if len(self.constraint_evidence[constraint_id]) > 20:
            self.constraint_evidence[constraint_id] = self.constraint_evidence[constraint_id][-20:]

    # def _calculate_similarity(self, exp1: 'Experience', exp2: 'Experience') -> float:
    #     """计算经验相似度"""
    #     # 1. 文本相似度 (使用 difflib)
    #     text_similarity = difflib.SequenceMatcher(
    #         None,
    #         exp1.prompt.lower() if hasattr(exp1, 'prompt') and exp1.prompt else "",
    #         exp2.prompt.lower() if hasattr(exp2, 'prompt') and exp2.prompt else ""
    #     ).ratio()
    #
    #     # 2. 向量相似度 (使用 SentenceTransformer 或 TF-IDF)
    #     vector_similarity = 0.0
    #     if hasattr(exp1, 'embeddings') and hasattr(exp2, 'embeddings') and \
    #        exp1.embeddings is not None and exp2.embeddings is not None:
    #         # 处理不同维度的向量
    #         min_dim = min(len(exp1.embeddings), len(exp2.embeddings))
    #         vec1 = exp1.embeddings[:min_dim]
    #         vec2 = exp2.embeddings[:min_dim]
    #         vector_similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
    #
    #     # 3. TF-IDF 相似度
    #     tfidf_similarity = 0.0
    #     try:
    #         if hasattr(self.llm_manager.embedding_manager, 'tfidf_vectorizer') and \
    #                 self.llm_manager.embedding_manager.tfidf_vectorizer is not None and \
    #                 hasattr(exp1, 'prompt') and hasattr(exp2, 'prompt') and exp1.prompt and exp2.prompt:
    #             # 转换为TF-IDF向量
    #             tfidf_matrix = self.llm_manager.embedding_manager.tfidf_vectorizer.transform([
    #                 exp1.prompt,
    #                 exp2.prompt
    #             ])
    #             # 计算余弦相似度
    #             tfidf_similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    #     except Exception as e:
    #         print(f"TF-IDF相似度计算错误: {e}")
    #         tfidf_similarity = 0.0
    #
    #     # 4. 任务类型相似度
    #     type_similarity = 1.0
    #     if hasattr(exp1, 'task_type') and hasattr(exp2, 'task_type'):
    #         type_similarity = 1.0 if exp1.task_type == exp2.task_type else 0.0
    #
    #     # 5. 难度相似度
    #     difficulty_similarity = 1.0
    #     if hasattr(exp1, 'difficulty') and hasattr(exp2, 'difficulty'):
    #         difficulty_similarity = 1 - abs(exp1.difficulty - exp2.difficulty) / max(exp1.difficulty, exp2.difficulty, 1e-5)
    #
    #     # 6. DAG 结构相似度
    #     dag_similarity = 0.0
    #     if hasattr(exp1, 'subtask_dag') and exp1.subtask_dag and \
    #        hasattr(exp2, 'subtask_dag') and exp2.subtask_dag:
    #         type_count1 = defaultdict(int)
    #         type_count2 = defaultdict(int)
    #         for node in exp1.subtask_dag.nodes.values():
    #             type_count1[node.task_type] += 1
    #         for node in exp2.subtask_dag.nodes.values():
    #             type_count2[node.task_type] += 1
    #
    #         type_similarity_val = 0
    #         total_types = set(type_count1.keys()) | set(type_count2.keys())
    #         for t in total_types:
    #             count1 = type_count1.get(t, 0)
    #             count2 = type_count2.get(t, 0)
    #             type_similarity_val += min(count1, count2) / max(count1, count2, 1)
    #         type_similarity_val /= len(total_types) if total_types else 1
    #
    #         node_ratio = min(len(exp1.subtask_dag.nodes), len(exp2.subtask_dag.nodes)) / max(
    #             len(exp1.subtask_dag.nodes), len(exp2.subtask_dag.nodes), 1)
    #
    #         success_pattern = 0.0
    #         if hasattr(exp1, 'success') and hasattr(exp2, 'success') and exp1.success and exp2.success:
    #             common_rules = set(getattr(exp1, 'applied_rules', [])) & set(getattr(exp2, 'applied_rules', []))
    #             success_pattern = len(common_rules) / max(len(getattr(exp1, 'applied_rules', [])),
    #                                                    len(getattr(exp2, 'applied_rules', [])), 1)
    #
    #         dag_similarity = 0.4 * type_similarity_val + 0.3 * node_ratio + 0.3 * success_pattern

        # 7. 规则相似度
        rule_similarity = 0.0
        if hasattr(exp1, 'applied_rules') and hasattr(exp2, 'applied_rules') and \
           exp1.applied_rules and exp2.applied_rules:
            common_rules = set(exp1.applied_rules) & set(exp2.applied_rules)
            rule_similarity = len(common_rules) / max(len(exp1.applied_rules), len(exp2.applied_rules), 1)

        # 权重分配 - 纳入TF-IDF
        weights = {
            'text': 0.15,
            'vector': 0.15,
            'tfidf': 0.25,
            'type': 0.15,
            'difficulty': 0.05,
            'dag': 0.15,
            'rule': 0.10
        }

        # 计算加权相似度
        weighted_similarity = (
                weights['text'] * text_similarity +
                weights['vector'] * vector_similarity +
                weights['tfidf'] * tfidf_similarity +
                weights['type'] * type_similarity +
                weights['difficulty'] * difficulty_similarity +
                weights['dag'] * dag_similarity +
                weights['rule'] * rule_similarity
        )

        return weighted_similarity

    def _find_related_tasks(self, experience: 'Experience', threshold: float = 0.6) -> List[int]:
        related = []
        for task_type, experiences in self.experiences.items():
            for exp in experiences:
                similarity = self._calculate_similarity(experience, exp)
                if similarity > threshold:
                    related.append(exp.task_id)
        return related

    def _smart_eviction(self, task_type: str, new_experience: Experience) -> None:
        experiences = self.experiences[task_type]

        scores = []
        current_time = time.time()

        for i, exp in enumerate(experiences):
            time_factor = np.exp(-(current_time - exp.timestamp) / (7 * 86400))

            rule_quality = len(set(exp.applied_rules) - set(exp.violated_rules)) / max(len(exp.applied_rules),
                                                                                       1) if exp.applied_rules else 1.0
            success_factor = 1.5 if exp.success else 0.5
            success_factor *= (0.7 + 0.3 * rule_quality)

            similarity = self._calculate_similarity(new_experience, exp)

            difficulty_factor = exp.difficulty / 5.0

            connectivity = len(exp.related_tasks) / max(len(experiences), 1)

            rule_constraint_quality = 1.0
            if exp.success and exp.subtask_dag:
                total_constraints = len(exp.subtask_dag.constraints)
                satisfied = len(exp.satisfied_constraints)
                constraint_ratio = satisfied / max(total_constraints, 1)
                rule_constraint_quality = 0.6 + 0.4 * constraint_ratio

            score = (0.2 * time_factor +
                     0.25 * success_factor +
                     0.2 * similarity +
                     0.1 * difficulty_factor +
                     0.15 * connectivity +
                     0.1 * rule_constraint_quality)

            scores.append((score, i))

        scores.sort()
        idx_to_remove = scores[0][1]
        self.experiences[task_type].pop(idx_to_remove)
        self.experiences[task_type].append(new_experience)

    # ===== 关键：添加缺失的add_experience方法 =====
    def add_experience(self, experience: Experience) -> None:
        """添加经验到经验池"""
        task_type = experience.task_type

        # 生成嵌入向量（如果缺失）
        if not experience.embeddings:
            experience.embeddings = self._compute_embeddings(experience.prompt)

        # 初始化任务类型分组
        if task_type not in self.experiences:
            self.experiences[task_type] = []

        # 查找相关任务
        if len(self.experiences[task_type]) == 0:
            experience.related_tasks = []
        else:
            experience.related_tasks = self._find_related_tasks(experience)

        # 智能淘汰
        if len(self.experiences[task_type]) >= self.capacity:
            self._smart_eviction(task_type, experience)
        else:
            self.experiences[task_type].append(experience)

    def get_relevant_experiences(self, task, top_k: int = 3) -> List[Experience]:
        from experience.experience_pool import Experience
        if not task.prompt:
            return []

        task_vector = self._compute_embeddings(task.prompt)

        if not any(self.experiences.values()):
            return []

        scores = []
        for experiences in self.experiences.values():
            for exp in experiences:
                current_exp = Experience(
                    task_id=task.task_id,
                    task_type=task.task_type,
                    prompt=task.prompt,
                    execution_time=0,
                    success=False,
                    result="",
                    timestamp=time.time(),
                    difficulty=0,
                    subtask_dag=task.subtask_dag,
                    embeddings=task_vector,
                    applied_rules=task.applied_rules if hasattr(task, 'applied_rules') else []
                )

                similarity = self._calculate_similarity(current_exp, exp)
                heapq.heappush(scores, (-similarity, exp.timestamp, exp))

        return [heapq.heappop(scores)[2] for _ in range(min(top_k, len(scores)))]