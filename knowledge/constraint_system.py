from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable, TYPE_CHECKING
import numpy as np
import time
from collections import defaultdict

if TYPE_CHECKING:
    from core.types import Subtask, SubtaskDAG
    from knowledge.domain_knowledge import DomainKnowledgeBase
    from experience.experience_pool import ExperiencePool


@dataclass
class ConstraintEvidence:
    experience_id: int
    constraint_id: str
    was_satisfied: bool
    value: Any
    timestamp: float = field(default_factory=time.time)


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
        """基于新证据更新约束置信度"""
        self.evidence.append(ConstraintEvidence(experience_id, self.id, was_satisfied, value))

        # 保留最近20条证据
        if len(self.evidence) > 20:
            self.evidence = self.evidence[-20:]

        # 计算满足率 - 基于时间衰减的加权平均
        total_weight = 0.0
        satisfied_weight = 0.0
        current_time = time.time()

        for ev in self.evidence:
            # 14天半衰期
            time_factor = np.exp(-(current_time - ev.timestamp) / (14 * 86400))
            if ev.was_satisfied:
                satisfied_weight += time_factor
            total_weight += time_factor

        self.confidence = satisfied_weight / max(total_weight, 1e-5)


class ConstraintPropagator:
    """约束传播器，负责约束的传播和验证"""

    def __init__(self, knowledge_base: Any, experience_pool: Any):
        self.kb = knowledge_base
        self.pool = experience_pool

        # 添加检查确保所有传播方法都已定义
        self._validate_propagation_methods()

    def _validate_propagation_methods(self):
        """验证所有约束传播方法是否存在"""
        missing_methods = []
        for constraint_id in self.kb.constraint_propagation.keys():
            method = self.kb.constraint_propagation[constraint_id]
            if not hasattr(self.kb, method.__name__):
                missing_methods.append(method.__name__)

        if missing_methods:
            raise AttributeError(f"Missing propagation methods: {', '.join(missing_methods)}")

    def propagate_constraints(self, main_task: str, task_type: str,
                              current_dag: Any) -> Dict[str, TaskConstraint]:
        """传播约束到当前DAG"""
        new_constraints = {}

        # 1. 从领域知识库获取初始约束
        domain_constraints = self.kb.get_constraints_for_domain(task_type)
        for constraint in domain_constraints:
            if constraint.id in current_dag.constraints:
                continue

            # 传播到DAG级别
            propagated = self._propagate_to_dag(constraint, main_task, current_dag)
            if propagated:
                new_constraints[propagated.id] = propagated
                current_dag.constraints[propagated.id] = propagated

        # 2. 从现有约束传播到新子任务
        if current_dag.nodes:
            last_subtask = list(current_dag.nodes.values())[-1]
            for constraint_id, constraint in current_dag.constraints.items():
                propagated = self.kb.propagate_constraint(
                    constraint_id, constraint, last_subtask, current_dag
                )
                if propagated and propagated.id not in current_dag.constraints:
                    new_constraints[propagated.id] = propagated
                    current_dag.constraints[propagated.id] = propagated

        return new_constraints

    def _propagate_to_dag(self, constraint: TaskConstraint, main_task: str,
                          current_dag: Any) -> Optional[TaskConstraint]:
        """将约束传播到DAG级别"""
        # 检查是否是关键约束
        if "must" in constraint.description.lower() or "required" in constraint.description.lower():
            if constraint.value in current_dag.constraints.values():
                return None

            return TaskConstraint(
                id=f"dag_{constraint.id}",
                description=f"DAG-level: {constraint.description}",
                type=constraint.type,
                value=constraint.value,
                source="domain_rule",
                propagation_level=0,
                related_rules=constraint.related_rules.copy()
            )

        return None

    def validate_constraint_satisfaction(self, subtask: Any, current_dag: Any) -> List[Dict]:
        """验证子任务是否满足所有相关约束"""
        issues = []

        for constraint_id, constraint in current_dag.constraints.items():
            if self._constraint_applies_to_subtask(constraint, subtask, current_dag):
                is_satisfied, message = self._check_constraint_satisfaction(
                    constraint, subtask, current_dag
                )
                if not is_satisfied:
                    issues.append({
                        "severity": "high",  # 约束违反通常被视为高严重性
                        "message": message,
                        "constraint_id": constraint_id
                    })

        return issues

    def _constraint_applies_to_subtask(self, constraint: TaskConstraint,
                                       subtask: Any, dag: Any) -> bool:
        """检查约束是否适用于当前子任务"""
        if constraint.type == "dependency":
            parts = constraint.value.split("→")
            if len(parts) != 2:
                return True

            prerequisite = parts[0].strip()
            target = parts[1].strip()
            return target == subtask.task_type

        elif constraint.type == "resource":
            return any(res in subtask.required_resources for res in constraint.value)

        return True

    def _check_constraint_satisfaction(self, constraint: TaskConstraint,
                                       subtask: Any, dag: Any) -> Tuple[bool, str]:
        """检查特定约束是否被满足"""
        if constraint.type == "dependency":
            parts = constraint.value.split("→")
            if len(parts) != 2:
                return True, "Invalid dependency format"

            prerequisite = parts[0].strip()
            target = parts[1].strip()

            if subtask.task_type != target:
                return True, "Not applicable to this subtask type"

            # 检查前提条件是否满足
            has_prerequisite = any(
                node.task_type == prerequisite for node in dag.nodes.values()
            )

            if not has_prerequisite:
                return False, f"Missing prerequisite task type: {prerequisite}"

            return True, f"Prerequisite {prerequisite} exists"

        elif constraint.type == "resource":
            required = constraint.value
            if not all(res in subtask.required_resources for res in required):
                missing = [res for res in required if res not in subtask.required_resources]
                return False, f"Missing required resources: {missing}"
            return True, "All resources specified"

        elif constraint.type == "time":
            if subtask.estimated_time > constraint.value:
                return False, f"Estimated time ({subtask.estimated_time}s) exceeds limit ({constraint.value}s)"
            return True, "Time estimate within limit"

        return True, "Constraint check bypassed"