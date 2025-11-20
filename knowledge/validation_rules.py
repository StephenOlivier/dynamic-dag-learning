from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Tuple, Optional, Any, TYPE_CHECKING
import numpy as np
import time
from collections import defaultdict

if TYPE_CHECKING:
    from experience.experience_pool import ExperiencePool
    from core.types import Experience, Subtask, SubtaskDAG
    from knowledge.domain_knowledge import DomainKnowledgeBase

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
    check_func: Callable = None
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


class RuleBasedChecker:
    """基于规则的检查器，用于验证子任务合规性"""

    def __init__(self, experience_pool: Any, knowledge_base: Any):
        self.pool = experience_pool
        self.kb = knowledge_base
        self.propagator = None  # 将在外部设置
        self.active_rules: Dict[str, ValidationRule] = {}
        self._initialize_active_rules()

        self.knowledge_base = knowledge_base

    def set_constraint_propagator(self, propagator: Any) -> None:
        """设置约束传播器"""
        self.propagator = propagator

    def _initialize_active_rules(self):
        """初始化活跃规则集"""
        all_rules = []
        for domain, rules in self.kb.domain_rules.items():
            all_rules.extend(rules)

        for rule in all_rules:
            self.active_rules[rule.id] = rule

    def update_rule_confidence(self, rule_id: str, is_valid: bool,
                               context_similarity: float, experience_id: int, task_type: str) -> None:
        """更新特定规则的置信度"""
        if rule_id in self.active_rules:
            self.active_rules[rule_id].update_confidence(
                is_valid, context_similarity, experience_id, task_type
            )

    def get_relevant_rules(self, task_type: str, main_task: str,
                           current_dag: Any, top_k: int = 10) -> List[ValidationRule]:
        """获取与当前上下文最相关的规则"""
        domain_rules = self.kb.get_rules_for_domain(task_type)

        # 从经验池中检索相关经验
        from core.types import Experience
        relevant_exps = self.pool.get_relevant_experiences(
            Experience(
                task_id=0,
                task_type=task_type,
                prompt=main_task,
                execution_time=0,
                success=True,
                result="",
                timestamp=time.time(),
                difficulty=3.0,
                subtask_dag=current_dag
            ),
            top_k=3
        )

        # 分析历史使用情况
        rule_usage = defaultdict(lambda: {'count': 0, 'success': 0, 'similarity': 0.0})
        for exp in relevant_exps:
            from core.types import Experience
            similarity = self.pool._calculate_similarity(
                Experience(
                    task_id=0,
                    task_type=task_type,
                    prompt=main_task,
                    execution_time=0,
                    success=True,
                    result="",
                    timestamp=time.time(),
                    difficulty=3.0
                ),
                exp
            )

            for rule_id in exp.applied_rules:
                rule_usage[rule_id]['count'] += 1
                rule_usage[rule_id]['similarity'] = max(rule_usage[rule_id]['similarity'], similarity)
                if rule_id not in exp.violated_rules:
                    rule_usage[rule_id]['success'] += 1

        # 计算规则评分
        rule_scores = []
        for rule in domain_rules:
            base_score = rule.confidence

            # 结合历史使用证据
            if rule.id in rule_usage:
                usage = rule_usage[rule.id]
                history_score = (usage['success'] / usage['count']) * usage['similarity']
                base_score = 0.6 * base_score + 0.4 * history_score

            # 评估规则与当前DAG的相关性
            dag_relevance = self._check_rule_dag_relevance(rule, current_dag)

            # 综合评分
            final_score = 0.7 * base_score + 0.3 * dag_relevance
            rule_scores.append((final_score, rule))

        # 按评分排序并返回前k个
        rule_scores.sort(key=lambda x: x[0], reverse=True)
        return [rule for _, rule in rule_scores[:top_k]]

    def _check_rule_dag_relevance(self, rule: ValidationRule, dag: Any) -> float:
        """检查规则与当前DAG的相关性"""
        # 检查所需约束是否存在
        relevant_constraints = 0
        total_constraints = max(len(rule.required_constraints), 1)

        for constraint_id in rule.required_constraints:
            if constraint_id in dag.constraints:
                relevant_constraints += 1

        constraint_match = relevant_constraints / total_constraints

        # 检查任务类型匹配度
        task_type_match = 0.0
        if dag.nodes:
            last_task = list(dag.nodes.values())[-1]
            if last_task.task_type in rule.description.lower():
                task_type_match = 0.8
            elif any(word in rule.description.lower() for word in ["task", "step", "subtask"]):
                task_type_match = 0.5

        # 综合相关性评分
        return 0.6 * constraint_match + 0.4 * task_type_match

    def validate_subtask(
            self,
            subtask: Any,
            main_task: str,
            task_type: str,
            current_dag: Any
    ) -> Tuple[bool, List[Dict], List[str]]:
        """验证子任务是否符合规则"""
        # 传播约束
        if self.propagator:
            try:
                self.propagator.propagate_constraints(main_task, task_type, current_dag)
            except Exception as e:
                print(f"约束传播错误: {str(e)}")
                import traceback
                traceback.print_exc()

        # 获取相关规则
        try:
            relevant_rules = self.get_relevant_rules(task_type, main_task, current_dag)
        except Exception as e:
            print(f"获取相关规则错误: {str(e)}")
            relevant_rules = []

        # 应用规则检查
        issues = []
        applied_rule_ids = []

        for rule in relevant_rules:
            if not rule.check_func:
                continue

            try:
                is_valid, message = rule.check_func(subtask, main_task, current_dag, current_dag.constraints)
                applied_rule_ids.append(rule.id)

                if not is_valid:
                    issues.append({
                        "rule_id": rule.id,
                        "severity": rule.severity.value,
                        "message": message,
                        "confidence": rule.confidence
                    })
            except Exception as e:
                issues.append({
                    "rule_id": rule.id,
                    "severity": "high",
                    "message": f"Rule execution error: {str(e)}\nTraceback: {traceback.format_exc()}",
                    "confidence": 0.0
                })

        # 检查约束满足性
        if self.propagator:
            try:
                constraint_issues = self.propagator.validate_constraint_satisfaction(subtask, current_dag)
                issues.extend(constraint_issues)
            except Exception as e:
                issues.append({
                    "severity": "high",
                    "message": f"Constraint validation error: {str(e)}\nTraceback: {traceback.format_exc()}",
                    "confidence": 0.0
                })

        # 检查关键问题
        critical_issues = [i for i in issues if i["severity"] == "critical"]
        if critical_issues:
            return False, issues, applied_rule_ids

        return len(issues) == 0, issues, applied_rule_ids

    def record_validation_result(
            self,
            subtask: Any,
            main_task: str,
            task_type: str,
            current_dag: Any,
            is_valid: bool,
            issues: List[Dict],
            applied_rule_ids: List[str],
            experience_id: int
    ) -> None:
        """记录验证结果用于后续学习"""
        current_time = time.time()
        exp_similarity = 0.7

        # 更新应用规则的置信度
        for rule_id in applied_rule_ids:
            if rule_id in self.active_rules:
                is_rule_valid = not any(i["rule_id"] == rule_id for i in issues)
                self.update_rule_confidence(
                    rule_id, is_rule_valid, exp_similarity, experience_id, task_type
                )

        # 更新约束置信度
        for issue in issues:
            if "constraint_id" in issue and self.propagator:
                constraint_id = issue["constraint_id"]
                constraint = current_dag.constraints.get(constraint_id)
                if constraint:
                    constraint.update_confidence(False, constraint.value, experience_id)

            elif issue.get("rule_id") in self.active_rules:
                rule = self.active_rules[issue["rule_id"]]
                for constraint_id in rule.required_constraints:
                    if constraint_id in current_dag.constraints:
                        current_dag.constraints[constraint_id].update_confidence(
                            False, current_dag.constraints[constraint_id].value, experience_id
                        )

        # 更新成功规则和约束
        if is_valid:
            for rule_id in applied_rule_ids:
                if rule_id in self.active_rules:
                    self.update_rule_confidence(
                        rule_id, True, exp_similarity, experience_id, task_type
                    )

            for constraint_id, constraint in current_dag.constraints.items():
                constraint.update_confidence(True, constraint.value, experience_id)

    def update_from_experience(self, experience: 'Experience') -> None:
        """从经验中更新规则和约束"""
        # 检查经验中是否有应用的规则
        if hasattr(experience, 'applied_rules'):
            for rule_id in experience.applied_rules:
                is_valid = rule_id not in getattr(experience, 'violated_rules', [])
                context_similarity = 0.7  # 可以根据实际情况计算
                self.update_rule_confidence(
                    rule_id,
                    is_valid,
                    context_similarity,
                    experience.task_id,
                    experience.task_type
                )

        # 更新约束
        if self.propagator and hasattr(experience, 'subtask_dag'):
            for constraint_id, constraint in experience.subtask_dag.constraints.items():
                is_satisfied = constraint_id not in getattr(experience, 'broken_constraints', [])
                constraint.update_confidence(
                    is_satisfied,
                    constraint.value,
                    experience.task_id
                )