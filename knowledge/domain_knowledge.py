from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Callable, Any, TYPE_CHECKING
from enum import Enum
import numpy as np
import time
from dataclasses import dataclass, field
import difflib

from models.llm_manager import LLMManager
from knowledge.validation_rules import ValidationRule, TaskConstraint, SeverityLevel, RuleEvidence, ConstraintEvidence
from core.types import Subtask, SubtaskDAG
from experience.experience_pool import ExperiencePool, Experience

if TYPE_CHECKING:
    from core.types import Subtask, SubtaskDAG
    from knowledge.validation_rules import ValidationRule, TaskConstraint
    from experience.experience_pool import ExperiencePool, Experience


class DomainKnowledgeBase:
    def __init__(self):
        self.domain_rules: Dict[str, List[ValidationRule]] = defaultdict(list)
        self.domain_constraints: Dict[str, List[TaskConstraint]] = defaultdict(list)
        self.rule_dependencies: Dict[str, List[str]] = defaultdict(list)
        self.constraint_propagation: Dict[str, Callable] = {}

        # 添加检查，确保所有规则检查函数存在
        self._validate_rule_check_functions()

        self._initialize_general_rules()
        self._initialize_data_analysis_rules()
        self._initialize_code_generation_rules()
        self._initialize_translation_rules()
        self._initialize_story_rules()
        self._initialize_math_rules()
        self._initialize_summary_rules()
        self._initialize_combined_rules()

        # 再次验证
        self._validate_rule_check_functions()

    def _validate_rule_check_functions(self):
        """验证所有规则检查函数是否存在"""
        required_check_functions = [
            '_check_acyclic',
            '_check_dependencies_exist',
            '_check_progression',
            '_check_redundancy',
            # 数据分析
            '_check_da_cleaning_before_analysis',
            '_check_da_eda_before_modeling',
            # 代码生成
            '_check_code_requirements_before_impl',
            '_check_code_tests_after_impl',
            # 翻译
            '_check_trans_source_lang',
            '_check_trans_quality_check',
            # 故事
            '_check_story_setup_before_plot',
            '_check_story_outline_before_detail',
            # 数学
            '_check_math_understanding_before_solution',
            '_check_math_verification_after_solution',
            # 摘要
            '_check_summary_analysis_before_writing',
            '_check_summary_keypoints_before_final',
            # 组合
            '_check_combined_task_ordering'
        ]

        missing_functions = []
        for func_name in required_check_functions:
            if not hasattr(self, func_name):
                missing_functions.append(func_name)

        if missing_functions:
            raise AttributeError(
                f"DomainKnowledgeBase初始化不完整: "
                f"缺少规则检查函数 {', '.join(missing_functions)}"
            )

    def _initialize_general_rules(self):
        self.add_rule(ValidationRule(
            id="struct-rule-1",
            description="DAG must be acyclic",
            category="structural",
            severity=SeverityLevel.CRITICAL,
            check_func=lambda subtask, main, dag, constraints: self._check_acyclic(subtask, dag),
            required_constraints=[]
        ))

        self.add_rule(ValidationRule(
            id="struct-rule-2",
            description="All dependencies must exist in DAG",
            category="structural",
            severity=SeverityLevel.CRITICAL,
            check_func=lambda subtask, main, dag, constraints: self._check_dependencies_exist(subtask, dag),
            required_constraints=[]
        ))

        self.add_rule(ValidationRule(
            id="gen-rule-1",
            description="Subtask must advance toward main goal",
            category="general",
            severity=SeverityLevel.MEDIUM,
            check_func=lambda subtask, main, dag, constraints: self._check_progression(subtask, main, dag),
            required_constraints=[]
        ))

        self.add_rule(ValidationRule(
            id="gen-rule-2",
            description="No redundant subtasks",
            category="general",
            severity=SeverityLevel.MEDIUM,
            check_func=lambda subtask, main, dag, constraints: self._check_redundancy(subtask, dag),
            required_constraints=[]
        ))

        self.add_rule(ValidationRule(
            id="cg-rule-parallel-1",
            description="Design and environment setup can run in parallel",
            category="domain",
            domain="code_generation",
            severity=SeverityLevel.MEDIUM,
            check_func=lambda subtask, main, dag, constraints: self._check_parallel_design_env(subtask, dag),
            required_constraints=[]
        ))

        self.add_rule(ValidationRule(
            id="cg-rule-parallel-2",
            description="Implementation should depend on both design and environment setup",
            category="domain",
            domain="code_generation",
            severity=SeverityLevel.HIGH,
            check_func=lambda subtask, main, dag, constraints: self._check_implementation_dependencies(subtask, dag),
            required_constraints=[]
        ))

    def _initialize_translation_rules(self):
        """初始化翻译任务的规则和约束"""
        self.add_rule(ValidationRule(
            id="trans-rule-1",
            description="Source language must be identified before translation",
            category="domain",
            domain="translation",
            severity=SeverityLevel.HIGH,
            check_func=lambda subtask, main, dag, constraints: self._check_trans_source_lang(subtask, dag),
            required_constraints=["trans-constraint-1"]
        ))
        self.add_rule(ValidationRule(
            id="trans-rule-2",
            description="Quality check must follow translation",
            category="domain",
            domain="translation",
            severity=SeverityLevel.MEDIUM,
            check_func=lambda subtask, main, dag, constraints: self._check_trans_quality_check(subtask, dag),
            required_constraints=["trans-constraint-2"]
        ))
        self.add_constraint("translation", TaskConstraint(
            id="trans-constraint-1",
            description="Source language identification before translation",
            type="dependency",
            value="lang_identification → translation",
            source="domain_rule"
        ))
        self.add_constraint("translation", TaskConstraint(
            id="trans-constraint-2",
            description="Quality review after translation",
            type="dependency",
            value="translation → quality_check",
            source="domain_rule"
        ))
        self.constraint_propagation["trans-constraint-1"] = self._propagate_trans_lang_constraint
        self.constraint_propagation["trans-constraint-2"] = self._propagate_trans_quality_constraint

    def _initialize_story_rules(self):
        """初始化故事创作任务的规则和约束"""
        self.add_rule(ValidationRule(
            id="story-rule-1",
            description="Character and setting development must precede plot writing",
            category="domain",
            domain="story",
            severity=SeverityLevel.HIGH,
            check_func=lambda subtask, main, dag, constraints: self._check_story_setup_before_plot(subtask, dag),
            required_constraints=["story-constraint-1"]
        ))
        self.add_rule(ValidationRule(
            id="story-rule-2",
            description="Plot outline must be created before detailed writing",
            category="domain",
            domain="story",
            severity=SeverityLevel.HIGH,
            check_func=lambda subtask, main, dag, constraints: self._check_story_outline_before_detail(subtask, dag),
            required_constraints=["story-constraint-2"]
        ))
        self.add_constraint("story", TaskConstraint(
            id="story-constraint-1",
            description="Setup development before plot",
            type="dependency",
            value="setup → plot",
            source="domain_rule"
        ))
        self.add_constraint("story", TaskConstraint(
            id="story-constraint-2",
            description="Outline before detailed writing",
            type="dependency",
            value="outline → detailed_writing",
            source="domain_rule"
        ))
        self.constraint_propagation["story-constraint-1"] = self._propagate_story_setup_constraint
        self.constraint_propagation["story-constraint-2"] = self._propagate_story_outline_constraint

    def _initialize_math_rules(self):
        """初始化数学问题任务的规则和约束"""
        self.add_rule(ValidationRule(
            id="math-rule-1",
            description="Problem understanding must precede solution steps",
            category="domain",
            domain="math_problem",
            severity=SeverityLevel.HIGH,
            check_func=lambda subtask, main, dag, constraints: self._check_math_understanding_before_solution(subtask,
                                                                                                              dag),
            required_constraints=["math-constraint-1"]
        ))
        self.add_rule(ValidationRule(
            id="math-rule-2",
            description="Verification step should follow solution",
            category="domain",
            domain="math_problem",
            severity=SeverityLevel.MEDIUM,
            check_func=lambda subtask, main, dag, constraints: self._check_math_verification_after_solution(subtask,
                                                                                                            dag),
            required_constraints=["math-constraint-2"]
        ))
        self.add_constraint("math_problem", TaskConstraint(
            id="math-constraint-1",
            description="Understanding before solution",
            type="dependency",
            value="understanding → solution",
            source="domain_rule"
        ))
        self.add_constraint("math_problem", TaskConstraint(
            id="math-constraint-2",
            description="Verification after solution",
            type="dependency",
            value="solution → verification",
            source="domain_rule"
        ))
        self.constraint_propagation["math-constraint-1"] = self._propagate_math_understanding_constraint
        self.constraint_propagation["math-constraint-2"] = self._propagate_math_verification_constraint

    def _initialize_summary_rules(self):
        """初始化摘要任务的规则和约束"""
        self.add_rule(ValidationRule(
            id="summary-rule-1",
            description="Source text analysis must precede summarization",
            category="domain",
            domain="summary",
            severity=SeverityLevel.HIGH,
            check_func=lambda subtask, main, dag, constraints: self._check_summary_analysis_before_writing(subtask,
                                                                                                           dag),
            required_constraints=["summary-constraint-1"]
        ))
        self.add_rule(ValidationRule(
            id="summary-rule-2",
            description="Key points extraction must happen before final summary",
            category="domain",
            domain="summary",
            severity=SeverityLevel.HIGH,
            check_func=lambda subtask, main, dag, constraints: self._check_summary_keypoints_before_final(subtask, dag),
            required_constraints=["summary-constraint-2"]
        ))
        self.add_constraint("summary", TaskConstraint(
            id="summary-constraint-1",
            description="Text analysis before summarization",
            type="dependency",
            value="analysis → summarization",
            source="domain_rule"
        ))
        self.add_constraint("summary", TaskConstraint(
            id="summary-constraint-2",
            description="Key points extraction before final summary",
            type="dependency",
            value="keypoints → final_summary",
            source="domain_rule"
        ))
        self.constraint_propagation["summary-constraint-1"] = self._propagate_summary_analysis_constraint
        self.constraint_propagation["summary-constraint-2"] = self._propagate_summary_keypoints_constraint

    def _initialize_combined_rules(self):
        """初始化组合任务的规则和约束"""
        self.add_rule(ValidationRule(
            id="combined-rule-1",
            description="Subtasks must be properly ordered according to their types",
            category="domain",
            domain="combined_task",
            severity=SeverityLevel.HIGH,
            check_func=lambda subtask, main, dag, constraints: self._check_combined_task_ordering(subtask, dag),
            required_constraints=["combined-constraint-1"]
        ))
        self.add_constraint("combined_task", TaskConstraint(
            id="combined-constraint-1",
            description="Respect domain-specific ordering in combined tasks",
            type="dependency",
            value="domain_ordered_execution",
            source="domain_rule"
        ))
        self.constraint_propagation["combined-constraint-1"] = self._propagate_combined_ordering_constraint

    def _initialize_data_analysis_rules(self):
        self.add_rule(ValidationRule(
            id="da-rule-1",
            description="Data cleaning must precede analysis",
            category="domain",
            domain="data_analysis",
            severity=SeverityLevel.HIGH,
            check_func=lambda subtask, main, dag, constraints: self._check_da_cleaning_before_analysis(subtask, dag),
            required_constraints=["da-constraint-1"]
        ))

        self.add_rule(ValidationRule(
            id="da-rule-2",
            description="Exploratory analysis must precede modeling",
            category="domain",
            domain="data_analysis",
            severity=SeverityLevel.HIGH,
            check_func=lambda subtask, main, dag, constraints: self._check_da_eda_before_modeling(subtask, dag),
            required_constraints=["da-constraint-2"]
        ))

        self.add_constraint("data_analysis", TaskConstraint(
            id="da-constraint-1",
            description="Data must be cleaned before analysis",
            type="dependency",
            value="cleaning → analysis",
            source="domain_rule"
        ))

        self.add_constraint("data_analysis", TaskConstraint(
            id="da-constraint-2",
            description="Exploratory analysis must be performed before modeling",
            type="dependency",
            value="eda → modeling",
            source="domain_rule"
        ))

        self.constraint_propagation["da-constraint-1"] = self._propagate_da_cleaning_constraint
        self.constraint_propagation["da-constraint-2"] = self._propagate_da_eda_constraint

    def _initialize_code_generation_rules(self):
        self.add_rule(ValidationRule(
            id="code-rule-1",
            description="Requirements analysis must precede implementation",
            category="domain",
            domain="code_generation",
            severity=SeverityLevel.HIGH,
            check_func=lambda subtask, main, dag, constraints: self._check_code_requirements_before_impl(subtask, dag),
            required_constraints=["code-constraint-1"]
        ))

        self.add_rule(ValidationRule(
            id="code-rule-2",
            description="Unit tests should follow implementation",
            category="domain",
            domain="code_generation",
            severity=SeverityLevel.MEDIUM,
            check_func=lambda subtask, main, dag, constraints: self._check_code_tests_after_impl(subtask, dag),
            required_constraints=["code-constraint-2"]
        ))

        self.add_constraint("code_generation", TaskConstraint(
            id="code-constraint-1",
            description="Requirements must be analyzed before coding",
            type="dependency",
            value="requirements → implementation",
            source="domain_rule"
        ))

        self.add_constraint("code_generation", TaskConstraint(
            id="code-constraint-2",
            description="Unit tests should be written after implementation",
            type="dependency",
            value="implementation → tests",
            source="domain_rule"
        ))

        self.constraint_propagation["code-constraint-1"] = self._propagate_code_requirements_constraint
        self.constraint_propagation["code-constraint-2"] = self._propagate_code_tests_constraint

    def add_rule(self, rule: ValidationRule) -> None:
        self.domain_rules[rule.domain or "general"].append(rule)

    def add_constraint(self, domain: str, constraint: TaskConstraint) -> None:
        self.domain_constraints[domain].append(constraint)

    def get_rules_for_domain(self, domain: str) -> List[ValidationRule]:
        general_rules = self.domain_rules["general"]
        domain_rules = self.domain_rules.get(domain, [])
        return general_rules + domain_rules

    def get_constraints_for_domain(self, domain: str) -> List[TaskConstraint]:
        general_constraints = []
        domain_constraints = self.domain_constraints.get(domain, [])
        return general_constraints + domain_constraints

    def propagate_constraint(self, constraint_id: str, parent_constraint: TaskConstraint,
                             subtask: Subtask, current_dag: SubtaskDAG) -> Optional[TaskConstraint]:
        if constraint_id in self.constraint_propagation:
            return self.constraint_propagation[constraint_id](parent_constraint, subtask, current_dag)
        return None



    # ===== 规则检查函数 =====
    def _check_acyclic(self, subtask: Subtask, dag: SubtaskDAG) -> Tuple[bool, str]:
        """检查DAG是否无环 - 添加详细调试信息"""
        # 创建测试DAG
        test_dag = SubtaskDAG(
            nodes=dag.nodes.copy(),
            entry_points=dag.entry_points.copy(),
            exit_points=dag.exit_points.copy(),
            constraints=dag.constraints.copy()
        )
        test_dag.nodes[subtask.id] = subtask

        # 添加详细调试信息
        print("\nDEBUG: 检查循环依赖")
        print(f"当前DAG节点: {list(dag.nodes.keys())}")
        print(f"当前DAG依赖: {[f'{node_id} -> {node.dependencies}' for node_id, node in dag.nodes.items()]}")
        print(f"新子任务: {subtask.id} -> {subtask.dependencies}")

        # 改进的Kahn算法
        in_degree = {node_id: 0 for node_id in test_dag.nodes}
        for node_id, node in test_dag.nodes.items():
            for dep in node.dependencies:
                if dep in in_degree and dep != node_id:
                    in_degree[node_id] += 1

        print(f"入度计算: {in_degree}")

        # 创建没有入边的节点队列
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        visited_count = 0
        visited_nodes = []

        print(f"初始队列: {queue}")

        while queue:
            node_id = queue.pop(0)
            visited_count += 1
            visited_nodes.append(node_id)

            print(f"访问节点: {node_id}, 当前计数: {visited_count}")

            # 减少依赖此节点的所有节点的入度
            for current_id, current_node in test_dag.nodes.items():
                if node_id in current_node.dependencies:
                    in_degree[current_id] -= 1
                    if in_degree[current_id] == 0:
                        queue.append(current_id)
                        print(f"  节点 {current_id} 入度为0，加入队列")

        print(f"访问节点总数: {visited_count}, 总节点数: {len(test_dag.nodes)}")

        # 如果访问的节点数少于总节点数，说明有循环
        if visited_count != len(test_dag.nodes):
            # 尝试找出具体循环
            cycles = []
            unvisited = set(test_dag.nodes.keys()) - set(visited_nodes)

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

                        for next_node, n in test_dag.nodes.items():
                            if node in n.dependencies:
                                find_cycle(next_node)

                        path.pop()
                        visited.remove(node)

                    try:
                        find_cycle(start_node)
                    except RecursionError:
                        cycles.append(["...循环路径过长..."])

            cycle_str = "detected cycles" if not cycles else f"cycles: {cycles}"
            return False, f"Adding this subtask would create {cycle_str}"

        return True, "No cycles detected"

    def _check_parallel_design_env(self, subtask: Subtask, dag: SubtaskDAG) -> Tuple[bool, str]:
        """检查设计和环境准备是否可以并行执行"""
        has_requirements = any(node.task_type == "requirements" for node in dag.nodes.values())

        # 如果有需求分析，但没有并行的设计和环境准备
        if has_requirements:
            has_design = any(node.task_type == "design" for node in dag.nodes.values())
            has_env = any(node.task_type == "environment_setup" for node in dag.nodes.values())

            # 如果两者都不存在，或者只有一个存在
            if not (has_design and has_env):
                return False, "Design and environment setup should run in parallel after requirements"

        return True, "Design and environment setup correctly parallelized"

    def _check_implementation_dependencies(self, subtask: Subtask, dag: SubtaskDAG) -> Tuple[bool, str]:
        """检查实现任务是否依赖设计和环境准备"""
        if subtask.task_type == "implementation":
            design_dep = any(dep in [node.id for node in dag.nodes.values() if node.task_type == "design"]
                             for dep in subtask.dependencies)
            env_dep = any(dep in [node.id for node in dag.nodes.values() if node.task_type == "environment_setup"]
                          for dep in subtask.dependencies)

            if not design_dep or not env_dep:
                return False, "Implementation should depend on both design and environment setup"

        return True, "Implementation has correct dependencies"

    # ===== 通用规则检查函数 =====
    def _check_dependencies_exist(self, subtask: Subtask, dag: SubtaskDAG) -> Tuple[bool, str]:
        """检查所有依赖是否存在于DAG中"""
        for dep_id in subtask.dependencies:
            if dep_id not in dag.nodes and dep_id != subtask.id:
                return False, f"Dependency '{dep_id}' does not exist in the current DAG"
        return True, "All dependencies exist in DAG"

    def _check_progression(self, subtask: Subtask, main_task: str, dag: SubtaskDAG) -> Tuple[bool, str]:
        """检查子任务是否推进主任务"""
        try:
            main_words = set(main_task.lower().split())
            task_words = set(subtask.description.lower().split())

            overlap = len(main_words & task_words) / max(len(main_words), 1)

            if overlap < 0.1 and dag.nodes:
                last_task = list(dag.nodes.values())[-1]
                last_words = set(last_task.description.lower().split())
                continuity = len(task_words & last_words) / max(len(last_task.description.split()), 1)
                if continuity < 0.2:
                    return False, "Subtask does not seem related to main goal or previous steps"

            return True, "Subtask advances toward main goal"
        except Exception as e:
            return True, f"Progression check bypassed due to error: {str(e)}"

    def _check_redundancy(self, subtask: Subtask, dag: SubtaskDAG) -> Tuple[bool, str]:
        """检查子任务是否冗余 - 降低敏感度并改进计算"""
        for existing_id, existing_task in dag.nodes.items():
            if existing_id == subtask.id:
                continue

            # 计算描述相似度 - 使用更精确的方法
            desc1 = subtask.description.lower()
            desc2 = existing_task.description.lower()

            # 使用词频而非简单集合交集
            words1 = desc1.split()
            words2 = desc2.split()

            # 计算Jaccard相似度，但考虑词频
            word_count1 = {}
            word_count2 = {}

            for word in words1:
                word_count1[word] = word_count1.get(word, 0) + 1
            for word in words2:
                word_count2[word] = word_count2.get(word, 0) + 1

            # 计算加权交集
            intersection = 0
            union = len(words1) + len(words2)

            for word in set(word_count1.keys()) | set(word_count2.keys()):
                intersection += min(word_count1.get(word, 0), word_count2.get(word, 0))

            # 计算加权Jaccard相似度
            if union > 0:
                similarity = intersection / union
            else:
                similarity = 0

            # 提高冗余阈值，避免误判
            if similarity > 0.85:  # 从0.7提高到0.85
                return False, f"Redundant with existing subtask '{existing_id}' (similarity: {similarity:.2f}): {existing_task.description}"

            # 额外检查：确保任务类型不完全相同
            if subtask.task_type == existing_task.task_type and similarity > 0.6:
                return False, f"Task type '{subtask.task_type}' already exists with high similarity"

        return True, "No redundancy detected"

    # ===== 数据分析领域特定检查 =====
    def _check_da_cleaning_before_analysis(self, subtask: Subtask, dag: SubtaskDAG) -> Tuple[bool, str]:
        """改进数据分析规则检查 - 更灵活的清洗步骤检测"""
        if subtask.task_type in ["analysis", "modeling", "reporting"]:
            # 检查是否有任何清洗或预处理步骤
            has_cleaning = any(
                "clean" in node.task_type.lower() or
                "preprocess" in node.task_type.lower() or
                "preparation" in node.task_type.lower() or
                "wrangle" in node.task_type.lower()
                for node in dag.nodes.values()
            )

            if not has_cleaning:
                return False, "Data analysis requires prior data cleaning or preprocessing step"
        return True, "Data cleaning/preprocessing precedes analysis"

    def _check_da_eda_before_modeling(self, subtask: Subtask, dag: SubtaskDAG) -> Tuple[bool, str]:
        if subtask.task_type == "modeling":
            has_eda = any(node.task_type == "eda" for node in dag.nodes.values())
            if not has_eda:
                return False, "Modeling requires prior exploratory data analysis"
        return True, "Exploratory analysis precedes modeling"

    # ===== 代码生成领域特定检查 =====
    def _check_code_requirements_before_impl(self, subtask: Subtask, dag: SubtaskDAG) -> Tuple[bool, str]:
        if subtask.task_type == "implementation":
            has_requirements = any(node.task_type == "requirements" for node in dag.nodes.values())
            if not has_requirements:
                return False, "Code implementation requires prior requirements analysis"
        return True, "Requirements analysis precedes implementation"

    def _check_code_tests_after_impl(self, subtask: Subtask, dag: SubtaskDAG) -> Tuple[bool, str]:
        if subtask.task_type == "testing" and "unit" in subtask.description.lower():
            has_impl = any(node.task_type == "implementation" for node in dag.nodes.values())
            if not has_impl:
                return False, "Unit tests should follow implementation, not precede it"
        return True, "Unit tests follow implementation"

    # ===== 翻译任务规则检查函数 =====
    def _check_trans_source_lang(self, subtask: Subtask, dag: SubtaskDAG) -> Tuple[bool, str]:
        """检查翻译任务中源语言识别是否在翻译前"""
        if subtask.task_type == "translation":
            has_lang_ident = any(node.task_type == "lang_identification" for node in dag.nodes.values())
            if not has_lang_ident:
                return False, "Translation requires prior language identification"
        return True, "Source language identified before translation"

    def _check_trans_quality_check(self, subtask: Subtask, dag: SubtaskDAG) -> Tuple[bool, str]:
        """检查翻译任务中质量检查是否在翻译后"""
        if subtask.task_type == "quality_check":
            has_translation = any(node.task_type == "translation" for node in dag.nodes.values())
            if not has_translation:
                return False, "Quality check requires completed translation"
        return True, "Quality check follows translation"

    # ===== 故事创作任务规则检查函数 =====
    def _check_story_setup_before_plot(self, subtask: Subtask, dag: SubtaskDAG) -> Tuple[bool, str]:
        """检查故事创作中设定开发是否在情节写作前"""
        if subtask.task_type in ["plot", "detailed_writing", "revision"]:
            has_setup = any(node.task_type == "setup" for node in dag.nodes.values())
            if not has_setup:
                return False, "Plot writing requires prior story setup"
        return True, "Story setup precedes plot development"

    def _check_story_outline_before_detail(self, subtask: Subtask, dag: SubtaskDAG) -> Tuple[bool, str]:
        """检查故事创作中大纲是否在详细写作前"""
        if subtask.task_type in ["detailed_writing", "revision"]:
            has_outline = any(node.task_type == "outline" for node in dag.nodes.values())
            if not has_outline:
                return False, "Detailed writing requires prior plot outline"
        return True, "Plot outline precedes detailed writing"

    # ===== 数学问题任务规则检查函数 =====
    def _check_math_understanding_before_solution(self, subtask: Subtask, dag: SubtaskDAG) -> Tuple[bool, str]:
        """检查数学问题中理解是否在求解前"""
        if subtask.task_type in ["solution", "verification"]:
            has_understanding = any(node.task_type == "understanding" for node in dag.nodes.values())
            if not has_understanding:
                return False, "Solution requires prior problem understanding"
        return True, "Problem understanding precedes solution"

    def _check_math_verification_after_solution(self, subtask: Subtask, dag: SubtaskDAG) -> Tuple[bool, str]:
        """检查数学问题中验证是否在求解后"""
        if subtask.task_type == "verification":
            has_solution = any(node.task_type == "solution" for node in dag.nodes.values())
            if not has_solution:
                return False, "Verification requires completed solution"
        return True, "Verification follows solution"

    # ===== 摘要任务规则检查函数 =====
    def _check_summary_analysis_before_writing(self, subtask: Subtask, dag: SubtaskDAG) -> Tuple[bool, str]:
        """检查摘要任务中分析是否在摘要前"""
        if subtask.task_type in ["keypoints", "final_summary"]:
            has_analysis = any(node.task_type == "analysis" for node in dag.nodes.values())
            if not has_analysis:
                return False, "Summarization requires prior text analysis"
        return True, "Text analysis precedes summarization"

    def _check_summary_keypoints_before_final(self, subtask: Subtask, dag: SubtaskDAG) -> Tuple[bool, str]:
        """检查摘要任务中关键点提取是否在最终摘要前"""
        if subtask.task_type == "final_summary":
            has_keypoints = any(node.task_type == "keypoints" for node in dag.nodes.values())
            if not has_keypoints:
                return False, "Final summary requires prior key points extraction"
        return True, "Key points extraction precedes final summary"

    # ===== 组合任务规则检查函数 =====
    def _check_combined_task_ordering(self, subtask: Subtask, dag: SubtaskDAG) -> Tuple[bool, str]:
        """检查组合任务中子任务的顺序"""
        # 检查是否有违反领域特定顺序的情况
        if dag.nodes and subtask.task_type != "setup":
            last_task = list(dag.nodes.values())[-1]
            # 简单检查：确保任务类型不同（避免冗余）
            if subtask.task_type == last_task.task_type:
                return False, "Redundant task type in combined task"

        # 检查是否有必要的前置任务
        if "visualization" in subtask.task_type and "data_analysis" in subtask.description.lower():
            has_analysis = any("analysis" in node.task_type.lower() for node in dag.nodes.values())
            if not has_analysis:
                return False, "Visualization requires prior data analysis"

        return True, "Subtasks properly ordered"

    # ===== 数据分析领域约束传播方法 =====
    def _propagate_da_cleaning_constraint(self, parent_constraint: TaskConstraint,
                                          subtask: Subtask, dag: SubtaskDAG) -> Optional[TaskConstraint]:
        """传播数据分析的清洗约束"""
        if subtask.task_type in ["analysis", "modeling", "reporting"]:
            return TaskConstraint(
                id=f"{subtask.id}_cleaning_req",
                description="Requires cleaned data as input",
                type="dependency",
                value="cleaning",
                source="constraint_propagation",
                propagation_level=parent_constraint.propagation_level + 1
            )
        return None

    def _propagate_da_eda_constraint(self, parent_constraint: TaskConstraint,
                                     subtask: Subtask, dag: SubtaskDAG) -> Optional[TaskConstraint]:
        """传播数据分析的EDA约束"""
        if subtask.task_type == "modeling":
            return TaskConstraint(
                id=f"{subtask.id}_eda_req",
                description="Requires exploratory analysis results",
                type="dependency",
                value="eda",
                source="constraint_propagation",
                propagation_level=parent_constraint.propagation_level + 1
            )
        return None

    # ===== 代码生成领域约束传播方法 =====
    def _propagate_code_requirements_constraint(self, parent_constraint: TaskConstraint,
                                                subtask: Subtask, dag: SubtaskDAG) -> Optional[TaskConstraint]:
        """传播代码生成的需求约束"""
        if subtask.task_type == "implementation":
            return TaskConstraint(
                id=f"{subtask.id}_requirements_req",
                description="Requires completed requirements specification",
                type="dependency",
                value="requirements",
                source="constraint_propagation",
                propagation_level=parent_constraint.propagation_level + 1
            )
        return None

    def _propagate_code_tests_constraint(self, parent_constraint: TaskConstraint,
                                         subtask: Subtask, dag: SubtaskDAG) -> Optional[TaskConstraint]:
        """传播代码生成的测试约束"""
        if subtask.task_type == "testing" and "unit" in subtask.description.lower():
            return TaskConstraint(
                id=f"{subtask.id}_impl_req",
                description="Requires completed implementation for testing",
                type="dependency",
                value="implementation",
                source="constraint_propagation",
                propagation_level=parent_constraint.propagation_level + 1
            )
        return None

    # ===== 翻译任务约束传播方法 =====
    def _propagate_trans_lang_constraint(self, parent_constraint: TaskConstraint,
                                         subtask: Subtask, dag: SubtaskDAG) -> Optional[TaskConstraint]:
        """传播翻译任务的语言约束"""
        if subtask.task_type in ["translation", "quality_check"]:
            return TaskConstraint(
                id=f"{subtask.id}_lang_req",
                description="Requires language identification as input",
                type="dependency",
                value="lang_identification",
                source="constraint_propagation",
                propagation_level=parent_constraint.propagation_level + 1
            )
        return None

    def _propagate_trans_quality_constraint(self, parent_constraint: TaskConstraint,
                                            subtask: Subtask, dag: SubtaskDAG) -> Optional[TaskConstraint]:
        """传播翻译任务的质量检查约束"""
        if subtask.task_type == "quality_check":
            return TaskConstraint(
                id=f"{subtask.id}_translation_req",
                description="Requires completed translation for quality check",
                type="dependency",
                value="translation",
                source="constraint_propagation",
                propagation_level=parent_constraint.propagation_level + 1
            )
        return None

    # ===== 故事创作任务约束传播方法 =====
    def _propagate_story_setup_constraint(self, parent_constraint: TaskConstraint,
                                          subtask: Subtask, dag: SubtaskDAG) -> Optional[TaskConstraint]:
        """传播故事创作的设定约束"""
        if subtask.task_type in ["plot", "detailed_writing", "revision"]:
            return TaskConstraint(
                id=f"{subtask.id}_setup_req",
                description="Requires story setup as input",
                type="dependency",
                value="setup",
                source="constraint_propagation",
                propagation_level=parent_constraint.propagation_level + 1
            )
        return None

    def _propagate_story_outline_constraint(self, parent_constraint: TaskConstraint,
                                            subtask: Subtask, dag: SubtaskDAG) -> Optional[TaskConstraint]:
        """传播故事创作的大纲约束"""
        if subtask.task_type in ["detailed_writing", "revision"]:
            return TaskConstraint(
                id=f"{subtask.id}_outline_req",
                description="Requires plot outline as input",
                type="dependency",
                value="outline",
                source="constraint_propagation",
                propagation_level=parent_constraint.propagation_level + 1
            )
        return None

    # ===== 数学问题任务约束传播方法 =====
    def _propagate_math_understanding_constraint(self, parent_constraint: TaskConstraint,
                                                 subtask: Subtask, dag: SubtaskDAG) -> Optional[TaskConstraint]:
        """传播数学问题的理解约束"""
        if subtask.task_type in ["solution", "verification"]:
            return TaskConstraint(
                id=f"{subtask.id}_understanding_req",
                description="Requires problem understanding as input",
                type="dependency",
                value="understanding",
                source="constraint_propagation",
                propagation_level=parent_constraint.propagation_level + 1
            )
        return None

    def _propagate_math_verification_constraint(self, parent_constraint: TaskConstraint,
                                                subtask: Subtask, dag: SubtaskDAG) -> Optional[TaskConstraint]:
        """传播数学问题的验证约束"""
        if subtask.task_type == "verification":
            return TaskConstraint(
                id=f"{subtask.id}_solution_req",
                description="Requires completed solution for verification",
                type="dependency",
                value="solution",
                source="constraint_propagation",
                propagation_level=parent_constraint.propagation_level + 1
            )
        return None

    # ===== 摘要任务约束传播方法 =====
    def _propagate_summary_analysis_constraint(self, parent_constraint: TaskConstraint,
                                               subtask: Subtask, dag: SubtaskDAG) -> Optional[TaskConstraint]:
        """传播摘要任务的分析约束"""
        if subtask.task_type in ["keypoints", "final_summary"]:
            return TaskConstraint(
                id=f"{subtask.id}_analysis_req",
                description="Requires text analysis as input",
                type="dependency",
                value="analysis",
                source="constraint_propagation",
                propagation_level=parent_constraint.propagation_level + 1
            )
        return None

    def _propagate_summary_keypoints_constraint(self, parent_constraint: TaskConstraint,
                                                subtask: Subtask, dag: SubtaskDAG) -> Optional[TaskConstraint]:
        """传播摘要任务的关键点约束"""
        if subtask.task_type == "final_summary":
            return TaskConstraint(
                id=f"{subtask.id}_keypoints_req",
                description="Requires key points extraction as input",
                type="dependency",
                value="keypoints",
                source="constraint_propagation",
                propagation_level=parent_constraint.propagation_level + 1
            )
        return None

    # ===== 组合任务约束传播方法 =====
    def _propagate_combined_ordering_constraint(self, parent_constraint: TaskConstraint,
                                                subtask: Subtask, dag: SubtaskDAG) -> Optional[TaskConstraint]:
        """传播组合任务的顺序约束"""
        # 组合任务的约束传播
        if dag.nodes and subtask.task_type != "setup":
            last_task = list(dag.nodes.values())[-1]
            return TaskConstraint(
                id=f"{subtask.id}_ordering_req",
                description="Requires proper domain-specific ordering",
                type="dependency",
                value=last_task.task_type,
                source="constraint_propagation",
                propagation_level=parent_constraint.propagation_level + 1
            )
        return None