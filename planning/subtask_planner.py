from typing import Tuple, Optional, Any, List, Dict
import time
import re
from experience.experience_pool import ExperiencePool, Experience
from knowledge.validation_rules import RuleBasedChecker
from knowledge.domain_knowledge import DomainKnowledgeBase
from planning.subtask_predictor import SubtaskPredictor
from core.types import Subtask, SubtaskDAG
import difflib
import numpy as np
from collections import defaultdict


class SubtaskPlanner:
    """任务规划器，负责生成和验证子任务DAG"""

    def __init__(self, experience_pool: ExperiencePool):
        """初始化任务规划器"""
        self.experience_pool = experience_pool
        self.predictor = SubtaskPredictor(experience_pool)
        self.knowledge_base = DomainKnowledgeBase()
        self.checker = RuleBasedChecker(experience_pool, self.knowledge_base)
        self.max_prediction_attempts = 3
        # 动态子任务上限 - 基于任务难度
        self.base_max_subtasks = 10  # 基础值
        self.difficulty_factor = 1.5  # 难度系数

    def get_dynamic_max_subtasks(self, difficulty: float) -> int:
        """根据任务难度动态计算最大子任务数"""
        # 基础值 + 难度调整 (难度范围1-5)
        max_subtasks = self.base_max_subtasks + int((difficulty - 3) * self.difficulty_factor)
        # 限制在合理范围内
        return max(8, min(25, max_subtasks))

    def _initialize_domain_specific_steps(self, dag: SubtaskDAG, task_type: str, main_task: str):
        """根据任务类型添加必要的初始步骤"""
        # 仅在DAG为空时添加初始步骤
        if dag.nodes:
            return

        print(f"  * 添加{task_type}任务的初始步骤...")

        # 数据分析任务 - 创建并行路径
        if task_type == "data_analysis":
            cleaning_task = Subtask(
                id=self.predictor._generate_subtask_id(),
                description="Clean and preprocess customer churn data",
                task_type="cleaning",
                dependencies=[],
                required_resources={"tools": ["pandas", "numpy"], "data": ["raw_data.csv"]},
                expected_output="Cleaned dataset with no missing values or outliers",
                difficulty=3.5,
                estimated_time=120.0
            )
            dag.add_subtask(cleaning_task)
            print(f"    - 添加初始步骤: {cleaning_task.description}")
            
            # 添加并行的数据探索任务
            exploration_task = Subtask(
                id=self.predictor._generate_subtask_id(),
                description="Explore data structure and basic statistics",
                task_type="exploration",  # 数据探索任务
                dependencies=[],  # 与清洗任务并行
                required_resources={"tools": ["pandas", "numpy"], "data": ["raw_data.csv"]},
                expected_output="Basic statistics and data structure overview",
                difficulty=3.0,
                estimated_time=90.0
            )
            dag.add_subtask(exploration_task)
            print(f"    - 添加并行探索步骤: {exploration_task.description}")

        # 翻译任务
        elif task_type == "translation":
            lang_ident_task = Subtask(
                id=self.predictor._generate_subtask_id(),
                description="Identify source and target languages",
                task_type="lang_identification",
                dependencies=[],
                required_resources={"tools": ["language_detector"]},
                expected_output="Source: English, Target: Chinese",
                difficulty=2.0,
                estimated_time=30.0
            )
            dag.add_subtask(lang_ident_task)
            print(f"    - 添加初始步骤: {lang_ident_task.description}")

        # 代码生成任务
        elif task_type == "code_generation":
            requirements_task = Subtask(
                id=self.predictor._generate_subtask_id(),
                description="Analyze requirements and specifications",
                task_type="requirements",
                dependencies=[],
                required_resources={"tools": ["documentation"]},
                expected_output="Detailed requirements document",
                difficulty=3.0,
                estimated_time=90.0
            )
            dag.add_subtask(requirements_task)
            print(f"    - 添加初始步骤: {requirements_task.description}")

        # 数学问题
        elif task_type == "math_problem":
            understanding_task = Subtask(
                id=self.predictor._generate_subtask_id(),
                description="Understand and analyze the mathematical problem",
                task_type="understanding",
                dependencies=[],
                required_resources={"tools": ["calculator"]},
                expected_output="Clear problem statement and approach",
                difficulty=2.5,
                estimated_time=60.0
            )
            dag.add_subtask(understanding_task)
            print(f"    - 添加初始步骤: {understanding_task.description}")

        # 故事创作
        elif task_type == "story":
            setup_task = Subtask(
                id=self.predictor._generate_subtask_id(),
                description="Define story setting and characters",
                task_type="setup",
                dependencies=[],
                required_resources={"tools": ["creativity"]},
                expected_output="Story premise and main characters",
                difficulty=3.0,
                estimated_time=90.0
            )
            dag.add_subtask(setup_task)
            print(f"    - 添加初始步骤: {setup_task.description}")

        # 文本摘要
        elif task_type == "summary":
            analysis_task = Subtask(
                id=self.predictor._generate_subtask_id(),
                description="Analyze the source document",
                task_type="analysis",
                dependencies=[],
                required_resources={"tools": ["reading"]},
                expected_output="Key themes and concepts identified",
                difficulty=2.8,
                estimated_time=120.0
            )
            dag.add_subtask(analysis_task)
            print(f"    - 添加初始步骤: {analysis_task.description}")

        # 组合任务
        elif task_type == "combined_task":
            # 组合任务通常需要分析部分
            if "analyze" in main_task.lower() or "data" in main_task.lower():
                analysis_task = Subtask(
                    id=self.predictor._generate_subtask_id(),
                    description="Analyze the survey data",
                    task_type="analysis",
                    dependencies=[],
                    required_resources={"tools": ["pandas", "numpy"], "data": ["survey_data.csv"]},
                    expected_output="Key insights from survey data",
                    difficulty=3.5,
                    estimated_time=180.0
                )
                dag.add_subtask(analysis_task)
                print(f"    - 添加初始步骤: {analysis_task.description}")

    def _is_task_completed(self, dag: SubtaskDAG, main_task: str, task_type: str) -> Tuple[bool, str]:
        """通用任务完成检查 - 适用于所有任务类型"""
        if not dag.nodes:
            return False, "No subtasks generated yet"

        # 通用完成条件：有出口点且最后一个任务已执行关键操作
        has_exit_points = bool(dag.exit_points)
        last_task = list(dag.nodes.values())[-1]

        # 领域特定完成条件
        if task_type == "data_analysis":
            has_cleaning = any("clean" in node.task_type.lower() for node in dag.nodes.values())
            has_eda = any("eda" in node.task_type.lower() or "exploration" in node.task_type.lower() for node in dag.nodes.values())
            has_modeling = any("model" in node.task_type.lower() for node in dag.nodes.values())
            has_report = any("report" in node.task_type.lower() for node in dag.nodes.values())

            completed = has_cleaning and has_eda and has_modeling and has_report
            msg = "Data analysis task completed" if completed else "Missing key analysis steps"
            return completed, msg

        elif task_type == "translation":
            has_identification = any("lang" in node.task_type.lower() for node in dag.nodes.values())
            has_translation = any("trans" in node.task_type.lower() for node in dag.nodes.values())
            has_quality_check = any("quality" in node.task_type.lower() for node in dag.nodes.values())

            completed = has_identification and has_translation and has_quality_check
            msg = "Translation task completed" if completed else "Missing translation or quality check steps"
            return completed, msg

        elif task_type == "code_generation":
            has_requirements = any("requirement" in node.task_type.lower() for node in dag.nodes.values())
            has_implementation = any("implement" in node.task_type.lower() for node in dag.nodes.values())
            has_testing = any("test" in node.task_type.lower() for node in dag.nodes.values())

            completed = has_requirements and has_implementation and has_testing
            msg = "Code generation task completed" if completed else "Missing implementation or testing steps"
            return completed, msg

        elif task_type == "math_problem":
            has_understanding = any("understand" in node.task_type.lower() for node in dag.nodes.values())
            has_solution = any("solution" in node.task_type.lower() for node in dag.nodes.values())
            has_verification = any("verify" in node.task_type.lower() for node in dag.nodes.values())

            completed = has_understanding and has_solution and has_verification
            msg = "Math problem solved" if completed else "Missing solution or verification steps"
            return completed, msg

        elif task_type == "story":
            has_setup = any("setup" in node.task_type.lower() for node in dag.nodes.values())
            has_outline = any("outline" in node.task_type.lower() for node in dag.nodes.values())
            has_writing = any("writing" in node.task_type.lower() for node in dag.nodes.values())
            has_revision = any("revision" in node.task_type.lower() for node in dag.nodes.values())

            completed = has_setup and has_outline and has_writing and has_revision
            msg = "Story creation completed" if completed else "Missing story development steps"
            return completed, msg

        elif task_type == "summary":
            has_analysis = any("analysis" in node.task_type.lower() for node in dag.nodes.values())
            has_keypoints = any("keypoint" in node.task_type.lower() for node in dag.nodes.values())
            has_final = any("final" in node.task_type.lower() for node in dag.nodes.values())

            completed = has_analysis and has_keypoints and has_final
            msg = "Summary task completed" if completed else "Missing summary steps"
            return completed, msg

        elif task_type == "combined_task":
            # 检查是否有数据分析和可视化步骤
            has_analysis = any("analysis" in node.task_type.lower() for node in dag.nodes.values())
            has_visualization = any("visual" in node.task_type.lower() for node in dag.nodes.values())

            completed = has_analysis and has_visualization
            msg = "Combined task completed" if completed else "Missing analysis or visualization steps"
            return completed, msg

        # 通用完成条件（适用于其他任务类型）
        generic_completion = (
                has_exit_points and
                ("final" in last_task.description.lower() or
                 "conclusion" in last_task.description.lower() or
                 "result" in last_task.description.lower())
        )

        msg = "Task completed based on generic criteria" if generic_completion else "Awaiting final steps"
        return generic_completion, msg

    def _ensure_valid_exit_points(self, dag: SubtaskDAG):
        """确保DAG有有效的出口点"""
        if not dag.exit_points and dag.nodes:
            # 检查哪些节点没有被其他节点依赖
            potential_exit_points = []
            for node_id in dag.nodes:
                is_exit = True
                for other_id, other in dag.nodes.items():
                    if node_id in other.dependencies and other_id != node_id:
                        is_exit = False
                        break
                if is_exit:
                    potential_exit_points.append(node_id)

            # 如果找到有效的出口点，使用它们
            if potential_exit_points:
                dag.exit_points = potential_exit_points
            else:
                # 作为最后手段，将最后一个节点标记为出口点
                last_node_id = list(dag.nodes.keys())[-1]
                dag.exit_points = [last_node_id]

    def _can_skip_critical_issues(self, critical_issues: List[Dict], dag: SubtaskDAG, task_type: str) -> bool:
        """检查是否可以安全跳过关键问题"""
        for issue in critical_issues:
            # 对于循环问题，如果任务已完成则可以跳过
            if "cycle" in issue["message"].lower():
                is_completed, _ = self._is_task_completed(dag, "", task_type)
                if is_completed:
                    return True

            # 对于依赖问题，如果任务已完成则可以跳过
            elif "does not exist" in issue["message"].lower():
                is_completed, _ = self._is_task_completed(dag, "", task_type)
                if is_completed:
                    return True

        return False

    def _handle_dependency_issues(self, subtask: Subtask, dag: SubtaskDAG, issues: List[Dict]) -> List[Dict]:
        """智能处理依赖问题"""
        new_issues = []

        for issue in issues:
            # 处理无效依赖问题
            if "does not exist" in issue["message"].lower():
                # 尝试找到最可能的替代依赖
                valid_deps = []
                import re
                match = re.search(r"Dependency '(.+?)' does not exist", issue["message"])
                if match:
                    missing_dep = match.group(1)
                    # 尝试匹配相似ID
                    for node_id in dag.nodes.keys():
                        if missing_dep in node_id or node_id in missing_dep:
                            valid_deps.append(node_id)
                            break

                # 如果找到替代依赖，更新子任务
                if valid_deps:
                    subtask.dependencies = valid_deps
                    print(f"  ✅ 通过替换为有效依赖修复了问题: {valid_deps}")
                elif dag.nodes:
                    # 使用最后一个节点作为依赖
                    last_node_id = list(dag.nodes.keys())[-1]
                    subtask.dependencies = [last_node_id]
                    print(f"  ✅ 通过设置最后节点为依赖修复了问题: {last_node_id}")

            # 处理领域规则问题
            elif "data analysis requires prior data cleaning" in issue["message"].lower() and subtask.task_type in [
                "analysis", "modeling", "reporting"]:
                # 检查是否已经有清洗步骤
                has_cleaning = any(
                    "clean" in node.task_type.lower() for node in dag.nodes.values()
                )

                if not has_cleaning and dag.nodes:
                    # 在第一个位置插入清洗步骤
                    cleaning_task = Subtask(
                        id=self.predictor._generate_subtask_id(),
                        description="Clean and preprocess data",
                        task_type="cleaning",
                        dependencies=[],
                        required_resources={"tools": ["pandas", "numpy"]},
                        expected_output="Cleaned dataset ready for analysis",
                        difficulty=3.0,
                        estimated_time=90.0
                    )
                    # 临时移除现有节点，添加清洗步骤，再添加回现有节点
                    existing_nodes = list(dag.nodes.items())
                    dag.nodes.clear()
                    dag.entry_points.clear()
                    dag.exit_points.clear()

                    dag.add_subtask(cleaning_task)
                    for node_id, node in existing_nodes:
                        node.dependencies = [cleaning_task.id] if not node.dependencies else node.dependencies
                        dag.add_subtask(node)

                    print("  ✅ 自动添加了缺失的数据清洗步骤")

            else:
                new_issues.append(issue)

        return new_issues

    def plan_subtasks(
            self,
            task_id: int,
            task_type: str,
            main_task: str,
            difficulty: float
    ) -> Tuple[Optional[SubtaskDAG], str]:
        """规划任务的子任务DAG"""
        dag = SubtaskDAG(nodes={}, entry_points=[], exit_points=[])
        max_subtask_count = self.get_dynamic_max_subtasks(difficulty)

        print(f"  * 使用动态最大子任务数: {max_subtask_count} (基于难度 {difficulty}/5.0)")

        # 领域特定初始化
        self._initialize_domain_specific_steps(dag, task_type, main_task)

        all_issues = []
        task_counter = 0
        completion_check_interval = 3  # 每3个子任务检查一次完成状态

        while task_counter < max_subtask_count:
            task_counter += 1

            # 定期检查任务是否已完成
            if task_counter % completion_check_interval == 0 or task_counter == 1:
                is_completed, completion_msg = self._is_task_completed(dag, main_task, task_type)
                if is_completed:
                    print(f"  ✅ 任务完成检查: {completion_msg}")
                    break

            subtask = self.predictor.predict_next_subtask(main_task, dag, task_type)
            if not subtask:
                print("  ✅ 无更多子任务需要生成")
                break

            # 验证子任务
            is_valid, issues, _ = self.checker.validate_subtask(subtask, main_task, task_type, dag)

            # 智能处理问题
            issues = self._handle_dependency_issues(subtask, dag, issues)

            if is_valid:
                dag.add_subtask(subtask)
                continue

            all_issues.extend(issues)

            # 处理关键问题
            critical_issues = [issue for issue in issues if issue["severity"] in ["critical", "high"]]
            if critical_issues and task_counter >= self.max_prediction_attempts:
                # 检查是否可以安全跳过
                if not self._can_skip_critical_issues(critical_issues, dag, task_type):
                    return None, f"Failed to create valid DAG after {self.max_prediction_attempts} attempts. Issues: " + \
                                 ", ".join([f"{issue['severity']}:{issue['message']}" for issue in issues])
                print("  ⚠️ 跳过非致命关键问题，继续规划")

        # 最终任务完成检查
        is_completed, completion_msg = self._is_task_completed(dag, main_task, task_type)
        if not is_completed:
            print(f"  ⚠️ 任务未完全完成: {completion_msg}")
        else:
            print(f"  ✅ 任务完成: {completion_msg}")

        # 确保有有效的出口点
        self._ensure_valid_exit_points(dag)

        is_dag_valid, dag_message, _ = dag.validate_dag()
        if not is_dag_valid:
            return None, f"DAG validation failed: {dag_message}"

        return dag, f"DAG planning completed with {len(dag.nodes)} subtasks"

    def execute_and_store(
            self,
            task_id: int,
            task_type: str,
            main_task: str,
            difficulty: float,
            executor: Any = None
    ) -> Tuple[bool, str]:
        """规划并执行任务，存储经验"""
        dag, status = self.plan_subtasks(task_id, task_type, main_task, difficulty)
        if not dag:
            self.experience_pool.add_experience(
                Experience(
                    task_id=task_id,
                    task_type=task_type,
                    prompt=main_task,
                    execution_time=0,
                    success=False,
                    result="",
                    timestamp=time.time(),
                    difficulty=difficulty,
                    failure_reason=status
                )
            )
            return False, status

        start_time = time.time()
        try:
            # 使用任务执行器执行任务
            result = self.task_executor.execute_task(task_id, task_type, main_task, dag)
            execution_time = time.time() - start_time
            success = True
            failure_reason = None
        except Exception as e:
            execution_time = time.time() - start_time
            success = False
            result = str(e)
            failure_reason = str(e)

        self.experience_pool.add_experience(
            Experience(
                task_id=task_id,
                task_type=task_type,
                prompt=main_task,
                execution_time=execution_time,
                success=success,
                result=result,
                timestamp=time.time(),
                difficulty=difficulty,
                subtask_dag=dag,
                failure_reason=failure_reason
            )
        )

        if not success:
            self.checker.update_from_experience(
                Experience(
                    task_id=task_id,
                    task_type=task_type,
                    prompt=main_task,
                    execution_time=execution_time,
                    success=success,
                    result=result,
                    timestamp=time.time(),
                    difficulty=difficulty,
                    subtask_dag=dag,
                    failure_reason=failure_reason
                )
            )

        return success, result if success else f"Execution failed: {failure_reason}"

    def set_task_executor(self, executor: Any):
        """设置任务执行器"""
        self.task_executor = executor


# 验证类结构
if __name__ == "__main__":
    print("=" * 50)
    print("SubtaskPlanner 类结构验证")
    print("=" * 50)

    # 检查必需方法
    required_methods = [
        '_initialize_domain_specific_steps',
        '_is_task_completed',
        '_ensure_valid_exit_points',
        '_can_skip_critical_issues',
        '_handle_dependency_issues',
        'plan_subtasks',
        'execute_and_store',
        'set_task_executor'
    ]

    missing_methods = []
    for method in required_methods:
        if not hasattr(SubtaskPlanner, method):
            missing_methods.append(method)

    if missing_methods:
        print(f"错误: SubtaskPlanner缺少以下必需方法: {', '.join(missing_methods)}")
    else:
        print("✓ 所有必需方法已定义")

    # 尝试创建实例
    try:
        from experience.experience_pool import ExperiencePool

        pool = ExperiencePool()
        planner = SubtaskPlanner(pool)
        print("✓ SubtaskPlanner实例创建成功")

        # 尝试调用关键方法
        dag = SubtaskDAG()
        planner._initialize_domain_specific_steps(dag, "data_analysis", "Test task")
        print("✓ _initialize_domain_specific_steps方法调用成功")
    except Exception as e:
        print(f"✗ 实例创建或方法调用失败: {str(e)}")

    print("=" * 50)