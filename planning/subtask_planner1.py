from typing import Tuple, Optional, Any, List
import time
from core.types import Subtask, SubtaskDAG, Experience
from experience.experience_pool import ExperiencePool
from knowledge.validation_rules import RuleBasedChecker
from knowledge.domain_knowledge import DomainKnowledgeBase
from planning.subtask_predictor import SubtaskPredictor


class SubtaskPlanner:
    def __init__(self, experience_pool: ExperiencePool):
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

    def set_task_executor(self, executor: Any):
        self.task_executor = executor

    # def plan_subtasks(
    #         self,
    #         task_id: int,
    #         task_type: str,
    #         main_task: str,
    #         difficulty: float
    # ) -> Tuple[Optional[SubtaskDAG], str]:
    #     dag = SubtaskDAG(nodes={}, entry_points=[], exit_points=[])
    #
    #     # 修复3: 对于数据分析任务，确保首先添加数据清洗步骤
    #     if task_type == "data_analysis" and not dag.nodes:
    #         cleaning_task = Subtask(
    #             id=self.predictor._generate_subtask_id(),
    #             description="Clean and preprocess customer churn data",
    #             task_type="cleaning",
    #             dependencies=[],
    #             required_resources={"tools": ["pandas", "numpy"], "data": ["raw_data.csv"]},
    #             expected_output="Cleaned dataset with no missing values or outliers",
    #             difficulty=3.5,
    #             estimated_time=120.0
    #         )
    #         dag.add_subtask(cleaning_task)
    #         print("  ✅ 自动添加了数据清洗步骤作为入口点")
    #
    #     all_issues = []
    #     task_counter = 0
    #
    #     while True:
    #         task_counter += 1
    #         if task_counter > 10:
    #             return None, "Exceeded maximum subtask count (10)"
    #
    #         subtask = self.predictor.predict_next_subtask(main_task, dag, task_type)
    #         if not subtask:
    #             break
    #
    #         is_valid, issues, _ = self.checker.validate_subtask(subtask, main_task, task_type, dag)
    #
    #         # 修复4: 处理无效依赖问题
    #         invalid_dep_issues = [i for i in issues if "does not exist" in i["message"].lower()]
    #         if invalid_dep_issues:
    #             print("  ⚠️ 检测到无效依赖，尝试修复...")
    #
    #             # 尝试找到最可能的替代依赖
    #             valid_deps = []
    #             for issue in invalid_dep_issues:
    #                 # 从错误消息中提取缺失的依赖ID
    #                 import re
    #                 match = re.search(r"Dependency '(.+?)' does not exist", issue["message"])
    #                 if match:
    #                     missing_dep = match.group(1)
    #                     # 尝试匹配相似ID
    #                     for node_id in dag.nodes.keys():
    #                         if missing_dep in node_id or node_id in missing_dep:
    #                             valid_deps.append(node_id)
    #                             break
    #
    #             # 如果找到替代依赖，使用它们
    #             if valid_deps:
    #                 subtask.dependencies = valid_deps
    #                 print(f"  ✅ 通过替换为有效依赖修复了问题: {valid_deps}")
    #                 is_valid, issues, _ = self.checker.validate_subtask(subtask, main_task, task_type, dag)
    #             elif dag.nodes:
    #                 # 如果没有找到替代依赖，使用最后一个节点
    #                 last_node_id = list(dag.nodes.keys())[-1]
    #                 subtask.dependencies = [last_node_id]
    #                 print(f"  ✅ 通过设置最后节点为依赖修复了问题: {last_node_id}")
    #                 is_valid, issues, _ = self.checker.validate_subtask(subtask, main_task, task_type, dag)
    #
    #         # 修复5: 处理数据分析规则检查失败
    #         da_rule_issues = [i for i in issues if "data analysis requires prior data cleaning" in i["message"].lower()]
    #         if da_rule_issues and subtask.task_type in ["analysis", "modeling", "reporting"]:
    #             print("  ⚠️ 检测到数据分析规则检查失败，尝试添加数据清洗步骤...")
    #
    #             # 检查是否已经有清洗步骤
    #             has_cleaning = any(
    #                 "clean" in node.task_type.lower() for node in dag.nodes.values()
    #             )
    #
    #             if not has_cleaning:
    #                 # 添加数据清洗步骤
    #                 cleaning_task = Subtask(
    #                     id=self.predictor._generate_subtask_id(),
    #                     description="Clean and preprocess data",
    #                     task_type="cleaning",
    #                     dependencies=[],
    #                     required_resources={"tools": ["pandas", "numpy"]},
    #                     expected_output="Cleaned dataset ready for analysis",
    #                     difficulty=3.0,
    #                     estimated_time=90.0
    #                 )
    #                 dag.add_subtask(cleaning_task)
    #                 print("  ✅ 添加了缺失的数据清洗步骤")
    #
    #                 # 重新验证
    #                 is_valid, issues, _ = self.checker.validate_subtask(subtask, main_task, task_type, dag)
    #
    #         if is_valid:
    #             dag.add_subtask(subtask)
    #             continue
    #
    #         all_issues.extend(issues)
    #
    #         if task_counter >= self.max_prediction_attempts:
    #             critical_issues = [issue for issue in issues if issue["severity"] in ["critical", "high"]]
    #             if not critical_issues:
    #                 dag.add_subtask(subtask)
    #                 break
    #             return None, f"Failed to create valid DAG after {self.max_prediction_attempts} attempts. Issues: " + \
    #                          ", ".join([f"{issue['severity']}:{issue['message']}" for issue in issues])
    #
    #     is_dag_valid, dag_message, _ = dag.validate_dag()
    #     if not is_dag_valid:
    #         return None, f"DAG validation failed: {dag_message}"
    #
    #     return dag, "DAG planning completed successfully"

    def plan_subtasks(
            self,
            task_id: int,
            task_type: str,
            main_task: str,
            difficulty: float
    ) -> Tuple[Optional[SubtaskDAG], str]:
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