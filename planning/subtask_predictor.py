import time
import json
import re
import time
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING
import random
import numpy as np
from utils.types import Subtask, SubtaskDAG  # 从共享模块导入
from experience.experience_pool import ExperiencePool, Experience
from models.llm_manager import LLMManager
from knowledge.validation_rules import ValidationRule, TaskConstraint

if TYPE_CHECKING:
    from experience.experience_pool import ExperiencePool, Experience
    from utils.dag_utils import SubtaskDAG, Subtask
    from models.llm_manager import LLMManager

class SubtaskPredictor:
    """支持OpenAI和本地模型的子任务预测器"""

    def __init__(self, experience_pool: "ExperiencePool"):
        self.experience_pool = experience_pool
        self.llm_manager = LLMManager()
        self.task_counter = 0

    def _generate_subtask_id(self) -> str:
        self.task_counter += 1
        return f"subtask_{int(time.time())}_{self.task_counter}"

    def _create_context(self, main_task: str, current_dag: SubtaskDAG) -> str:
        """创建任务上下文，用于指导LLM生成下一个子任务"""
        if not current_dag.nodes:
            return "This is the beginning of the task."

        # 收集已完成的子任务信息
        context_lines = ["Current task progress:"]

        # 按执行顺序获取节点
        execution_order = current_dag.get_execution_order()
        for node_id in execution_order:
            node = current_dag.nodes[node_id]
            context_lines.append(f"- {node.description} [{node.task_type}]")

        # 添加任务完成状态
        completed_steps = len(current_dag.nodes)
        total_estimated_steps = max(5, int(completed_steps * 1.5))  # 估计总步数
        progress = min(100, int((completed_steps / total_estimated_steps) * 100))
        context_lines.append(f"\nProgress: {completed_steps}/{total_estimated_steps} steps completed ({progress}%)")

        # 添加领域特定提示
        if "analyze" in main_task.lower() or "data" in main_task.lower():
            context_lines.append("\nData analysis specific context: Focus on statistical methods and visualization")
        elif "code" in main_task.lower() or "implement" in main_task.lower():
            context_lines.append("\nCode generation specific context: Focus on design patterns and testing")
        elif "translate" in main_task.lower():
            context_lines.append("\nTranslation specific context: Focus on language accuracy and cultural adaptation")

        return "\n".join(context_lines)

    def predict_next_subtask(
            self,
            main_task: str,
            current_dag: SubtaskDAG,
            task_type: str
    ) -> Optional[Subtask]:
        """预测下一个子任务 - 支持多依赖关系"""
        # 创建上下文
        context = self._create_context(main_task, current_dag)

        # 生成子任务
        subtask = self._generate_subtask_with_llm(
            main_task, task_type, not current_dag.nodes, current_dag, context
        )

        if subtask is None:
            return None

        # 增强子任务 - 支持多依赖
        subtask = self._enhance_subtask_dependencies(subtask, current_dag, task_type)

        return subtask

    def _enhance_subtask_dependencies(
            self,
            subtask: Subtask,
            current_dag: SubtaskDAG,
            task_type: str
    ) -> Subtask:
        """增强子任务的依赖关系 - 支持多依赖"""
        # 如果LLM生成的依赖列表为空或只包含最后一个任务，尝试添加更多依赖
        last_task_id = list(current_dag.nodes.keys())[-1] if current_dag.nodes else None
        if not subtask.dependencies or (len(subtask.dependencies) == 1 and 
                                        subtask.dependencies[0] == last_task_id):

            # 根据任务类型确定可能的依赖
            possible_dependencies = []

            if task_type == "code_generation":
                # 实现任务可以依赖设计和环境准备
                if subtask.task_type == "implementation":
                    for node_id, node in current_dag.nodes.items():
                        if node.task_type in ["design", "environment_setup", "requirements"]:
                            possible_dependencies.append(node_id)

                # 测试任务可以依赖实现和文档
                elif subtask.task_type == "testing":
                    for node_id, node in current_dag.nodes.items():
                        if node.task_type in ["implementation", "documentation"]:
                            possible_dependencies.append(node_id)

            elif task_type == "data_analysis":
                # 建模任务可以依赖EDA和数据准备
                if subtask.task_type == "modeling":
                    for node_id, node in current_dag.nodes.items():
                        if node.task_type in ["eda", "data_preparation", "feature_engineering"]:
                            possible_dependencies.append(node_id)

                # 可视化任务可以依赖EDA和建模
                elif subtask.task_type == "visualization":
                    for node_id, node in current_dag.nodes.items():
                        if node.task_type in ["eda", "modeling", "data_preparation"]:
                            possible_dependencies.append(node_id)

                # 报告任务可以依赖建模、可视化和EDA
                elif subtask.task_type == "reporting":
                    for node_id, node in current_dag.nodes.items():
                        if node.task_type in ["modeling", "visualization", "eda"]:
                            possible_dependencies.append(node_id)

            elif task_type == "combined_task":
                # 对于组合任务，可以有更多并行路径
                for node_id, node in current_dag.nodes.items():
                    # 不同类型的任务可以依赖不同类型的任务
                    if subtask.task_type == "integration":
                        if node.task_type in ["analysis", "modeling", "code_generation", "data_preparation"]:
                            possible_dependencies.append(node_id)
                    elif subtask.task_type in ["analysis", "modeling"]:
                        if node.task_type in ["data_preparation", "cleaning", "eda"]:
                            possible_dependencies.append(node_id)

            # 确保至少有一个依赖，但可以有多个
            if possible_dependencies:
                # 选择1到所有可能的依赖，以增加并行性
                num_deps = random.randint(1, min(len(possible_dependencies), 3))  # 最多3个依赖
                selected_deps = random.sample(possible_dependencies, num_deps)
                subtask.dependencies = selected_deps
                print(f"  ✅ 增强依赖关系: 添加 {len(selected_deps)} 个额外依赖")
            else:
                # 如果没有找到特定依赖，至少依赖最后一个任务（除非是第一个任务）
                if current_dag.nodes and last_task_id:
                    subtask.dependencies = [last_task_id]

        return subtask

    def _predict_initial_subtask(
            self,
            main_task: str,
            task_type: str
    ) -> "Subtask":
        from experience.experience_pool import Experience
        from utils.dag_utils import SubtaskDAG, Subtask

        relevant_exps = self.experience_pool.get_relevant_experiences(
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
            top_k=2
        )
        if relevant_exps and relevant_exps[0].subtask_dag:
            first_node_id = relevant_exps[0].subtask_dag.entry_points[0]
            first_subtask = relevant_exps[0].subtask_dag.nodes[first_node_id]
            new_id = self._generate_subtask_id()
            new_desc = self._adapt_description(
                first_subtask.description,
                main_task
            )
            return Subtask(
                id=new_id,
                description=new_desc,
                task_type=first_subtask.task_type,
                dependencies=[],
                required_resources=first_subtask.required_resources.copy(),
                expected_output=first_subtask.expected_output,
                difficulty=first_subtask.difficulty,
                estimated_time=first_subtask.estimated_time
            )
        # Fall back to LLM generation
        return self._generate_subtask_with_llm(
            main_task,
            task_type,
            is_first=True,
            current_dag=SubtaskDAG(nodes={}, entry_points=[], exit_points=[])
        )

    def _predict_followup_subtask(
            self,
            main_task: str,
            current_dag: "SubtaskDAG",
            task_type: str
    ) -> Optional["Subtask"]:
        candidate_nodes = self._find_candidate_nodes_for_next_subtask(current_dag)
        if not candidate_nodes:
            return None

        relevant_exps = self.experience_pool.get_relevant_experiences(
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
            top_k=2
        )

        if relevant_exps and relevant_exps[0].subtask_dag:
            next_subtask = self._analyze_history_for_next_subtask(
                relevant_exps,
                current_dag,
                candidate_nodes
            )
            if next_subtask:
                return next_subtask

        return self._generate_subtask_with_llm(
            main_task,
            task_type,
            is_first=False,
            current_dag=current_dag,
            current_context=self._describe_current_state(main_task, current_dag)
        )

    def _find_candidate_nodes_for_next_subtask(
            self,
            current_dag: "SubtaskDAG"
    ) -> List[str]:
        return current_dag.exit_points.copy()

    def _analyze_history_for_next_subtask(
            self,
            relevant_exps: List["Experience"],
            current_dag: "SubtaskDAG",
            candidate_nodes: List[str]
    ) -> Optional["Subtask"]:
        for exp in relevant_exps:
            if not exp.subtask_dag:
                continue
            current_nodes = set(current_dag.nodes.keys())
            history_nodes = set(exp.subtask_dag.nodes.keys())
            if current_nodes.issubset(history_nodes):
                for node_id, node in exp.subtask_dag.nodes.items():
                    if node_id not in current_nodes:
                        valid_dependency = False
                        for dep in node.dependencies:
                            if dep in current_nodes:
                                valid_dependency = True
                                break
                        if valid_dependency or not node.dependencies:
                            new_id = self._generate_subtask_id()
                            new_desc = self._adapt_description(
                                node.description,
                                list(current_dag.nodes.values())[-1].description
                            )
                            return Subtask(
                                id=new_id,
                                description=new_desc,
                                task_type=node.task_type,
                                dependencies=node.dependencies,
                                required_resources=node.required_resources.copy(),
                                expected_output=node.expected_output,
                                difficulty=node.difficulty,
                                estimated_time=node.estimated_time
                            )
        return None

    def _adapt_description(self, original_desc: str, context: str) -> str:
        """改进描述适配，避免重复"""
        # 确保描述有变化
        if "data" in context.lower() and "analyze" not in original_desc.lower():
            return f"Analyze the {original_desc.lower()}"

        # 添加随机修饰词，避免完全重复
        modifiers = ["further", "detailed", "comprehensive", "in-depth", "advanced", "comparative", "statistical"]

        # 检查是否已有修饰词
        has_modifier = any(mod in original_desc.lower() for mod in modifiers)

        # 随机决定是否添加修饰词（60%概率）
        if not has_modifier and random.random() < 0.6:
            modifier = random.choice(modifiers)
            # 避免重复添加相同修饰词
            if modifier not in original_desc.lower():
                return f"{modifier.capitalize()} {original_desc}"

        # 确保描述有足够变化
        if random.random() < 0.3:  # 30%概率添加额外描述
            additions = [
                "with a focus on key metrics",
                "including data visualization",
                "using appropriate statistical methods",
                "with detailed explanations",
                "considering business implications"
            ]
            addition = random.choice(additions)
            return f"{original_desc} {addition}"

        return original_desc

    def _describe_current_state(self, main_task: str, current_dag: "SubtaskDAG") -> str:
        node_descriptions = [f"- {node.description}" for node in current_dag.nodes.values()]
        return f"Main task: {main_task}\nCurrent progress:\n" + "\n".join(node_descriptions)

    def _generate_subtask_with_llm(
            self,
            main_task: str,
            task_type: str,
            is_first: bool,
            current_dag: "SubtaskDAG",
            current_context: str = ""
    ) -> "Subtask":
        """使用本地LLM生成子任务 - 修复依赖ID问题"""
        from core.types import Subtask
        domain_guidance = ""  # 确保始终初始化为空字符串

        # 领域特定指导
        if task_type == "code_generation":
            domain_guidance = """For code generation tasks, use these specific task types:
        - requirements: Analyze requirements and specifications
        - design: Design system architecture and components
        - implementation: Implement core functionality
        - testing: Write and execute unit tests
        - documentation: Create documentation and examples
        - deployment: Prepare for deployment

        Each subtask description should be specific and actionable, NOT generic like 'Next step'.
        For example: 'Implement Fibonacci function with memoization in Python' instead of 'Next step for...'

        The task sequence should typically be: requirements → design → implementation → testing → documentation"""

        elif task_type == "translation":
            domain_guidance = """For translation tasks, use these specific task types:
        - lang_identification: Identify source and target languages
        - translation: Translate content from source to target language
        - quality_check: Verify translation quality and accuracy
        - localization: Adapt content for cultural context

        Each subtask description should be specific and actionable, NOT generic like 'Next step'.
        For example: 'Translate technical terms related to AI ethics' instead of 'Next step for...'

        The task sequence should typically be: lang_identification → translation → quality_check → localization"""

        elif task_type == "data_analysis":
            domain_guidance = """For data analysis tasks, use these specific task types:
        - cleaning: Clean and preprocess raw data
        - eda: Perform exploratory data analysis
        - modeling: Build and validate predictive models
        - reporting: Create reports with key insights

        Each subtask description should be specific and actionable, NOT generic like 'Next step'.
        For example: 'Analyze correlations between customer demographics and churn rate' instead of 'Next step for...'

        The task sequence should typically be: cleaning → eda → modeling → reporting

        CRITICAL: Data analysis tasks often have parallel paths:
        * cleaning → eda → modeling
        * cleaning → feature_engineering → modeling
        * eda & feature_engineering → modeling"""

        elif task_type == "math_problem":
            domain_guidance = """For math problems, use these specific task types:
        - understanding: Understand and analyze the problem
        - solution: Solve the mathematical problem
        - verification: Verify the solution

        Each subtask description should be specific and actionable.
        For example: 'Apply quadratic formula to solve the equation' instead of 'Next step for...'

        The task sequence should be: understanding → solution → verification"""

        elif task_type == "story":
            domain_guidance = """For story creation, use these specific task types:
        - setup: Define story setting and characters
        - outline: Create plot outline and key events
        - detailed_writing: Write detailed story content
        - revision: Revise and edit the story

        Each subtask description should be specific and actionable.
        For example: 'Develop robot character's personality and background' instead of 'Next step for...'

        The task sequence should be: setup → outline → detailed_writing → revision"""

        elif task_type == "summary":
            domain_guidance = """For text summarization, use these specific task types:
        - analysis: Analyze the source document
        - keypoints: Extract key points and themes
        - final_summary: Create the final summary

        Each subtask description should be specific and actionable.
        For example: 'Identify main arguments about climate change impacts' instead of 'Next step for...'

        The task sequence should be: analysis → keypoints → final_summary"""

        elif task_type == "combined_task":
            domain_guidance = """For combined tasks, identify and handle multiple components:
        - For data analysis + code generation: 
          * data_analysis: cleaning → eda → modeling
          * code_generation: requirements → implementation → testing
          * integration: Combine analysis results with code

        Each subtask description should be specific and actionable.
        For example: 'Analyze survey data to identify key user preferences' instead of 'Next step for...'

        CRITICAL: Combined tasks often have parallel execution paths that should be identified"""

        # 确保domain_guidance始终是一个字符串
        if not isinstance(domain_guidance, str):
            domain_guidance = ""

        prompt = f"""You are an expert task planner. Generate the {'first' if is_first else 'next'} subtask for the main task.

Main task: {main_task}

{domain_guidance}

{'Current progress:' + chr(10) + current_context if not is_first else 'This is the beginning of the task.'}

Provide the subtask in JSON format with these fields:
- description: Clear, specific description of the subtask
- task_type: Type of this subtask (e.g., 'cleaning', 'analysis', 'modeling', 'exploration', 'visualization')
- dependencies: List of SUBTASK IDs this depends on (empty list for first subtask)
  IMPORTANT: A subtask can depend on MULTIPLE previous tasks, not just the last one
- required_resources: Dictionary of required resources
- expected_output: Description of expected output
- difficulty: Difficulty level from 1 (easy) to 5 (hard)
- estimated_time: Estimated time in seconds

CRITICAL INSTRUCTIONS:
1. SUBTASKS CAN HAVE MULTIPLE DEPENDENCIES - DO NOT LIMIT TO JUST THE LAST TASK
2. IDENTIFY OPPORTUNITIES FOR PARALLEL EXECUTION WHERE POSSIBLE
3. FOR EXAMPLE: A 'modeling' task might depend on BOTH 'eda' AND 'data_preparation'
4. FOR DATA ANALYSIS TASKS: Consider creating parallel paths like:
   - cleaning → eda → modeling
   - cleaning → feature_engineering → modeling  
   - eda & feature_engineering → modeling
5. USE EXACT SUBTASK IDs FROM CURRENT PROGRESS FOR DEPENDENCIES
6. IF UNSURE, PROVIDE A SENSIBLE DEFAULT, BUT ALWAYS ALLOW FOR MULTIPLE DEPENDENCIES

Respond ONLY with the JSON, no other text."""

        try:
            response = self.llm_manager.generate_response(
                prompt,
                max_tokens=512,
                temperature=0.3
            )

            # 尝试提取JSON块
            json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', response, re.DOTALL)
            if not json_match:
                json_match = re.search(r'({.*?})', response, re.DOTALL)

            if json_match:
                json_str = json_match.group(1)
                try:
                    data = json.loads(json_str)

                    # 修复1: 验证并修正依赖ID
                    fixed_dependencies = []
                    if "dependencies" in data and isinstance(data["dependencies"], list):
                        for dep in data["dependencies"]:
                            # 如果依赖ID是简化的（如"subtask_1"），尝试匹配完整ID
                            if dep.startswith("subtask_") and len(dep) < 15:
                                for node_id in current_dag.nodes.keys():
                                    if node_id.startswith(dep):
                                        fixed_dependencies.append(node_id)
                                        break
                            # 如果依赖是描述性文本，尝试匹配最相关的节点
                            elif isinstance(dep, str) and not dep.startswith("subtask_"):
                                # 简单实现：使用最后一个节点
                                if current_dag.nodes:
                                    fixed_dependencies = [list(current_dag.nodes.keys())[-1]]
                                    break
                            else:
                                fixed_dependencies.append(dep)

                    # 修复2: 确保依赖是列表
                    data["dependencies"] = fixed_dependencies if fixed_dependencies else []
                    return Subtask(
                        id=self._generate_subtask_id(),
                        description=data['description'],
                        task_type=data['task_type'],
                        dependencies=data['dependencies'],
                        required_resources=data['required_resources'],
                        expected_output=data['expected_output'],
                        difficulty=float(data['difficulty']),
                        estimated_time=float(data['estimated_time'])
                    )
                except json.JSONDecodeError:
                    pass

            # 如果JSON解析失败，使用默认值
            description = f"{'Initial' if is_first else 'Next'} step for {main_task[:30]}..."
            extracted_task_type = task_type if is_first else "followup"
            dependencies = []
            if not is_first and current_dag.nodes:
                dependencies = [list(current_dag.nodes.keys())[-1]]
            resources = {"tools": ["basic"]}
            expected_output = "Intermediate result"
            difficulty = 3.0
            estimated_time = 60.0

            return Subtask(
                id=self._generate_subtask_id(),
                description=description,
                task_type=extracted_task_type,
                dependencies=dependencies,
                required_resources=resources,
                expected_output=expected_output,
                difficulty=difficulty,
                estimated_time=estimated_time
            )
        except Exception as e:
            print(f"LLM generation error: {e}")
            # 降级方案
            base_desc = "Initial task analysis" if is_first else "Continue task execution"
            dependencies = []
            if not is_first and current_dag and current_dag.nodes:
                try:
                    dependencies = [list(current_dag.nodes.keys())[-1]]
                except:
                    dependencies = []
            return Subtask(
                id=self._generate_subtask_id(),
                description=base_desc,
                task_type=task_type,
                dependencies=dependencies,
                required_resources={"tools": ["fallback"]},
                expected_output="Basic output",
                difficulty=2.5,
                estimated_time=30.0
            )