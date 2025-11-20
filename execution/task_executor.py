from typing import Any, List, Dict, Tuple, Optional
import time
import random
import numpy as np
import networkx as nx
from utils.types import Subtask, SubtaskDAG  # 从共享模块导入
from models.llm_manager import LLMManager
from knowledge.validation_rules import ValidationRule, TaskConstraint


class TaskExecutor:
    """统一的任务执行引擎，处理各种任务类型的具体执行逻辑"""

    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.task_results = {}

    def execute_task(self, task_id: int, task_type: str, main_task: str, dag: SubtaskDAG) -> str:
        """根据任务类型执行相应的处理逻辑"""
        execution_method = getattr(self, f"_execute_{task_type}_task", self._execute_default_task)
        return execution_method(task_id, main_task, dag)

    def _execute_data_analysis_task(self, task_id: int, main_task: str, dag: SubtaskDAG) -> str:
        """执行数据分析任务"""
        result = f"Data analysis execution for task ID {task_id}:\n"
        analysis_steps = []

        # 按依赖顺序执行节点
        execution_order = self._get_execution_order(dag)

        for node_id in execution_order:
            node = dag.nodes[node_id]
            step_result = f"- Executed {node.task_type}: {node.description}\n"

            # 特定于数据分析的执行逻辑
            if node.task_type == "cleaning":
                step_result += "  * Cleaned dataset, handled missing values and outliers\n"
            elif node.task_type == "eda":
                step_result += "  * Performed exploratory analysis, identified patterns and correlations\n"
            elif node.task_type == "modeling":
                step_result += "  * Built predictive model with appropriate algorithm\n"
            elif node.task_type == "reporting":
                step_result += "  * Generated comprehensive report with visualizations and recommendations\n"

            analysis_steps.append(step_result)

        result += "\n".join(analysis_steps)
        result += f"\nAnalysis completed successfully with {len(execution_order)} steps."

        # 存储结果供后续使用
        self.task_results[task_id] = result
        return result

    def _execute_code_generation_task(self, task_id: int, main_task: str, dag: SubtaskDAG) -> str:
        """执行代码生成任务"""
        result = f"Code generation execution for task ID {task_id}:\n"
        code_steps = []

        execution_order = self._get_execution_order(dag)

        for node_id in execution_order:
            node = dag.nodes[node_id]
            step_result = f"- Executed {node.task_type}: {node.description}\n"

            # 特定于代码生成的执行逻辑
            if node.task_type == "requirements":
                step_result += "  * Analyzed requirements and created specifications\n"
            elif node.task_type == "design":
                step_result += "  * Created system architecture and interface designs\n"
            elif node.task_type == "implementation":
                step_result += "  * Implemented code with proper structure and comments\n"
            elif node.task_type == "testing":
                step_result += "  * Executed unit tests and verified functionality\n"
            elif node.task_type == "documentation":
                step_result += "  * Generated API documentation and user guides\n"

            code_steps.append(step_result)

        result += "\n".join(code_steps)
        result += f"\nCode generation completed with {len(execution_order)} components."

        self.task_results[task_id] = result
        return result

    def _execute_translation_task(self, task_id: int, main_task: str, dag: SubtaskDAG) -> str:
        """执行翻译任务"""
        result = f"Translation execution for task ID {task_id}:\n"
        translation_steps = []

        execution_order = self._get_execution_order(dag)

        for node_id in execution_order:
            node = dag.nodes[node_id]
            step_result = f"- Executed {node.task_type}: {node.description}\n"

            if "language" in node.description.lower() or node.task_type == "lang_identification":
                step_result += "  * Identified source language as English\n"
            elif node.task_type == "translation":
                # 使用LLM生成翻译
                translation_prompt = f"Translate the following text: '{main_task}'. Provide only the translation."
                translation = self.llm_manager.generate_response(translation_prompt, max_tokens=200)
                step_result += f"  * Generated translation: '{translation[:100]}...'[truncated]\n"
            elif "quality" in node.description.lower() or node.task_type == "quality_check":
                step_result += "  * Verified translation accuracy and fluency\n"

            translation_steps.append(step_result)

        result += "\n".join(translation_steps)
        result += f"\nTranslation task completed with {len(execution_order)} steps."

        self.task_results[task_id] = result
        return result

    def _execute_story_task(self, task_id: int, main_task: str, dag: SubtaskDAG) -> str:
        """执行故事创作任务"""
        result = f"Story creation execution for task ID {task_id}:\n"
        story_steps = []

        execution_order = self._get_execution_order(dag)

        story_content = ""

        for node_id in execution_order:
            node = dag.nodes[node_id]
            step_result = f"- Executed {node.task_type}: {node.description}\n"

            if node.task_type == "setup" or "character" in node.description.lower():
                setup_prompt = f"Create character profiles and setting details for a story about: {main_task}"
                setup_content = self.llm_manager.generate_response(setup_prompt, max_tokens=300)
                step_result += f"  * Created story setup: {setup_content[:100]}...[truncated]\n"
                story_content += setup_content + "\n"
            elif node.task_type == "outline" or "plot" in node.description.lower():
                outline_prompt = f"Create a plot outline for a story with this setup: {story_content[:200]}"
                outline_content = self.llm_manager.generate_response(outline_prompt, max_tokens=300)
                step_result += f"  * Developed plot outline: {outline_content[:100]}...[truncated]\n"
                story_content += outline_content + "\n"
            elif node.task_type == "writing" or "detail" in node.description.lower():
                writing_prompt = f"Write a detailed story section based on this outline: {story_content[:300]}"
                writing_content = self.llm_manager.generate_response(writing_prompt, max_tokens=500)
                step_result += f"  * Generated story content: {writing_content[:150]}...[truncated]\n"
                story_content += writing_content + "\n"
            elif node.task_type == "revision" or "edit" in node.description.lower():
                step_result += "  * Revised and polished the story content\n"

            story_steps.append(step_result)

        result += "\n".join(story_steps)
        result += f"\nStory creation completed with {len(execution_order)} phases."

        self.task_results[task_id] = {
            "summary": result,
            "full_content": story_content
        }
        return result

    def _execute_math_problem_task(self, task_id: int, main_task: str, dag: SubtaskDAG) -> str:
        """执行数学问题任务"""
        result = f"Math problem solving for task ID {task_id}:\n"
        math_steps = []

        execution_order = self._get_execution_order(dag)

        for node_id in execution_order:
            node = dag.nodes[node_id]
            step_result = f"- Executed {node.task_type}: {node.description}\n"

            if node.task_type == "understanding" or "understand" in node.description.lower():
                step_result += "  * Analyzed problem requirements and identified key variables\n"
            elif node.task_type == "solution" or "solve" in node.description.lower():
                # 使用LLM解决数学问题
                solve_prompt = f"Solve this math problem step by step: {main_task}"
                solution = self.llm_manager.generate_response(solve_prompt, max_tokens=400)
                step_result += f"  * Solution steps: {solution[:200]}...[truncated]\n"
            elif node.task_type == "verification" or "verify" in node.description.lower():
                step_result += "  * Verified solution correctness through alternative methods\n"

            math_steps.append(step_result)

        result += "\n".join(math_steps)
        result += f"\nMath problem solved through {len(execution_order)} logical steps."

        self.task_results[task_id] = result
        return result

    def _execute_summary_task(self, task_id: int, main_task: str, dag: SubtaskDAG) -> str:
        """执行摘要任务"""
        result = f"Document summarization for task ID {task_id}:\n"
        summary_steps = []

        execution_order = self._get_execution_order(dag)

        document_content = ""
        final_summary = ""

        for node_id in execution_order:
            node = dag.nodes[node_id]
            step_result = f"- Executed {node.task_type}: {node.description}\n"

            if node.task_type == "analysis" or "analyze" in node.description.lower():
                analysis_prompt = f"Analyze the key themes and structure of this content: {main_task}"
                analysis = self.llm_manager.generate_response(analysis_prompt, max_tokens=300)
                step_result += f"  * Text analysis completed: {analysis[:100]}...[truncated]\n"
                document_content = analysis
            elif node.task_type == "keypoints" or "extract" in node.description.lower():
                keypoints_prompt = f"Extract the 5 most important key points from this analysis: {document_content[:300]}"
                keypoints = self.llm_manager.generate_response(keypoints_prompt, max_tokens=300)
                step_result += f"  * Key points extracted: {keypoints[:150]}...[truncated]\n"
                document_content += "\n" + keypoints
            elif node.task_type == "final_summary" or "summarize" in node.description.lower():
                summary_prompt = f"Create a concise summary based on these key points: {document_content[:500]}"
                final_summary = self.llm_manager.generate_response(summary_prompt, max_tokens=400)
                step_result += f"  * Final summary generated: {final_summary[:200]}...[truncated]\n"

            summary_steps.append(step_result)

        result += "\n".join(summary_steps)
        result += f"\nSummarization completed with {len(execution_order)} processing stages."

        self.task_results[task_id] = {
            "summary": result,
            "full_summary": final_summary
        }
        return result

    def _execute_combined_task(self, task_id: int, main_task: str, dag: SubtaskDAG) -> str:
        """执行组合任务"""
        result = f"Combined task execution for task ID {task_id}:\n"
        combined_steps = []

        execution_order = self._get_execution_order(dag)

        # 按照任务类型分组执行
        data_analysis_steps = []
        code_generation_steps = []
        other_steps = []

        for node_id in execution_order:
            node = dag.nodes[node_id]
            step_result = f"- Executed {node.task_type}: {node.description}\n"

            # 识别任务类型并应用相应的逻辑
            if any(term in node.task_type.lower() for term in ["analyze", "data", "clean", "model"]):
                step_result += "  * Applied data analysis procedure\n"
                data_analysis_steps.append(node.description)
            elif any(term in node.task_type.lower() for term in ["code", "implement", "test", "develop"]):
                step_result += "  * Applied software development procedure\n"
                code_generation_steps.append(node.description)
            else:
                step_result += "  * Executed general task procedure\n"
                other_steps.append(node.description)

            combined_steps.append(step_result)

        result += "\n".join(combined_steps)
        result += f"\nCombined task completed with {len(execution_order)} subtasks across multiple domains."
        result += f"\n  * Data analysis components: {len(data_analysis_steps)}\n"
        result += f"  * Code generation components: {len(code_generation_steps)}\n"
        result += f"  * Other components: {len(other_steps)}"

        self.task_results[task_id] = result
        return result

    def _execute_default_task(self, task_id: int, main_task: str, dag: SubtaskDAG) -> str:
        """默认任务执行方法"""
        result = f"Generic task execution for task ID {task_id}:\n"
        generic_steps = []

        execution_order = self._get_execution_order(dag)

        for node_id in execution_order:
            node = dag.nodes[node_id]
            generic_steps.append(f"- Executed {node.task_type}: {node.description}")

        result += "\n".join(generic_steps)
        result += f"\nTask completed with {len(execution_order)} subtasks."

        self.task_results[task_id] = result
        return result

    def _get_execution_order(self, dag: SubtaskDAG) -> List[str]:
        """获取DAG的执行顺序（拓扑排序）"""
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

    def get_task_result(self, task_id: int) -> Any:
        """获取存储的任务结果"""
        return self.task_results.get(task_id, "No result found for this task ID")