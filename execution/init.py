"""
任务执行模块

该模块负责处理各种任务类型的具体执行逻辑，将DAG结构转换为实际执行结果。
支持多种任务类型，包括数据分析、代码生成、翻译、数学问题、故事创作、摘要和组合任务。
"""

from .task_executor import TaskExecutor

__all__ = [
    'TaskExecutor'
]