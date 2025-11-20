"""
经验管理模块

该模块负责系统中历史任务经验的存储、检索和管理，支持基于经验的学习和决策。
主要功能包括：
- 经验的存储和检索
- 相似任务识别
- 基于历史经验的规则置信度更新
- 智能经验淘汰策略
- 与领域知识库的集成

经验结构包含完整的任务信息、执行结果、子任务DAG、应用的规则和约束等元数据。
"""

from experience_pool import ExperiencePool, Experience, RuleEvidence, ConstraintEvidence

__all__ = [
    'ExperiencePool',
    'Experience',
    'RuleEvidence',
    'ConstraintEvidence'
]