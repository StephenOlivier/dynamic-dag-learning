import time
import os
from collections import Counter

# 导入系统组件
from experience.experience_pool import ExperiencePool
from planning.subtask_planner import SubtaskPlanner
from execution.task_executor import TaskExecutor
from models.llm_manager import LLMManager, OPENAI_AVAILABLE
from task_generator import Task, generate_random_task


def main():
    print("=" * 60)
    print("智能任务规划与执行系统")
    print("=" * 60)

    # 初始化系统组件
    print("\n1. 初始化系统组件...")
    experience_pool = ExperiencePool(capacity=50)
    planner = SubtaskPlanner(experience_pool)
    llm_manager = LLMManager()
    task_executor = TaskExecutor(llm_manager)
    planner.set_task_executor(task_executor)

    # 检查当前使用的模型
    llm_type = "OpenAI API" if OPENAI_AVAILABLE else "本地模型（DeepSeek）"
    embedding_type = "SentenceTransformer" if llm_manager.embedding_manager.embedding_model else "简易词袋模型"

    print(f"  * LLM 服务: {llm_type}")
    print(f"  * 嵌入模型: {embedding_type}")
    print(f"  * 经验池容量: {experience_pool.capacity}")
    print(f"  * 最大规划尝试次数: {planner.max_prediction_attempts}")

    # 定义测试任务
    test_tasks = [
        {
            "task_id": 1001,
            "task_type": "data_analysis",
            "main_task": "Analyze customer churn data to identify key factors and provide recommendations",
            "difficulty": 4.2,
            "priority": 3.0
        },
        # ... 其他测试任务定义 ...
    ]

    # 执行任务
    for i, task in enumerate(test_tasks):
    # ... 任务执行逻辑 ...

    # 系统总结
    print("\n" + "=" * 60)
    print("系统执行完成!")
    print("=" * 60)
    print(f"最终经验池大小: {sum(len(exps) for exps in experience_pool.experiences.values())} 条经验")
    print(f"累计处理任务类型: {len(experience_pool.experiences)} 种")
    print("系统已准备好处理更多任务!")


if __name__ == "__main__":
    main()