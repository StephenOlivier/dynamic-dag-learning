#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能任务规划与执行系统 - 完整演示
演示系统的主要功能和工作流程
"""

import time
import json
from typing import List, Dict, Any
from core.types import Subtask, SubtaskDAG
from models.llm_manager import LLMManager
from experience.experience_pool import ExperiencePool
from planning.subtask_planner import SubtaskPlanner
from execution.task_executor import TaskExecutor

def demonstrate_system_overview():
    """演示系统概览"""
    print("=" * 60)
    print("智能任务规划与执行系统 - 完整演示")
    print("=" * 60)
    print("\n系统架构:")
    print("1. LLM管理器 - 负责大语言模型接口")
    print("2. 经验池 - 存储和检索任务执行经验")
    print("3. 任务规划器 - 生成子任务DAG")
    print("4. 任务执行器 - 执行具体任务")
    print("5. 核心类型 - 定义数据结构")

def demonstrate_task_planning():
    """演示任务规划过程"""
    print("\n" + "=" * 60)
    print("任务规划演示")
    print("=" * 60)
    
    # 初始化组件
    print("1. 初始化系统组件...")
    experience_pool = ExperiencePool()
    planner = SubtaskPlanner(experience_pool)
    llm_manager = LLMManager()
    
    # 设置任务执行器
    executor = TaskExecutor(llm_manager)
    planner.set_task_executor(executor)
    
    print("2. 定义测试任务...")
    main_task = "Analyze customer churn data to identify key factors and provide recommendations"
    task_type = "data_analysis"
    difficulty = 4.2
    
    print(f"   任务: {main_task}")
    print(f"   类型: {task_type}")
    print(f"   难度: {difficulty}/5.0")
    
    print("\n3. 开始规划子任务DAG...")
    dag, status = planner.plan_subtasks(1, task_type, main_task, difficulty)
    
    if dag:
        print(f"   ✅ 规划成功! 生成了 {len(dag.nodes)} 个子任务")
        print(f"   状态: {status}")
        
        print("\n4. 生成的子任务列表:")
        for i, (node_id, node) in enumerate(dag.nodes.items(), 1):
            print(f"   {i}. [{node.task_type}] {node.description}")
            if node.dependencies:
                dep_str = ", ".join(node.dependencies)
                print(f"      依赖于: {dep_str}")
        
        print(f"\n5. DAG结构分析:")
        print(f"   入口点: {dag.entry_points}")
        print(f"   出口点: {dag.exit_points}")
        
        # 验证DAG
        is_valid, msg, issues = dag.validate_dag()
        print(f"   DAG验证: {'✅ 有效' if is_valid else '❌ 无效'}")
        if issues:
            print(f"   发现问题: {len(issues)} 个")
            for issue in issues:
                print(f"     - {issue['severity']}: {issue['message']}")
        
        return dag
    else:
        print(f"   ❌ 规划失败: {status}")
        return None

def demonstrate_task_execution(dag: SubtaskDAG):
    """演示任务执行过程"""
    print("\n" + "=" * 60)
    print("任务执行演示")
    print("=" * 60)
    
    # 初始化执行器
    llm_manager = LLMManager()
    executor = TaskExecutor(llm_manager)
    
    main_task = "Analyze customer churn data to identify key factors and provide recommendations"
    task_id = 1
    task_type = "data_analysis"
    
    print("1. 开始执行任务...")
    print(f"   任务ID: {task_id}")
    print(f"   任务类型: {task_type}")
    print(f"   任务描述: {main_task}")
    
    try:
        result = executor.execute_task(task_id, task_type, main_task, dag)
        print("\n2. 执行结果:")
        print(result)
        
        print("\n3. 执行完成!")
        return True
    except Exception as e:
        print(f"\n3. 执行失败: {str(e)}")
        return False

def demonstrate_experience_learning():
    """演示经验学习过程"""
    print("\n" + "=" * 60)
    print("经验学习演示")
    print("=" * 60)
    
    print("1. 初始化经验池...")
    experience_pool = ExperiencePool()
    
    print("2. 添加模拟经验...")
    from experience.experience_pool import Experience
    
    # 添加一些模拟经验
    for i in range(3):
        exp = Experience(
            task_id=i + 1,
            task_type="data_analysis" if i % 2 == 0 else "code_generation",
            prompt=f"Sample task {i + 1}",
            execution_time=120.5,
            success=True,
            result=f"Completed task {i + 1} successfully",
            timestamp=time.time() - i * 3600,  # 模拟不同时间戳
            difficulty=3.0 + (i * 0.5),
            applied_rules=[f"rule_{j}" for j in range(i + 1)],
            violated_rules=[],
            satisfied_constraints=[f"constraint_{k}" for k in range(i + 1)],
            broken_constraints=[]
        )
        experience_pool.add_experience(exp)
        print(f"   添加经验: 任务 {i + 1}")
    
    print(f"\n3. 当前经验池状态:")
    print(f"   经验数量: {len(experience_pool.experiences)}")
    print(f"   容量限制: {experience_pool.capacity}")
    
    print(f"\n4. 检索相关经验...")
    query_exp = Experience(
        task_id=0,
        task_type="data_analysis",
        prompt="Analyze customer data",
        execution_time=0,
        success=True,
        result="",
        timestamp=time.time(),
        difficulty=3.5
    )
    
    relevant_exps = experience_pool.get_relevant_experiences(query_exp, top_k=2)
    print(f"   找到 {len(relevant_exps)} 个相关经验:")
    for i, exp in enumerate(relevant_exps, 1):
        print(f"   {i}. 任务ID: {exp.task_id}, 类型: {exp.task_type}, 难度: {exp.difficulty}")

def demonstrate_dag_features():
    """演示DAG特性"""
    print("\n" + "=" * 60)
    print("DAG特性演示")
    print("=" * 60)
    
    # 创建一个示例DAG
    dag = SubtaskDAG()
    
    # 添加子任务
    subtask1 = Subtask(
        id="task_1",
        description="数据清洗",
        task_type="cleaning",
        dependencies=[],
        difficulty=2.5,
        estimated_time=60.0
    )
    
    subtask2 = Subtask(
        id="task_2",
        description="探索性数据分析",
        task_type="eda",
        dependencies=["task_1"],
        difficulty=3.0,
        estimated_time=120.0
    )
    
    subtask3 = Subtask(
        id="task_3",
        description="特征工程",
        task_type="feature_engineering",
        dependencies=["task_1"],
        difficulty=3.5,
        estimated_time=180.0
    )
    
    subtask4 = Subtask(
        id="task_4",
        description="模型训练",
        task_type="modeling",
        dependencies=["task_2", "task_3"],
        difficulty=4.0,
        estimated_time=300.0
    )
    
    subtask5 = Subtask(
        id="task_5",
        description="结果报告",
        task_type="reporting",
        dependencies=["task_4"],
        difficulty=2.0,
        estimated_time=60.0
    )
    
    # 添加到DAG
    for task in [subtask1, subtask2, subtask3, subtask4, subtask5]:
        dag.add_subtask(task)
    
    print("1. 创建的DAG结构:")
    for node_id, node in dag.nodes.items():
        deps = " -> ".join(node.dependencies) if node.dependencies else "无"
        print(f"   {node_id}: {node.description} (类型: {node.task_type})")
        print(f"       依赖: {deps}")
    
    print(f"\n2. DAG属性:")
    print(f"   入口点: {dag.entry_points}")
    print(f"   出口点: {dag.exit_points}")
    print(f"   节点总数: {len(dag.nodes)}")
    
    print(f"\n3. 拓扑排序 (执行顺序):")
    execution_order = dag.get_execution_order()
    for i, node_id in enumerate(execution_order, 1):
        node = dag.nodes[node_id]
        print(f"   {i}. {node_id}: {node.description}")
    
    print(f"\n4. 关键路径分析:")
    critical_path, total_time = dag.compute_critical_path()
    print(f"   关键路径: {' -> '.join(critical_path)}")
    print(f"   总时间: {total_time} 秒")
    
    print(f"\n5. DAG验证:")
    is_valid, msg, issues = dag.validate_dag()
    print(f"   验证结果: {'✅ 有效' if is_valid else '❌ 无效'}")
    if issues:
        for issue in issues:
            print(f"   - {issue['severity']}: {issue['message']}")

def main():
    """主函数 - 运行完整演示"""
    print("开始智能任务规划与执行系统演示...")
    
    # 演示系统概览
    demonstrate_system_overview()
    
    # 演示DAG特性
    demonstrate_dag_features()
    
    # 演示任务规划
    dag = demonstrate_task_planning()
    
    if dag:
        # 演示任务执行
        demonstrate_task_execution(dag)
    
    # 演示经验学习
    demonstrate_experience_learning()
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)
    print("\n系统核心特点:")
    print("• 智能任务分解: 将复杂任务分解为有序的子任务DAG")
    print("• 经验驱动: 利用历史经验优化任务规划")
    print("• 约束验证: 确保任务规划符合规则和约束")
    print("• 动态适应: 根据任务类型和难度调整规划策略")
    print("• 并行执行: 支持子任务的并行执行以提高效率")

if __name__ == "__main__":
    main()