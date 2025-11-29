import time
import os
import random
from collections import Counter, defaultdict
import numpy as np
import json
import sys

# 设置随机种子确保可重现性
random.seed(42)
np.random.seed(42)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 60)
    print("智能任务规划与执行系统")
    print("=" * 60)

    # 1. 首先初始化不依赖其他模块的基础组件
    print("\n1. 初始化基础组件...")

    # 检查OpenAI API是否可用
    OPENAI_AVAILABLE = "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"]
    llm_type = "OpenAI API" if OPENAI_AVAILABLE else "本地模型（DeepSeek）"
    print(f"  * 检测到 LLM 服务: {llm_type}")

    # 2. 初始化经验池
    print("\n2. 初始化经验管理系统...")
    from experience.experience_pool import ExperiencePool
    experience_pool = ExperiencePool(capacity=50)

    # 3. 初始化任务执行器
    print("\n3. 初始化任务执行引擎...")
    from models.llm_manager import LLMManager
    llm_manager = LLMManager()
    from execution.task_executor import TaskExecutor
    task_executor = TaskExecutor(llm_manager)

    # 4. 初始化规划器
    print("\n4. 初始化任务规划系统...")
    from planning.subtask_planner import SubtaskPlanner
    planner = SubtaskPlanner(experience_pool)
    planner.set_task_executor(task_executor)  # 注入任务执行器

    # 5. 系统配置摘要
    embedding_type = "SentenceTransformer" if llm_manager.embedding_manager.embedding_model else "简易词袋模型"
    print(f"\n系统配置摘要:")
    print(f"  * LLM 服务: {llm_type}")
    print(f"  * 嵌入模型: {embedding_type}")
    print(f"  * 经验池容量: {experience_pool.capacity}")
    print(f"  * 最大规划尝试次数: {planner.max_prediction_attempts}")

    # 6. 定义测试任务集
    print("\n" + "=" * 60)
    print("6. 定义测试任务集")
    print("=" * 60)

    from task_generator import generate_random_task

    test_tasks = [
        {
            "task_id": 1001,
            "task_type": "data_analysis",
            "main_task": "Analyze customer churn data to identify key factors and provide recommendations",
            "difficulty": 4.2,
            "priority": 3.0
        },
        {
            "task_id": 1002,
            "task_type": "code_generation",
            "main_task": "Implement a Python function to calculate Fibonacci numbers with memoization",
            "difficulty": 3.5,
            "priority": 2.5
        },
        {
            "task_id": 1003,
            "task_type": "translation",
            "main_task": "Translate a technical document about AI ethics from English to Chinese",
            "difficulty": 3.0,
            "priority": 2.0
        },
        {
            "task_id": 1004,
            "task_type": "math_problem",
            "main_task": "Solve the equation: 3x^2 + 5x - 2 = 0 and verify the solution",
            "difficulty": 2.5,
            "priority": 1.5
        },
        {
            "task_id": 1005,
            "task_type": "story",
            "main_task": "Write a short story about a robot learning to paint",
            "difficulty": 4.0,
            "priority": 3.5
        },
        {
            "task_id": 1006,
            "task_type": "summary",
            "main_task": "Summarize a research paper about climate change impacts on agriculture",
            "difficulty": 3.8,
            "priority": 2.8
        },
        {
            "task_id": 1007,
            "task_type": "combined_task",
            "main_task": "Analyze survey data about user preferences, then write a Python script to visualize the results",
            "difficulty": 4.5,
            "priority": 4.0
        }
    ]

    # 7. 任务执行阶段
    print("\n" + "=" * 60)
    print("7. 任务执行阶段")
    print("=" * 60)

    for i, task in enumerate(test_tasks):
        print(f"\n{'-' * 50}")
        print(f"任务 #{i + 1}/{len(test_tasks)}: {task['task_type'].upper()}")
        print(f"任务描述: {task['main_task']}")
        print(f"难度: {task['difficulty']}/5.0, 优先级: {task['priority']}/5.0")
        print("-" * 50)

        # 规划任务DAG
        print("  [规划] 生成子任务DAG...")
        dag, status = planner.plan_subtasks(
            task_id=task["task_id"],
            task_type=task["task_type"],
            main_task=task["main_task"],
            difficulty=task["difficulty"]
        )

        if dag:
            print(f"  ✅ DAG规划成功! 共 {len(dag.nodes)} 个子任务")
            print("  [DAG结构]:")

            # ===== 调用DAG可视化和分析工具 =====
            try:
                # 尝试从正确的路径导入
                from utils.dag_utils import analyze_dag_parallelism, visualize_dag_with_parallelism
            except ImportError:
                try:
                    # 尝试从备用路径导入
                    from dag_utils import analyze_dag_parallelism, visualize_dag_with_parallelism
                    print("  ✓ 成功从备用路径导入DAG工具")
                except ImportError:
                    print("  ⚠️ DAG分析工具不可用，跳过可视化")

                    # 定义空函数作为回退
                    def analyze_dag_parallelism(dag):
                        return {
                            "critical_path": [],
                            "critical_time": 0,
                            "potential_speedup": 1.0,
                            "max_parallelism": 1
                        }

                    def visualize_dag_with_parallelism(dag, filename=None):
                        print(f"  ⚠️ 无法生成可视化，但会保存DAG结构到 {filename}")
                        # 保存基本DAG信息到文件
                        if filename:
                            with open(filename.replace('.png', '.txt'), 'w') as f:
                                f.write(f"DAG结构 ({len(dag.nodes)} 节点):\n")
                                for node_id, node in dag.nodes.items():
                                    f.write(f"- {node_id} [{node.task_type}]: {node.description}\n")
                                    if node.dependencies:
                                        f.write(f"  依赖: {', '.join(node.dependencies)}\n")

        else:
            print(f"  ❌ DAG规划失败: {status}")
            continue

        # 打印DAG结构
        for node_id, node in dag.nodes.items():
            deps = ", ".join(node.dependencies) if node.dependencies else "无依赖"
            print(f"    • 任务ID: {node_id}")
            print(f"      - 描述: {node.description}")
            print(f"      - 类型: {node.task_type}")
            print(f"      - 依赖: {deps}")
            print(f"      - 难度: {node.difficulty:.1f}/5.0, 预估时间: {node.estimated_time:.0f}秒")

        # 执行任务
        print("\n  [执行] 运行任务DAG...")
        start_time = time.time()

        try:
            # 使用任务执行器执行任务
            result = task_executor.execute_task(
                task_id=task["task_id"],
                task_type=task["task_type"],
                main_task=task["main_task"],
                dag=dag
            )
            execution_time = time.time() - start_time
            success = True
            failure_reason = None
        except Exception as e:
            execution_time = time.time() - start_time
            success = False
            result = str(e)
            failure_reason = str(e)
            print(f"  ❌ 任务执行失败: {str(e)}")

        # 存储执行经验
        from experience.experience_pool import Experience
        experience = Experience(
            task_id=task["task_id"],
            task_type=task["task_type"],
            prompt=task["main_task"],
            execution_time=execution_time,
            success=success,
            result=result,
            timestamp=time.time(),
            difficulty=task["difficulty"],
            subtask_dag=dag,
            failure_reason=failure_reason
        )

        experience_pool.add_experience(experience)
        print(f"  ✅ 任务执行{'成功' if success else '失败'}, 耗时: {execution_time:.2f}秒")
        print("  [执行结果预览]:")
        print(f"    {result[:200]}..." if len(result) > 200 else f"    {result}")

        # 更新规则置信度
        if not success:
            planner.checker.update_from_experience(experience)

        # 暂停以便观察
        time.sleep(1.0)

    # 8. 演示经验学习效果
    print("\n" + "=" * 60)
    print("8. 经验学习效果演示")
    print("=" * 60)

    # 执行一个与之前相似的任务，但使用已积累的经验
    similar_task = {
        "task_id": 2001,
        "task_type": "data_analysis",
        "main_task": "Analyze user behavior data to identify key engagement patterns and provide actionable recommendations",
        "difficulty": 4.3,
        "priority": 3.2
    }

    print(f"\n原始任务: Analyze customer churn data to identify key factors and provide recommendations")
    print(f"相似任务: {similar_task['main_task']}")
    print("\n[无经验时规划] vs [有经验时规划] 对比:")

    # 8.1 无经验时的规划（使用新的空经验池）
    print("\n  [无经验规划]")
    empty_pool = ExperiencePool(capacity=10)
    temp_planner = SubtaskPlanner(empty_pool)
    temp_planner.set_task_executor(task_executor)

    dag_no_exp, status_no_exp = temp_planner.plan_subtasks(
        task_id=similar_task["task_id"],
        task_type=similar_task["task_type"],
        main_task=similar_task["main_task"],
        difficulty=similar_task["difficulty"]
    )

    if dag_no_exp:
        print(f"    ✅ 规划成功，子任务数: {len(dag_no_exp.nodes)}")
        node_types_no_exp = [node.task_type for node in dag_no_exp.nodes.values()]
        print(f"    任务类型分布: {dict(Counter(node_types_no_exp))}")
    else:
        print(f"    ❌ 规划失败: {status_no_exp}")

    # 8.2 有经验时的规划（使用已填充的经验池）
    print("\n  [有经验规划]")
    dag_with_exp, status_with_exp = planner.plan_subtasks(
        task_id=similar_task["task_id"],
        task_type=similar_task["task_type"],
        main_task=similar_task["main_task"],
        difficulty=similar_task["difficulty"]
    )

    if dag_with_exp:
        print(f"    ✅ 规划成功，子任务数: {len(dag_with_exp.nodes)}")
        node_types_with_exp = [node.task_type for node in dag_with_exp.nodes.values()]
        print(f"    任务类型分布: {dict(Counter(node_types_with_exp))}")

        # 比较两个DAG的差异
        if dag_no_exp:
            diff_count = abs(len(dag_no_exp.nodes) - len(dag_with_exp.nodes))
            print(f"    与无经验相比: 子任务数量 {'增加' if diff_count > 0 else '减少'} {abs(diff_count)} 个")
    else:
        print(f"    ❌ 规划失败: {status_with_exp}")

    #DAG 可视化
    try:
        from utils.dag_utils import visualize_dag_with_parallelism

        # 生成无经验DAG可视化
        visualize_dag_with_parallelism(
            no_experience_dag,
            filename="/kaggle/working/no_experience_dag.png"
        )

        # 生成有经验DAG可视化
        visualize_dag_with_parallelism(
            with_experience_dag,
            filename="/kaggle/working/with_experience_dag.png"
        )

        print("  * DAG可视化已生成，可在输出文件中比较两种规划的差异")
    except ImportError:
        print("  ⚠️ DAG分析工具不可用，跳过可视化对比")

    # 9. 系统统计
    print("\n" + "=" * 60)
    print("9. 系统统计与总结")
    print("=" * 60)

    print(f"\n经验池统计:")
    print(f"  * 总经验数: {sum(len(exps) for exps in experience_pool.experiences.values())}")
    print(f"  * 按任务类型分布:")
    for task_type, exps in experience_pool.experiences.items():
        success_rate = sum(1 for exp in exps if exp.success) / len(exps) * 100 if exps else 0
        avg_time = sum(exp.execution_time for exp in exps) / len(exps) if exps else 0
        print(f"    - {task_type}: {len(exps)} 条经验 (成功率: {success_rate:.1f}%, 平均耗时: {avg_time:.2f}秒)")

    print(f"\n领域知识库统计:")
    domain_stats = {}
    for domain, rules in planner.checker.knowledge_base.domain_rules.items():
        domain_stats[domain] = {
            'rules': len(rules),
            'constraints': len(planner.checker.knowledge_base.domain_constraints.get(domain, []))
        }

    for domain, stats in domain_stats.items():
        print(f"  * {domain}: {stats['rules']} 条规则, {stats['constraints']} 个约束")

    # ===== 分析并可视化最近的经验DAG =====
    try:
        from utils.dag_utils import analyze_dag_parallelism, visualize_dag_with_parallelism

        # 获取最近的经验
        recent_experiences = []
        for exp_list in experience_pool.experiences.values():
            recent_experiences.extend(exp_list)

        # 按时间排序
        recent_experiences.sort(key=lambda x: x.timestamp, reverse=True)

        # 分析最近5个DAG
        for i, exp in enumerate(recent_experiences[:5]):
            if hasattr(exp, 'subtask_dag') and exp.subtask_dag:
                print(f"\n分析最近经验 #{i + 1} (任务ID: {exp.task_id}, 类型: {exp.task_type})")
                parallelism = analyze_dag_parallelism(exp.subtask_dag)
                print(f"  * 关键路径时间: {parallelism['critical_time']:.1f}秒")
                print(f"  * 潜在加速比: {parallelism['potential_speedup']:.2f}x")
                print(f"  * 最大并行度: {parallelism['max_parallelism']}")

                # 生成可视化
                visualize_dag_with_parallelism(
                    exp.subtask_dag,
                    filename=f"/kaggle/working/recent_experience_dag_{exp.task_id}.png"
                )

        print("\n  * 所有DAG可视化已生成，可在输出文件中查看")
    except ImportError:
        print("\n  ⚠️ DAG分析工具不可用，跳过详细分析")

    # 10. 结束信息
    print("\n" + "=" * 60)
    print("系统执行完成!")
    print("=" * 60)
    print(f"最终经验池大小: {sum(len(exps) for exps in experience_pool.experiences.values())} 条经验")
    print(f"累计处理任务类型: {len(experience_pool.experiences)} 种")
    print("系统已准备好处理更多任务!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n程序执行过程中发生错误: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)