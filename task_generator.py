import random
import string
from typing import Dict, List, Optional, Union, Any
import json
import time


class Task:
    """任务数据类，表示单个任务的基本信息"""

    def __init__(self, task_id: int, prompt: str, task_type: str, priority: float = 1.0):
        """
        初始化一个任务对象

        Args:
            task_id: 任务的唯一标识符
            prompt: 任务的具体内容或指令
            task_type: 任务类型（如data_analysis, code_generation等）
            priority: 任务优先级（1.0-5.0，越高表示越重要）
        """
        self.task_id = task_id
        self.prompt = prompt
        self.task_type = task_type
        self.priority = priority
        self.timestamp = time.time()
        self.metadata: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """将任务对象转换为字典"""
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "task_type": self.task_type,
            "priority": self.priority,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """从字典数据创建任务对象"""
        task = cls(
            task_id=data["task_id"],
            prompt=data["prompt"],
            task_type=data["task_type"],
            priority=data["priority"]
        )
        task.timestamp = data.get("timestamp", time.time())
        task.metadata = data.get("metadata", {})
        return task

    def __str__(self) -> str:
        return f"Task(id={self.task_id}, type={self.task_type}, priority={self.priority:.1f})"


class TaskGenerator:
    """任务生成器，支持多种任务类型的随机生成"""

    def __init__(self, seed: Optional[int] = None):
        """
        初始化任务生成器

        Args:
            seed: 随机数种子，用于可重现的随机生成
        """
        self.random = random.Random(seed)
        self.task_counter = 0

        # 任务类型定义
        self.task_types = [
            'data_analysis',
            'code_generation',
            'translation',
            'math_problem',
            'story',
            'summary',
            'combined_task'
        ]

        # 任务模板
        self.task_templates = {
            'data_analysis': [
                "Analyze {dataset} data to identify key patterns and insights",
                "Perform exploratory analysis on {dataset} dataset to understand trends",
                "Create a predictive model for {target} using {dataset} data",
                "Generate actionable recommendations based on analysis of {dataset} data",
                "Visualize key relationships in {dataset} dataset to support decision making"
            ],
            'code_generation': [
                "Implement a Python function that {functionality}",
                "Create a {language} class to handle {feature}",
                "Write a script that {purpose} with error handling",
                "Develop a {component_type} module for {system}",
                "Build a {tool_type} utility that {functionality}"
            ],
            'translation': [
                "Translate the following text from {source_lang} to {target_lang}: '{text}'",
                "Provide a professional translation of this {content_type} from {source_lang} to {target_lang}",
                "Translate and localize this content for {target_region} audience: '{text}'",
                "Adapt the following {content_type} culturally for {target_lang} readers: '{text}'",
                "Translate this technical documentation from {source_lang} to {target_lang}"
            ],
            'math_problem': [
                "Solve the equation: {equation}",
                "Calculate the value of {expression}",
                "Find the derivative of the function f(x) = {function}",
                "Determine the integral of {function} with respect to x",
                "Solve the system of equations: {equations}"
            ],
            'story': [
                "Write a short story about {theme}",
                "Create a narrative exploring the concept of {concept}",
                "Compose a tale about {character} who discovers {discovery}",
                "Write a creative story where {premise}",
                "Craft a short narrative with the theme of {theme} in a {setting} setting"
            ],
            'summary': [
                "Summarize the main points of an article on {topic}",
                "Provide a concise summary of the research paper about {subject}",
                "Summarize the key arguments from the debate on {issue}",
                "Create a brief executive summary of the report on {topic}",
                "Summarize the findings from the study on {research_area}"
            ],
            'combined_task': [
                "First {task1}, then {task2} to produce a final result",
                "Analyze {data} to extract insights, then write code to visualize the findings",
                "Research {topic}, summarize key points, and provide actionable recommendations",
                "Translate the {document}, then analyze the sentiment of the translated content",
                "Solve the mathematical problem {problem}, then write a program to verify the solution"
            ]
        }

        # 优先级映射
        self.priority_map = {
            'data_analysis': 3.5,
            'code_generation': 4.0,
            'translation': 2.5,
            'math_problem': 2.0,
            'story': 3.0,
            'summary': 3.0,
            'combined_task': 4.5
        }

        # 模板参数
        self.template_params = {
            'dataset': ['customer', 'sales', 'user behavior', 'market trends', 'financial', 'product usage'],
            'target': ['customer churn', 'sales growth', 'user engagement', 'market performance'],
            'functionality': ['calculates factorial', 'sorts an array', 'processes text data',
                              'generates random numbers', 'validates email addresses'],
            'language': ['Python', 'JavaScript', 'Java', 'C++', 'Rust'],
            'feature': ['data validation', 'user authentication', 'file processing', 'API integration'],
            'purpose': ['automates data collection', 'processes user inputs', 'generates reports'],
            'component_type': ['data processing', 'user interface', 'backend service', 'database connector'],
            'system': ['e-commerce platform', 'data analytics dashboard', 'content management system'],
            'tool_type': ['command-line', 'web-based', 'desktop', 'cloud-based'],
            'source_lang': ['English', 'Spanish', 'French', 'German', 'Chinese', 'Japanese'],
            'target_lang': ['English', 'Spanish', 'French', 'German', 'Chinese', 'Japanese'],
            'text': ['The quick brown fox jumps over the lazy dog',
                     'Artificial intelligence is transforming industries',
                     'Climate change poses significant challenges to global ecosystems'],
            'content_type': ['article', 'document', 'email', 'social media post', 'technical manual'],
            'target_region': ['North American', 'European', 'Asian', 'Latin American'],
            'equation': ['3x^2 + 5x - 2 = 0', '2x + 7 = 15', 'x^2 - 4x + 4 = 0', '5x - 3 = 2x + 9'],
            'expression': ['(3 + 7) * 5 / 2', '2^4 + 3^2', 'sqrt(25) + 10', 'log(100) + e^2'],
            'function': ['x^3 + 2x^2 - 5x + 1', '3x^2 + 4x - 7', 'sin(x) + cos(x)', 'e^x * ln(x)'],
            'equations': ['x + y = 5, 2x - y = 1', '3x + 2y = 12, x - y = 1', '2x + 3y = 6, 4x - y = 5'],
            'theme': ['a journey of self-discovery', 'the power of friendship', 'overcoming obstacles',
                      'the consequences of technology', 'a second chance'],
            'concept': ['time travel', 'artificial consciousness', 'parallel universes', 'the nature of reality'],
            'character': ['a scientist', 'an artist', 'a detective', 'a teacher', 'an explorer'],
            'discovery': ['an ancient artifact', 'a hidden talent', 'a secret identity', 'an unexpected truth'],
            'premise': ['robots develop emotions', 'time freezes for everyone except one person',
                        'dreams become reality', 'memories can be shared'],
            'setting': ['futuristic city', 'remote island', 'small town', 'space station', 'medieval kingdom'],
            'topic': ['climate change', 'artificial intelligence', 'economic inequality', 'healthcare innovation'],
            'subject': ['machine learning applications', 'biological diversity', 'quantum computing',
                        'renewable energy'],
            'issue': ['data privacy', 'social media influence', 'sustainable development', 'global health'],
            'research_area': ['cognitive psychology', 'astrophysics', 'medical genetics', 'environmental science'],
            'data': ['user survey responses', 'social media interactions', 'sales transaction history',
                     'website analytics'],
            'document': ['user manual', 'business proposal', 'research paper', 'marketing brochure'],
            'problem': ['finding the maximum value of a quadratic function', 'calculating compound interest',
                        'determining probability of events'],
            'task1': ['analyze the data', 'solve the equation', 'translate the text', 'research the topic'],
            'task2': ['create a visualization', 'write a report', 'develop a recommendation', 'build a model']
        }

    def _fill_template(self, template: str, task_type: str) -> str:
        """填充模板中的占位符"""
        result = template
        max_attempts = 5

        for _ in range(max_attempts):
            start_idx = result.find('{')
            if start_idx == -1:
                break

            end_idx = result.find('}', start_idx)
            if end_idx == -1:
                break

            placeholder = result[start_idx + 1:end_idx]
            if placeholder in self.template_params:
                value = self.random.choice(self.template_params[placeholder])
                result = result[:start_idx] + value + result[end_idx + 1:]
            else:
                # 如果找不到对应的参数，跳过这个占位符
                result = result[:start_idx] + result[end_idx + 1:]

        return result

    def generate_task(self, task_id: Optional[int] = None, task_type: Optional[str] = None) -> Task:
        """
        生成一个随机任务

        Args:
            task_id: 指定任务ID，如果不指定则自增
            task_type: 指定任务类型，如果不指定则随机选择

        Returns:
            Task: 生成的任务对象
        """
        if task_id is None:
            self.task_counter += 1
            task_id = self.task_counter

        if task_type is None:
            task_type = self.random.choice(self.task_types)

        # 确保task_type有效
        if task_type not in self.task_templates:
            task_type = self.random.choice(self.task_types)

        # 选择模板并填充
        template = self.random.choice(self.task_templates[task_type])
        prompt = self._fill_template(template, task_type)

        # 获取优先级
        priority = self.priority_map.get(task_type, 2.5)

        # 创建任务
        task = Task(
            task_id=task_id,
            prompt=prompt,
            task_type=task_type,
            priority=priority
        )

        # 添加额外元数据
        task.metadata = {
            "generator_version": "1.0",
            "complexity_estimate": self._estimate_complexity(task_type, prompt),
            "keywords": self._extract_keywords(prompt)
        }

        return task

    def generate_batch(self, count: int, task_types: Optional[List[str]] = None) -> List[Task]:
        """
        批量生成任务

        Args:
            count: 要生成的任务数量
            task_types: 限制生成的任务类型列表，如果为None则使用所有类型

        Returns:
            List[Task]: 生成的任务列表
        """
        tasks = []
        for i in range(count):
            if task_types:
                task_type = self.random.choice(task_types)
            else:
                task_type = None
            tasks.append(self.generate_task(task_id=self.task_counter + i + 1, task_type=task_type))
        self.task_counter += count
        return tasks

    def generate_specific_task(self, task_type: str, custom_params: Optional[Dict[str, str]] = None) -> Task:
        """
        生成特定类型的任务，可选自定义参数

        Args:
            task_type: 任务类型
            custom_params: 自定义模板参数

        Returns:
            Task: 生成的任务对象
        """
        self.task_counter += 1
        task_id = self.task_counter

        if task_type not in self.task_templates:
            raise ValueError(f"Unsupported task type: {task_type}")

        # 选择模板
        template = self.random.choice(self.task_templates[task_type])

        # 填充模板
        if custom_params:
            result = template
            for placeholder, value in custom_params.items():
                result = result.replace(f"{{{placeholder}}}", value)
            prompt = result
        else:
            prompt = self._fill_template(template, task_type)

        # 获取优先级
        priority = self.priority_map.get(task_type, 2.5)

        return Task(
            task_id=task_id,
            prompt=prompt,
            task_type=task_type,
            priority=priority
        )

    def _estimate_complexity(self, task_type: str, prompt: str) -> float:
        """估计任务复杂度（1.0-5.0）"""
        # 基础复杂度
        base_complexity = {
            'data_analysis': 3.5,
            'code_generation': 4.0,
            'translation': 2.0,
            'math_problem': 3.0,
            'story': 2.5,
            'summary': 2.5,
            'combined_task': 4.5
        }

        complexity = base_complexity.get(task_type, 2.5)

        # 根据提示内容调整
        if "complex" in prompt.lower() or "advanced" in prompt.lower():
            complexity += 0.5
        if "multiple" in prompt.lower() or "various" in prompt.lower() or "different" in prompt.lower():
            complexity += 0.3
        if len(prompt.split()) > 20:
            complexity += 0.2

        return min(max(complexity, 1.0), 5.0)

    def _extract_keywords(self, prompt: str) -> List[str]:
        """从任务提示中提取关键词"""
        # 简单实现：取名词和重要动词
        words = prompt.lower().split()
        keywords = []

        # 常见重要词
        important_words = ['analyze', 'create', 'develop', 'implement', 'translate', 'solve', 'write', 'summarize']
        for word in words:
            word_clean = word.strip('.,:;!?()[]{}"\'')
            if word_clean in important_words or len(word_clean) > 4:
                keywords.append(word_clean)

        return keywords[:5]  # 限制关键词数量


# 为了向后兼容，保留原始函数接口
def generate_random_task(task_id: int) -> Task:
    """
    生成随机任务（兼容旧接口）

    Args:
        task_id: 任务ID

    Returns:
        Task: 生成的任务对象
    """
    generator = TaskGenerator()
    return generator.generate_task(task_id=task_id)


# 示例用法
if __name__ == "__main__":
    # 初始化生成器
    generator = TaskGenerator(seed=42)

    print("=== 任务生成示例 ===")

    # 生成单个任务
    task1 = generator.generate_task()
    print(f"任务1: {task1}")
    print(f"  内容: {task1.prompt}")
    print(f"  优先级: {task1.priority}")
    print(f"  元数据: {task1.metadata}")

    # 生成特定类型的任务
    task2 = generator.generate_specific_task(
        task_type="code_generation",
        custom_params={
            "functionality": "sorts a list of numbers using quicksort algorithm",
            "language": "Python"
        }
    )
    print(f"\n任务2: {task2}")
    print(f"  内容: {task2.prompt}")

    # 批量生成任务
    print("\n批量生成任务:")
    tasks = generator.generate_batch(5, task_types=["data_analysis", "summary"])
    for i, task in enumerate(tasks, 1):
        print(f"  任务 {i}: [{task.task_type}] {task.prompt[:50]}...")