import os
import time
import random
import numpy as np
from typing import Optional, Any
from models.embedding_models import LocalEmbeddingManager

# 全局模型配置
OPENAI_AVAILABLE = "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"]


class LLMManager:
    """统一的LLM管理器，根据可用性选择OpenAI或本地模型"""

    def __init__(self):
        self.openai_available = OPENAI_AVAILABLE
        self.local_llm_available = False
        self.embedding_manager = LocalEmbeddingManager()
        # 不尝试加载本地模型，直接使用模拟响应

    def compute_embeddings(self, text: str) -> np.ndarray:
        """统一的嵌入计算接口"""
        if self.openai_available:
            try:
                import openai
                response = openai.Embedding.create(
                    model="text-embedding-ada-002",
                    input=text
                )
                return np.array(response['data'][0]['embedding'])
            except Exception as e:
                print(f"OpenAI API调用错误: {e}")
                return self.embedding_manager.compute_embeddings(text)
        else:
            return self.embedding_manager.compute_embeddings(text)

    def generate_response(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
        """统一的LLM响应生成接口"""
        if self.openai_available:
            try:
                import openai
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"OpenAI API调用错误: {e}")

        # 所有模型都不可用，使用模拟响应
        return self._generate_simulated_response(prompt)

    def _generate_simulated_response(self, prompt: str) -> str:
        """生成模拟响应（当所有模型都不可用时）"""
        prompt_lower = prompt.lower()

        # 根据任务类型生成不同响应
        if any(term in prompt_lower for term in ["analyze", "data", "customer", "churn"]):
            responses = [
                "First, we need to clean the data to handle missing values and outliers. Then, exploratory data analysis should be performed to identify patterns. Finally, build a predictive model to identify key churn factors.",
                "The analysis should begin with data preprocessing, followed by feature engineering. Next, apply statistical methods to identify significant factors. Finally, develop actionable recommendations based on findings.",
                "Start with data quality assessment, then proceed to descriptive statistics. After identifying potential factors, use correlation analysis and machine learning models to validate hypotheses about churn drivers."
            ]
        elif any(term in prompt_lower for term in ["code", "implement", "program"]):
            responses = [
                "First, define the requirements clearly. Then, design the architecture with appropriate modules. Implement with unit tests for each component. Finally, integrate and validate against requirements.",
                "Begin with a requirements analysis phase. Next, create a detailed design document. Then implement with test-driven development. Finally, perform integration testing and documentation.",
                "Start by breaking down the problem into smaller components. Design interfaces between components. Implement each component with thorough error handling. Finally, test and optimize the solution."
            ]
        else:
            responses = [
                f"Based on the task '{prompt[:30]}...', a structured approach is recommended. Break down the task into manageable subtasks with clear dependencies and validation points at each stage.",
                f"To address '{prompt[:30]}...', consider the following steps: 1) Analyze requirements 2) Identify dependencies 3) Plan execution sequence 4) Validate intermediate results",
                f"Analysis of '{prompt[:30]}...' suggests a multi-step approach. First ensure all prerequisites are met, then proceed with sequential execution while monitoring for constraint violations."
            ]

        return random.choice(responses)