import os
import time
import random
import numpy as np
import torch
from typing import Optional, Any
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import openai
from models.embedding_models import LocalEmbeddingManager

# 全局模型配置
DEEPSEEK_MODEL = None
DEEPSEEK_TOKENIZER = None
MODEL_PATH = "/kaggle/input/deepseek-r1-0528/transformers/deepseek-r1-0528-qwen3-8b/1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OPENAI_AVAILABLE = "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"]


def init_deepseek_model(use_cpu=False):
    """初始化全局DeepSeek模型实例（仅加载一次）"""
    global DEEPSEEK_MODEL, DEEPSEEK_TOKENIZER
    if DEEPSEEK_MODEL is not None:
        return
    print(f"正在加载DeepSeek模型从路径: {MODEL_PATH}")
    print(f"使用设备: {'CPU' if use_cpu else 'GPU（如果可用）'}")
    try:
        # CPU模式不能使用4-bit量化
        if use_cpu:
            print("在CPU模式下加载模型（不支持4-bit量化）...")
            DEEPSEEK_MODEL = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                device_map="cpu",
                config=AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True),
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
            DEEPSEEK_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)
            print("CPU模式下DeepSeek模型加载成功！")
            return
        # GPU模式 - 尝试使用4-bit量化
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        DEEPSEEK_MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            config=AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True),
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        DEEPSEEK_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)
        print("DeepSeek模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("尝试简化配置...")
        try:
            # 尝试不使用量化加载
            DEEPSEEK_MODEL = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                device_map="auto",
                trust_remote_code=True,
                config=AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True),
                torch_dtype=torch.float16
            )
            DEEPSEEK_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)
            print("模型加载成功（无量化）！")
        except Exception as e:
            print(f"严重错误: 无法加载模型: {e}")
            print("将仅使用SentenceTransformer嵌入")


class LLMManager:
    """统一的LLM管理器，根据可用性选择OpenAI或本地模型"""

    def __init__(self):
        self.openai_available = OPENAI_AVAILABLE
        self.local_llm_available = False
        self.embedding_manager = LocalEmbeddingManager()
        self._setup_local_model()

    def _setup_local_model(self):
        """配置本地模型"""
        if not self.openai_available:
            try:
                init_deepseek_model()
                self.local_llm_available = True
                print("本地LLM模型加载成功，将用于任务生成")
            except Exception as e:
                print(f"无法加载本地LLM模型: {e}")
                self.local_llm_available = False

    def compute_embeddings(self, text: str) -> np.ndarray:
        """统一的嵌入计算接口"""
        if self.openai_available:
            try:
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
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"OpenAI API调用错误: {e}")

        # OpenAI不可用或调用失败，使用本地模型
        if self.local_llm_available:
            try:
                # 准备输入
                inputs = DEEPSEEK_TOKENIZER(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(device)

                # 生成响应
                with torch.no_grad():
                    outputs = DEEPSEEK_MODEL.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=DEEPSEEK_TOKENIZER.eos_token_id
                    )

                # 解码响应
                response = DEEPSEEK_TOKENIZER.decode(outputs[0], skip_special_tokens=True)
                # 移除输入部分
                if response.startswith(prompt):
                    response = response[len(prompt):].strip()

                return response

            except Exception as e:
                print(f"DeepSeek生成错误: {e}")

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