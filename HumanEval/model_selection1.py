import numpy as np
import random
import json
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Union
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from scipy.spatial.distance import cosine
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig, AutoConfig

DEEPSEEK_MODEL = None
DEEPSEEK_TOKENIZER = None
MODEL_PATH = "/kaggle/input/deepseek-r1-0528/transformers/deepseek-r1-0528-qwen3-8b/1"  # 根据实际情况修改
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_deepseek_model(use_cpu=False):
    """初始化全局DeepSeek模型实例（仅加载一次）

    参数:
        use_cpu: 是否强制使用CPU（默认为False，尝试使用GPU）
    """
    global DEEPSEEK_MODEL, DEEPSEEK_TOKENIZER

    if DEEPSEEK_MODEL is not None:
        return

    print(f"正在加载DeepSeek模型从路径: {MODEL_PATH}")
    print(f"使用设备: {'CPU' if use_cpu else 'GPU（如果可用）'}")
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(config)
    try:
        # CPU模式不能使用4-bit量化
        if use_cpu:
            print("在CPU模式下加载模型（不支持4-bit量化）...")

            # 在CPU上使用float32（CPU上float16支持有限）
            DEEPSEEK_MODEL = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                device_map="cpu",
                config=AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True),
                trust_remote_code=True,
                torch_dtype=torch.float32
            )

            DEEPSEEK_TOKENIZER = AutoTokenizer.from_pretrained(
                MODEL_PATH
            )

            print("CPU模式下DeepSeek模型加载成功！")
            return

        # GPU模式 - 尝试使用4-bit量化
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Load model
        DEEPSEEK_MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            config=AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True),
            trust_remote_code=True,
            torch_dtype=torch.float16
        )

        # Load tokenizer
        DEEPSEEK_TOKENIZER = AutoTokenizer.from_pretrained(
            MODEL_PATH
        )

        print("DeepSeek模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")

        # 如果是CPU模式下的错误，可能是因为尝试使用量化
        if use_cpu:
            print("CPU模式下加载失败，尝试简化配置...")
            try:
                DEEPSEEK_MODEL = AutoModelForCausalLM.from_pretrained(
                    MODEL_PATH,
                    device_map="cpu",
                    config=AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True),
                    trust_remote_code=True,
                    torch_dtype=torch.float32
                )
                DEEPSEEK_TOKENIZER = AutoTokenizer.from_pretrained(
                    MODEL_PATH
                )

                print("CPU模式下模型加载成功（简化配置）！")
                return
            except Exception as e:
                print(f"CPU模式下模型加载失败: {e}")
                print("请检查模型路径和环境配置")
                raise

        # 尝试不使用量化加载（GPU模式）
        try:
            print("尝试不使用量化加载模型...")
            DEEPSEEK_MODEL = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                device_map="auto",
                trust_remote_code=True,
                config=AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True),
                torch_dtype=torch.float16
            )
            DEEPSEEK_TOKENIZER = AutoTokenizer.from_pretrained(
                MODEL_PATH
            )
            print("模型加载成功（无量化）！")
        except Exception as e:
            print(f"严重错误: 无法加载模型: {e}")
            print("请检查模型路径和环境配置")
            raise


class MarkovModelSelector:
    def __init__(self, device: str = "cuda"):
        """
        初始化基于马尔科夫决策过程的模型选择器

        参数:
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.device = device
        try:
            # 使用SentenceTransformer获取高质量文本嵌入
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print(f"Using device: {device} for embeddings")
        except Exception as e:
            print(f"Warning: Could not load embedding model: {e}")
            self.embedding_model = None

        # 设置判别阈值
        self.tau = 0.3  # 根据数学定义，通常取0.3

        # 用于deepseek模型
        self.deepseek_model = None
        self.tokenizer = None
        self.setup_deepseek()

    def setup_deepseek(self):
        """配置DeepSeek模型"""
        try:
            init_deepseek_model()
            self.tokenizer = DEEPSEEK_TOKENIZER
            self.deepseek_model = DEEPSEEK_MODEL
            print("DeepSeek model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load DeepSeek model: {e}")
            print("Falling back to simulated responses for model selection")

    def generate_answer(self, task_description: str, return_hidden_states=False) -> Union[
        str, Tuple[str, torch.Tensor]]:
        """使用DeepSeek模型生成响应，并可选择返回隐藏状态作为梯度代理"""
        if self.deepseek_model is None:
            # 回退到更精细的模拟响应
            print(f"Using simulated response for task: {task_description[:50]}...")
            response = self.generate_simulated_response(task_description)
            if return_hidden_states:
                # 为模拟响应创建一个简单的"隐藏状态"代理
                embedding = self.embedding_model.encode([response])[0] if self.embedding_model else np.random.randn(384)
                return response, torch.tensor(embedding)
            return response

        try:
            # 准备输入
            inputs = DEEPSEEK_TOKENIZER(task_description, return_tensors="pt", truncation=True, max_length=512).to(
                device)

            prompt_tokens = inputs.input_ids.shape[1]

            # 生成响应，同时获取隐藏状态（作为梯度代理）
            with torch.no_grad():
                outputs = DEEPSEEK_MODEL.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.2,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=DEEPSEEK_TOKENIZER.eos_token_id,
                    output_hidden_states=True,  # 请求隐藏状态
                    return_dict_in_generate=True  # 以字典形式返回生成结果
                )

            total_tokens = outputs.sequences[0].shape[0]
            completion_tokens = total_tokens - prompt_tokens  # 生成内容的 token 数量

            # 解码响应
            response = DEEPSEEK_TOKENIZER.decode(outputs.sequences[0], skip_special_tokens=True)
            # 移除输入部分（如果需要）
            if response.startswith(task_description):
                response = response[len(task_description):].strip()

            # 获取隐藏状态作为目标函数梯度的代理
            if return_hidden_states and hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                last_token_hidden_state = outputs.hidden_states[-1][0][-1].detach()
                # === 修改返回：包含 token 计数 ===
                return response, prompt_tokens, completion_tokens, last_token_hidden_state
            elif return_hidden_states:
                if self.embedding_model:
                    embedding = self.embedding_model.encode([response])[0]
                    # === 修改返回：包含 token 计数 ===
                    return response, prompt_tokens, completion_tokens, torch.tensor(embedding)
                else:
                    # === 修改返回：包含 token 计数 ===
                    return response, prompt_tokens, completion_tokens, torch.randn(768)

            return response, prompt_tokens, completion_tokens

        except Exception as e:
            print(f"Error generating DeepSeek response: {e}")
            # 回退到模拟响应
            response = self.generate_simulated_response(task_description)
            if return_hidden_states:
                embedding = self.embedding_model.encode([response])[0] if self.embedding_model else np.random.randn(384)
                return response, torch.tensor(embedding)
            return response

    def generate_simulated_response(self, task_description: str) -> str:
        """
        生成更精细的模拟响应，确保不同任务有不同响应
        """
        task_lower = task_description.lower()

        # 1. 数学问题响应
        if any(term in task_lower for term in
               ["calculate", "solve", "math", "equation", "field", "polynomial", "group", "ring", "degree"]):
            # 基于任务关键词生成不同响应
            if "field" in task_lower or "extension" in task_lower:
                responses = [
                    "To determine the degree of the field extension, we need to analyze the minimal polynomials of the generators and apply the tower law. Q(sqrt(2), sqrt(3), sqrt(18)) can be simplified since sqrt(18) = 3sqrt(2), so it's equivalent to Q(sqrt(2), sqrt(3)).",
                    "The field extension Q(sqrt(2), sqrt(3), sqrt(18)) has degree 4 over Q. This is because sqrt(18) = 3sqrt(2), so the extension is equivalent to Q(sqrt(2), sqrt(3)), which has degree 4 as [Q(sqrt(2)):Q] = 2 and [Q(sqrt(2), sqrt(3)):Q(sqrt(2))] = 2.",
                    "Considering the field extension Q(sqrt(2), sqrt(3), sqrt(18)), we note that sqrt(18) = 3sqrt(2), so this is equivalent to Q(sqrt(2), sqrt(3)). The degree is 4 because the minimal polynomial of sqrt(3) over Q(sqrt(2)) is x²-3, which is irreducible.",
                    "The degree of Q(sqrt(2), sqrt(3), sqrt(18)) over Q is 4. This follows from the fact that sqrt(18) = 3sqrt(2), so the extension is Q(sqrt(2), sqrt(3)), and the basis is {1, sqrt(2), sqrt(3), sqrt(6)}."
                ]
            elif "group" in task_lower or "subgroup" in task_lower or "normal" in task_lower:
                responses = [
                    "To find the index of <p> in S_5 where p = (1, 2, 5, 4)(2, 3), we first determine the order of p. The permutation p has cycle structure that gives it order 4, so |<p>| = 4. Since |S_5| = 120, the index is 120/4 = 30.",
                    "The permutation p = (1, 2, 5, 4)(2, 3) simplifies to a single cycle of length 5 after composition. Therefore, |<p>| = 5, and the index in S_5 (which has order 120) is 120/5 = 24.",
                    "Analyzing p = (1, 2, 5, 4)(2, 3) in S_5, we compute its order by finding the least common multiple of its cycle lengths. After simplification, p is a 5-cycle, so |<p>| = 5. The index is |S_5|/|<p>| = 120/5 = 24.",
                    "The permutation p = (1, 2, 5, 4)(2, 3) in S_5 has order 4 (not 5 as might be initially thought). This is because the composition results in a permutation with cycle structure that gives order 4. Thus, the index is 120/4 = 30."
                ]
            elif "polynomial" in task_lower or "zeros" in task_lower or "finite field" in task_lower:
                responses = [
                    "To find zeros of x^5 + 3x^3 + x^2 + 2x in Z_5, we evaluate at each element of Z_5. Testing x=0: 0+0+0+0=0 ✓; x=1: 1+3+1+2=7≡2≠0; x=2: 32+24+4+4=64≡4≠0; x=3: 243+81+9+6=339≡4≠0; x=4: 1024+192+16+8=1240≡0 ✓. So zeros are 0 and 4.",
                    "Evaluating x^5 + 3x^3 + x^2 + 2x in Z_5: at x=0 it's 0, at x=1 it's 1+3+1+2=7≡2, at x=2 it's 32+24+4+4=64≡4, at x=3 it's 243+81+9+6=339≡4, at x=4 it's 1024+192+16+8=1240≡0. Therefore, the zeros are 0 and 4.",
                    "The polynomial x^5 + 3x^3 + x^2 + 2x in Z_5 factors as x(x^4 + 3x^2 + x + 2). Testing values: x=0 is a root, x=1: 1+3+1+2=7≡2≠0, x=2: 16+12+2+2=32≡2≠0, x=3: 81+27+3+2=113≡3≠0, x=4: 256+48+4+2=310≡0. So zeros are 0 and 4.",
                    "In Z_5, x^5 + 3x^3 + x^2 + 2x = x(x^4 + 3x^2 + x + 2). Testing each element: f(0)=0, f(1)=1+3+1+2=7≡2, f(2)=32+24+4+4=64≡4, f(3)=243+81+9+6=339≡4, f(4)=1024+192+16+8=1240≡0. Thus, the zeros are 0 and 4."
                ]
            else:
                responses = [
                    "To solve this mathematical problem, we need to apply the relevant theorems and definitions. For field extensions, we use the tower law and minimal polynomials. For group theory problems, we consider subgroup properties and Lagrange's theorem.",
                    "This is a complex mathematical question requiring careful analysis of the underlying structures. We should identify the key concepts involved and apply the appropriate mathematical principles to derive the solution.",
                    "Let's approach this step by step, identifying the mathematical structures involved and applying the relevant theorems to determine the correct solution. The answer depends on understanding the specific properties of the mathematical objects in question.",
                    "The solution requires understanding of advanced mathematical concepts. We need to analyze the problem in the context of the relevant mathematical theory and apply the appropriate techniques to find the correct answer."
                ]
            return random.choice(responses)

        # 2. 语言/逻辑问题响应
        elif any(term in task_lower for term in ["word", "letter", "puzzle", "riddle"]):
            responses = [
                "I think the word might be 'starting': starting → staring → string → sting → sing → sin → in → I. Let me verify each step: staring is a word, string is a word, sting is a word, sing is a word, sin is a word, in is a word, and I is a word.",
                "After analysis, 'station' doesn't work properly as removing letters doesn't consistently yield valid words. 'Starting' appears to be the correct solution as it reduces correctly through all steps to a single letter.",
                "This is a word ladder problem. 'Starting' meets all criteria: starting → staring (valid), staring → string (valid), string → sting (valid), sting → sing (valid), sing → sin (valid), sin → in (valid), in → I (valid).",
                "The solution 'starting' satisfies all conditions of the puzzle. Each step of removing one letter results in a valid English word, continuing until only one letter remains."
            ]
            return random.choice(responses)

        # 3. 通用问题响应
        else:
            # 基于任务描述的哈希值选择不同响应，确保相同任务总是得到相同响应
            hash_val = hash(task_description) % 4
            responses = [
                f"Analysis shows that this problem requires careful consideration of multiple factors. Based on the task description '{task_description[:30]}...', the optimal approach involves...",
                f"To address this task '{task_description[:30]}...', we need to break it down into manageable components. The key challenge appears to be...",
                f"Considering the requirements of '{task_description[:30]}...', the most effective strategy would involve sequential processing with validation at each step...",
                f"After evaluating the problem '{task_description[:30]}...', I recommend a structured approach that accounts for dependencies and potential edge cases..."
            ]
            return responses[hash_val]

    def compute_gradient_angle_matrix(self, task: Dict, num_agents: int = 4) -> np.ndarray:
        """
        计算目标函数梯度之间的夹角相似度矩阵 G

        关键修改：基于马尔科夫决策过程定义 G[i,j] = <∇f_i, ∇f_j> / (||∇f_i|| · ||∇f_j||)
        这表示agent i和agent j的目标函数梯度之间的夹角的余弦值

        参数:
            task: 单个任务字典
            num_agents: 模拟的agent数量

        返回:
            G: 相似度矩阵，其中G[i,j]表示agent i和agent j的决策方向夹角的余弦值
        """
        task_desc = task.get("description", "")
        print(f"  Computing gradient angle matrix for task: {task_desc[:50]}...")

        # 为任务生成多个agent的响应及其隐藏状态（作为梯度代理）
        hidden_states = []
        responses = []

        for i in range(num_agents):
            response, hidden_state = self.generate_answer(task_desc, return_hidden_states=True)
            responses.append(response)
            # 确保隐藏状态是numpy数组
            if isinstance(hidden_state, torch.Tensor):
                hidden_state = hidden_state.cpu().numpy()
            hidden_states.append(hidden_state)

            # 打印第一个agent的响应示例
            if i == 0:
                print(f"  Example response: {response[:150]}...")
                print(f"  Hidden state shape: {hidden_state.shape}")

        # 计算相似度矩阵（基于梯度夹角）
        n = len(hidden_states)
        G = np.zeros((n, n))

        print("  Calculating gradient angles between agents...")
        for i in range(n):
            for j in range(n):
                if i == j:
                    G[i, j] = 1.0  # 自身夹角为0度，余弦值为1
                else:
                    # 计算余弦相似度（即夹角的余弦值）
                    vec_i = hidden_states[i].flatten()
                    vec_j = hidden_states[j].flatten()

                    # 确保向量不为零
                    norm_i = np.linalg.norm(vec_i)
                    norm_j = np.linalg.norm(vec_j)
                    if norm_i == 0 or norm_j == 0:
                        cos_sim = 0.0
                    else:
                        cos_sim = np.dot(vec_i, vec_j) / (norm_i * norm_j)

                    # 确保值在[-1,1]范围内
                    G[i, j] = np.clip(cos_sim, -1.0, 1.0)

        # 可视化梯度方向分布（可选）
        self.visualize_gradient_directions(hidden_states, task_desc)

        return G

    def visualize_gradient_directions(self, hidden_states, task_desc):
        """可视化梯度方向分布（简化版）"""
        try:
            # 降维到2D以便可视化
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt

            # 只在有足够数据点时进行可视化
            if len(hidden_states) >= 2:
                # 降维
                tsne = TSNE(n_components=2, random_state=42,
                            perplexity=min(5, len(hidden_states) - 1))  # 确保perplexity < n_samples
                embeddings_2d = tsne.fit_transform(np.array([hs.flatten() for hs in hidden_states]))

                # 绘制
                plt.figure(figsize=(8, 6))
                plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=range(len(hidden_states)),
                            cmap='viridis', s=100, alpha=0.7)

                # 添加标签
                for i, (x, y) in enumerate(embeddings_2d):
                    plt.text(x, y, f'Agent {i + 1}', fontsize=9)

                plt.title(f"Gradient Directions for: {task_desc[:30]}...")
                plt.xlabel("Dimension 1")
                plt.ylabel("Dimension 2")
                plt.colorbar(label="Agent ID")
                plt.grid(True, linestyle='--', alpha=0.7)

                # 保存图像
                os.makedirs("gradient_visualizations", exist_ok=True)
                plt.savefig(f"gradient_visualizations/gradient_directions_{hash(task_desc) % 10000}.png",
                            bbox_inches='tight')
                plt.close()

                print(
                    f"  Gradient direction visualization saved to gradient_visualizations/gradient_directions_{hash(task_desc) % 10000}.png")
        except Exception as e:
            print(f"  Could not create gradient visualization: {e}")

    def analyze_task_decision_direction(self, task: Dict, num_agents: int = 4) -> Tuple[float, Dict, np.ndarray]:
        """
        分析单个任务的决策方向

        修改：只返回决策方向（梯度相似度指标），而不是二元决策结果

        参数:
            task: 单个任务字典
            num_agents: 模拟的agent数量

        返回:
            (gradient_similarity, similarity_analysis, task_features):
                - gradient_similarity: 梯度相似度指标（平均夹角余弦值）
                - similarity_analysis: 相似度矩阵分析结果
                - task_features: 任务特征向量
        """
        task_desc = task.get("description", "")
        print(f"\nAnalyzing task decision direction for: {task_desc[:50]}...")

        # 为单个任务生成多个agent的响应
        # 使用基于梯度夹角的相似度矩阵
        G = self.compute_gradient_angle_matrix(task, num_agents)

        # 分析相似度矩阵
        similarity_analysis = self.analyze_similarity_matrix(G)

        # 提取任务特征
        task_features = self.extract_task_features(task)

        # 返回平均夹角余弦值作为梯度相似度指标
        gradient_similarity = similarity_analysis["G_mean"]

        return (gradient_similarity, similarity_analysis, task_features)

    def extract_task_features(self, task: Dict) -> np.ndarray:
        """
        从单个任务中提取特征向量

        参数:
            task: 单个任务字典

        返回:
            x: 任务特征向量 [复杂度, 确定性, 开放性, 任务类型]
        """
        description = task.get("description", "").lower()

        # 1. 任务复杂度 (基于描述长度和关键词)
        length_factor = min(len(description) / 300, 1.0)
        complexity_keywords = ["analyze", "evaluate", "compare", "determine", "calculate", "explain"]
        keyword_factor = sum(1 for kw in complexity_keywords if kw in description) / len(complexity_keywords)
        complexity = (length_factor + keyword_factor) / 2

        # 2. 任务确定性 (问题是否有明确答案)
        deterministic_keywords = ["calculate", "determine", "find", "what is", "solve"]
        deterministic = 1.0 if any(kw in description for kw in deterministic_keywords) else 0.5

        # 3. 任务开放性 (问题是否有多种可能答案)
        open_ended_keywords = ["discuss", "debate", "evaluate", "compare", "analyze"]
        openness = 1.0 if any(kw in description for kw in open_ended_keywords) else 0.3

        # 4. 任务类型
        if any(term in description for term in
               ["calculate", "solve", "math", "equation", "field", "polynomial", "group", "ring", "degree"]):
            task_type = 1.0  # 数学问题
        elif any(term in description for term in ["word", "letter", "puzzle", "riddle", "language"]):
            task_type = -1.0  # 语言问题
        else:
            task_type = 0.0  # 其他

        return np.array([complexity, deterministic, openness, task_type])

    def analyze_similarity_matrix(self, G: np.ndarray) -> Dict[str, float]:
        """
        分析相似度矩阵的特性

        参数:
            G: 相似度矩阵

        返回:
            analysis: 相似度矩阵的统计特征
        """
        n = G.shape[0]

        # 提取上三角部分（不包括对角线）进行分析
        triu_indices = np.triu_indices(n, k=1)
        off_diagonal_values = G[triu_indices]

        analysis = {
            "G_min": np.min(off_diagonal_values),
            "G_max": np.max(off_diagonal_values),
            "G_mean": np.mean(off_diagonal_values),
            "G_var": np.var(off_diagonal_values),
            "G_median": np.median(off_diagonal_values),
            "negative_count": np.sum(off_diagonal_values < -self.tau),
            "high_agreement_count": np.sum(off_diagonal_values > 0.7),
            "moderate_agreement_count": np.sum((off_diagonal_values > 0.3) & (off_diagonal_values <= 0.7)),
            "disagreement_count": np.sum(off_diagonal_values < 0)
        }

        # 打印关键指标
        print(f"  Gradient angle analysis:")
        print(f"    Min angle cosine: {analysis['G_min']:.4f} (τ = {self.tau})")
        print(f"    Max angle cosine: {analysis['G_max']:.4f}")
        print(f"    Mean angle cosine: {analysis['G_mean']:.4f}")
        print(f"    Negative pairs (cosθ < -τ): {analysis['negative_count']}")

        return analysis

    def select_model(self, tasks: List[Dict]) -> str:
        """
        使用马尔科夫决策过程选择最适合的模型类型

        关键修改：最终决策基于所有子任务的梯度夹角信息，而不是单个子任务的二元决策

        参数:
            tasks: 任务列表

        返回:
            model_type: "cooperation" 或 "debate"
        """
        print("\n" + "=" * 60)
        print(f"MARKOV DECISION PROCESS FOR MODEL SELECTION - Task Count: {len(tasks)}")
        print("=" * 60)

        # 打印任务描述以便调试
        print("\nTasks for analysis:")
        for i, task in enumerate(tasks):
            print(f"T{i + 1}: {task.get('description', '')[:100]}...")
        print("-" * 60)

        # 分析每个子任务的决策方向（梯度相似度）
        gradient_similarities = []  # 每个任务的梯度相似度（平均夹角余弦值）
        similarity_analyses = []  # 每个任务的相似度矩阵分析
        task_features_list = []  # 每个任务的特征

        for i, task in enumerate(tasks):
            # 分析单个任务的决策方向（只返回梯度相似度，不直接做决策）
            gradient_sim, sim_analysis, task_features = self.analyze_task_decision_direction(task)

            gradient_similarities.append(gradient_sim)
            similarity_analyses.append(sim_analysis)
            task_features_list.append(task_features)

            print(f"  Task {i + 1} gradient similarity: {gradient_sim:.4f}")
            print(f"    Min angle cosine: {sim_analysis['G_min']:.4f} (τ = {self.tau})")
            print(f"    Negative pairs: {sim_analysis['negative_count']}")

        # 计算全局梯度相似度指标
        global_min_cosine = min(sim['G_min'] for sim in similarity_analyses)
        global_negative_pairs = any(sim['negative_count'] > 0 for sim in similarity_analyses)
        avg_gradient_similarity = sum(gradient_similarities) / len(gradient_similarities)

        # 基于马尔科夫决策过程的数学定义做最终决策
        # Cooperation条件: min_{i,j} G_{ij} > τ
        # Debate条件: max_{i,j} G_{ij} < -τ or ∃i,j s.t. G_{ij} < -τ
        if global_min_cosine > self.tau:
            model_type = "cooperation"
            decision_reason = f"All agents show sufficient agreement (min G_ij = {global_min_cosine:.4f} > τ = {self.tau})"
        elif global_negative_pairs:
            model_type = "debate"
            min_negative = min(sim['G_min'] for sim in similarity_analyses if sim['G_min'] < -self.tau)
            decision_reason = f"Existence of conflicting views (min G_ij = {min_negative:.4f} < -τ)"
        else:
            # 如果没有明显对立意见但也不满足合作条件，基于平均相似度和任务类型决定
            avg_task_type = np.mean([tf[3] for tf in task_features_list])
            if avg_gradient_similarity > 0.5 or avg_task_type > 0:
                model_type = "cooperation"
                decision_reason = f"Moderate agreement with task type favoring cooperation (avg G_ij = {avg_gradient_similarity:.4f})"
            else:
                model_type = "debate"
                decision_reason = f"Low agreement without clear conflict, task type favors debate (avg G_ij = {avg_gradient_similarity:.4f})"

        # 打印分析结果
        print("\n" + "-" * 60)
        print("GLOBAL GRADIENT ANALYSIS")
        print("-" * 60)
        print(f"Global minimum angle cosine: {global_min_cosine:.4f} (τ = {self.tau})")
        print(f"Global negative pairs exist: {'Yes' if global_negative_pairs else 'No'}")
        print(f"Average gradient similarity: {avg_gradient_similarity:.4f}")

        print("\n" + "-" * 60)
        print("FINAL MODEL SELECTION DECISION")
        print("-" * 60)

        # 决策理由
        if model_type == "cooperation":
            print(f"DECISION: Selecting COOPERATION model")
        else:
            print(f"DECISION: Selecting DEBATE model")
        print(f"Rationale: {decision_reason}")

        print("=" * 60 + "\n")

        return model_type

    def get_roles_for_model(self, model_type: str, **kwargs) -> Tuple[List[str], Dict[str, str]]:
        """
        获取模型对应的ROLES和ROLE_MAP

        参数:
            model_type: 模型类型 ("cooperation" 或 "debate")

        返回:
            (roles, role_map): 角色列表和角色映射
        """
        from prompt_lib1 import ROLE_MAP, ROLE_MAP_DEBATE

        if model_type == "debate":
            print("\n" + "=" * 50)
            print("SETTING UP DEBATE ROLES")
            print("=" * 50)

            roles = ["Proposer", "Opposer", "FactChecker", "Moderator"]
            role_map = ROLE_MAP_DEBATE

            print("\nDebate Roles Configuration:")
            for role in roles:
                print(f"- {role}: {role_map[role][:100]}...")

            return roles, role_map
        else:  # cooperation
            print("\n" + "=" * 50)
            print("SETTING UP COOPERATION ROLES")
            print("=" * 50)

            # 根据任务特征选择更合适的合作角色
            roles = ["Assistant", "Assistant", "Assistant", "Assistant"]
            role_map = ROLE_MAP

            print("\nCooperation Roles Configuration:")
            print(f"- All agents are configured as 'Assistant' with collaborative focus")

            return roles, role_map
