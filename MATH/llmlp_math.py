import math
import re
import json
import time
import random
import sys
import os
import torch
from tqdm import tqdm
from utils import _strip_string, extract_math_answer, is_equiv, most_frequent
import backoff
from model_selection1 import MarkovModelSelector
from LLMLP import LLMLP
from DebateLLM import DebateLLM
from prompt_lib import ROLE_MAP_MATH, SYSTEM_PROMPT_MATH, ROLE_MAP_DEBATE, SYSTEM_PROMPT_DEBATE
from prettytable import PrettyTable
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig

# 从命令行获取参数
SUB_DIR = sys.argv[1]
MIN_FILENAME = int(sys.argv[2])
MAX_FILENAME = int(sys.argv[3])
MODEL_NAME = sys.argv[4]  # 模型名称，例如 "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DIR_NAME = "dyLAN_math_results_" + MODEL_NAME
RESPONSES_TOTAL = DIR_NAME + "/responses_total.txt"
TOKENS_TOTAL = DIR_NAME + "/tokens_total.txt"

# 初始化全局变量
ACTIVATION = "listwise"
DEBATE_TYPE = "structured"
TYPE = "math_exp"  # 专为数学问题设计的类型
MAX_PROBLEMS = 10

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

def construct_message(agents, question, qtype="math_exp"):
    """为DyLAN框架构建消息（保留用于兼容性）"""
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct."}

    prefix_string = "Follow the given examples and answer the mathematics problem.\n\n" + question + "\n\nThese are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[-1]["content"] if isinstance(agent, list) else agent
        response = "\n\nOne agent's solution: ```{}```".format(agent_response)
        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\nUsing the reasoning from other agents as additional advice with critical thinking, can you give an updated answer? Examine your solution and that other agents step by step. Notice that the former answers might be all wrong."""
    return {"role": "user", "content": prefix_string}


def parse_question_answer(subdir, file):
    """解析MATH数据集中的问题和答案"""

    def find_math_answer(s):
        """从LaTeX格式中提取答案"""
        if 'boxed' not in s:
            return None

        ans = s.split('boxed')[-1]
        if ans.startswith('{'):
            stack = 1
            a = ''
            for c in ans[1:]:
                if c == '{':
                    stack += 1
                    a += c
                elif c == '}':
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()

        return _strip_string(a)

    with open(os.path.join(subdir, file), 'r') as fp:
        try:
            problem_data = json.load(fp)
        except Exception as e:
            print(f"Error loading JSON from {file}", e)
            raise e

        question = problem_data["problem"]
        prob_level = problem_data["level"]
        prob_type = problem_data["type"]

        try:
            prob_level = int(prob_level.split("Level ")[1])
        except:
            prob_level = None

        answer = find_math_answer(problem_data['solution'])
        return question, prob_level, prob_type, answer


def get_roles_for_model(model_type):
    """根据模型类型获取对应的角色配置"""
    if model_type == "debate":
        print("\n" + "=" * 50)
        print("SETTING UP DEBATE ROLES")
        print("=" * 50)
        roles = ["Proposer", "Opposer", "FactChecker", "Moderator"]
        role_map = ROLE_MAP_MATH
        print("\nDebate Roles Configuration:")
        for role in roles:
            print(f"- {role}: {role_map[role][:100]}...")
        return roles, role_map
    else:  # cooperation
        print("\n" + "=" * 50)
        print("SETTING UP COOPERATION ROLES")
        print("=" * 50)
        roles = ["Assistant", "Assistant", "Assistant", "Assistant"]
        role_map = ROLE_MAP_MATH
        print("\nCooperation Roles Configuration:")
        print(f"- All agents are configured as 'Assistant' with mathematical focus")
        return roles, role_map


def compute_accuracy(gt, pred_solutions):
    """计算预测答案的准确性"""
    pred_answers = []
    for pred_solution in pred_solutions:
        pred_answer = extract_math_answer(pred_solution)
        if pred_answer:
            pred_answers.append(pred_answer)

    if not pred_answers:
        return False, None

    consensus_answer, count = most_frequent(pred_answers, is_equiv)
    is_correct = is_equiv(gt, consensus_answer)

    print(f"GT: {gt}")
    print(f"Consensus: {consensus_answer}")
    print(f"Accuracy: {'✓' if is_correct else '✗'}")

    return is_correct, consensus_answer

class TaskDecomposer:
    def __init__(self, model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"):
        """初始化任务分解器"""
        self.model = model
        print(f"Initializing TaskDecomposer")

    def get_default_tasks(self):
        """返回默认任务列表"""
        return [
            {
                "id": "t1",
                "description": "理解问题要求",
                "dependencies": []
            },
            {
                "id": "t2",
                "description": "生成可能答案候选列表",
                "dependencies": ["t1"]
            },
            {
                "id": "t3",
                "description": "验证每个答案是否满足条件",
                "dependencies": ["t2"]
            },
            {
                "id": "t4",
                "description": "整合结果并提供最终答案",
                "dependencies": ["t3"]
            }
        ]

    def decompose_query(self, query):
        """将查询分解为子任务"""
        prompt = f"""
        请将以下问题分解为多个独立但相关的子任务，以便多个智能体可以协作解决。
        每个子任务应该是明确的、可执行的，并且应该指明它与其他子任务的依赖关系。

        问题: {query}

        重要要求:
        1. 每个子任务的描述开头必须包含问题背景信息
        2. 问题背景信息必须保持一致，所有子任务使用相同的背景描述
        3. 在背景信息后添加具体子任务描述

        请以严格的JSON格式返回子任务列表，格式如下:
        [
          {{
            "id": "t1",
            "description": "background\\n子任务具体描述",
            "dependencies": []  // 这个子任务依赖的其他子任务ID列表
          }},
          {{
            "id": "t2",
            "description": "background\\n子任务具体描述",
            "dependencies": ["t1"]  // 这个子任务依赖t1
          }}
        ]

        其他要求:
        1. 只返回纯JSON，不要包含任何额外说明或文本
        2. 不要包含```json或```标记
        3. 确保JSON格式正确，可被Python的json.loads()解析
        4. 不要包含注释(// ...)，JSON标准不支持注释
        5. 使用双引号，不要使用单引号
        6. 确保JSON是有效的，没有语法错误
        7. 子任务不超过 10个，且每个子任务需要尽可能详细

        请直接输出JSON，不要有任何前缀或后缀。
        """

        print("正在分解查询为子任务...")

        try:
            init_deepseek_model()

            if DEEPSEEK_MODEL is None or DEEPSEEK_TOKENIZER is None:
                raise RuntimeError("DeepSeek模型未正确初始化")

            # 准备输入
            inputs = DEEPSEEK_TOKENIZER(prompt, return_tensors="pt").to(device)

            # 生成响应
            outputs = DEEPSEEK_MODEL.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.2,
                do_sample=True,
                top_p=0.9,
                pad_token_id=DEEPSEEK_TOKENIZER.eos_token_id
            )

            # 解码响应
            response = DEEPSEEK_TOKENIZER.decode(outputs[0], skip_special_tokens=True)
            # 移除输入部分
            if response.startswith(prompt):
                response = response[len(prompt):].strip()

            print(f"Raw model response: {response[:500]}...")  # 打印响应预览

            # 尝试提取有效JSON
            tasks = self.extract_valid_json(response)
            if tasks is not None:
                # 确保返回的是字典列表
                validated_tasks = []
                for task in tasks:
                    if isinstance(task, dict):
                        validated_tasks.append(task)
                    elif isinstance(task, str):
                        try:
                            task_dict = json.loads(task)
                            if isinstance(task_dict, dict):
                                validated_tasks.append(task_dict)
                        except:
                            pass

                if validated_tasks:
                    print(f"成功解析 {len(validated_tasks)} 个任务")
                    return validated_tasks

            print("无法从响应中提取有效JSON，使用默认任务分解")
            return self.get_default_tasks()

        except Exception as e:
            print(f"分解查询时出错: {e}")
            return self.get_default_tasks()

    def extract_valid_json(self, response):
        """从响应中提取有效的JSON"""
        # 尝试1：直接解析整个响应
        try:
            return json.loads(response)
        except:
            pass

        # 尝试2：提取代码块（如果有）
        if "```json" in response:
            parts = response.split("```json")
            if len(parts) > 1:
                json_str = parts[1].split("```")[0].strip()
                try:
                    return json.loads(json_str)
                except:
                    pass
        elif "```" in response:
            parts = response.split("```")
            if len(parts) > 1:
                json_str = parts[1].strip()
                try:
                    return json.loads(json_str)
                except:
                    pass

        # 尝试3：使用括号匹配提取JSON数组
        start_index = response.find('[')
        if start_index != -1:
            count = 1
            i = start_index + 1
            while i < len(response):
                if response[i] == '[':
                    count += 1
                elif response[i] == ']':
                    count -= 1

                if count == 0:
                    try:
                        return json.loads(response[start_index:i + 1])
                    except:
                        pass
                i += 1

        # 尝试4：使用括号匹配提取JSON对象
        start_index = response.find('{')
        if start_index != -1:
            count = 1
            i = start_index + 1
            while i < len(response):
                if response[i] == '{':
                    count += 1
                elif response[i] == '}':
                    count -= 1

                if count == 0:
                    try:
                        return json.loads(response[start_index:i + 1])
                    except:
                        pass
                i += 1

        # 尝试5：清理响应并重试
        try:
            cleaned_response = response.strip()
            # 移除开头的非JSON内容
            if not cleaned_response.startswith(('[', '{')):
                json_start = min(
                    cleaned_response.find('[') if cleaned_response.find('[') >= 0 else float('inf'),
                    cleaned_response.find('{') if cleaned_response.find('{') >= 0 else float('inf')
                )
                if json_start != float('inf'):
                    cleaned_response = cleaned_response[json_start:]

            # 确保以]或}结尾
            if cleaned_response.endswith(','):
                cleaned_response = cleaned_response[:-1]
            if not cleaned_response.endswith((']', '}')):
                json_end = max(
                    cleaned_response.rfind(']') if cleaned_response.rfind(']') >= 0 else -1,
                    cleaned_response.rfind('}') if cleaned_response.rfind('}') >= 0 else -1
                )
                if json_end >= 0:
                    cleaned_response = cleaned_response[:json_end + 1]

            return json.loads(cleaned_response)
        except Exception as e:
            print(f"JSON清理和解析失败: {str(e)}")
            return None

def main():
    # 创建输出目录
    os.makedirs(DIR_NAME, exist_ok=True)

    # 初始化统计
    total_responses = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    response_dict = {}

    # 创建进度条
    pbar = tqdm(desc="Processing MATH problems", total=MAX_PROBLEMS)
    problem_count = 0
    # 遍历数据集
    for subdir, _, files in os.walk(SUB_DIR):
        for file in files:
            if problem_count >= MAX_PROBLEMS:
                break

            # 检查文件是否在指定范围内
            try:
                file_num = int(os.path.splitext(file)[0])
                if not (MIN_FILENAME <= file_num <= MAX_FILENAME):
                    continue
            except:
                continue

            print(f"\n{'=' * 50}")
            print(f"Processing file: {file}")
            print(f"{'=' * 50}")

            # 解析问题
            question, prob_level, prob_type, gt_answer = parse_question_answer(subdir, file)
            if not gt_answer:
                print(f"Skipping problem - could not extract ground truth answer")
                continue

            print(f"Problem: {question[:100]}...")
            print(f"Level: {prob_level}, Type: {prob_type}")
            print(f"GT Answer: {gt_answer}")

            # 任务分解和模型选择
            print("\nAnalyzing problem characteristics...")
            model_selector = MarkovModelSelector()
            decomposer = TaskDecomposer()
            for i, query in enumerate(question):
                print(f"\n===== Processing Query #{i + 1} =====")
                print(f"Question: {query[:100]}...")
                tasks = decomposer.decompose_query(query)  # 使用当前循环的 query
            model_type = model_selector.select_model(tasks)

            print(f"Selected model type: {model_type}")

            # 获取角色配置
            ROLES, _ = get_roles_for_model(model_type)

            # 初始化对应的模型
            if model_type == "debate":
                print("\nInitializing DebateLLM with structured debate format...")
                llmlp = DebateLLM(MODEL_NAME, len(ROLES), ROLES, 5, DEBATE_TYPE, TYPE, MODEL_NAME)
            else:  # cooperation
                print("\nInitializing LLMLP with cooperative approach...")
                llmlp = LLMLP(MODEL_NAME, len(ROLES), ROLES, 3, ACTIVATION, TYPE, MODEL_NAME)

            llmlp.zero_grad()

            # 执行推理
            print("\nStarting problem solving process...")
            with tqdm(total=3, desc="Processing rounds", leave=False) as pbar_rounds:
                # 前向传播
                res, resp_cnt, completions, prompt_tokens, completion_tokens = llmlp.forward(question)
                pbar_rounds.update(1)
                total_responses += resp_cnt
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens

                # 反向传播
                print("Starting backward pass...")
                imp_score = llmlp.backward(res)
                pbar_rounds.update(1)

                # 重新组织重要性分数
                imp_score = [[imp_score[idx] for idx in range(len(ROLES) * rid, len(ROLES) * (rid + 1))]
                             for rid in range(min(3, llmlp.rounds))]
                pbar_rounds.update(1)

            # 显示结果
            print("\n" + "=" * 50)
            print("SOLUTION RESULTS")
            print("=" * 50)

            # 创建表格展示结果
            pt = PrettyTable()
            pt.add_column("Role", ROLES)
            for rid in range(min(3, llmlp.rounds)):
                responses = []
                for idx in range(len(ROLES)):
                    if rid < len(completions[idx]) and completions[idx][rid]:
                        # 截取前100字符显示
                        responses.append(completions[idx][rid][:100] + "..." if len(completions[idx][rid]) > 100 else
                                         completions[idx][rid])
                    else:
                        responses.append("No response")
                pt.add_column(f"Round {rid + 1}", responses)

            print(pt)
            print(f"\nFinal Answer: {res}")
            print(f"API Calls: {resp_cnt}")
            print(f"Prompt Tokens: {prompt_tokens}")
            print(f"Completion Tokens: {completion_tokens}")

            # 检查准确性
            is_correct, _ = compute_accuracy(gt_answer, [res])

            # 保存结果
            response_dict[question] = (completions, gt_answer, prob_level, prob_type, is_correct)

            # # 更新进度
            # pbar.update(1)

            # 更新计数器和进度
            problem_count += 1
            pbar.update(1)

    pbar.close()

    # 保存结果
    print("\nSaving results...")
    os.makedirs(DIR_NAME, exist_ok=True)

    # 保存所有问题的结果
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = os.path.join(DIR_NAME, f"results_{timestamp}.json")
    with open(output_file, "w") as f:
        json.dump(response_dict, f, indent=2)

    # 保存API调用统计
    with open(RESPONSES_TOTAL, "a") as f:
        f.write(f"{total_responses}\n")

    # 保存token统计
    with open(TOKENS_TOTAL, "a") as f:
        f.write(f"Prompt tokens: {total_prompt_tokens}, Completion tokens: {total_completion_tokens}\n")

    # 计算总体准确率
    correct_count = sum(1 for data in response_dict.values() if data[4])
    total_count = len(response_dict)
    accuracy = correct_count / total_count if total_count > 0 else 0

    print("\n" + "=" * 50)
    print("TRAINING COMPLETED")
    print("=" * 50)
    print(f"Total problems processed: {total_count}")
    print(f"Correct answers: {correct_count}")
    print(f"Overall accuracy: {accuracy:.2%}")
    print(f"Total API calls: {total_responses}")
    print(f"Total prompt tokens: {total_prompt_tokens}")
    print(f"Total completion tokens: {total_completion_tokens}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()