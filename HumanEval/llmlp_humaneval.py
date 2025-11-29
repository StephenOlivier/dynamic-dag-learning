import ast
import json
import os
import openai
import random
import sys
import torch
import backoff
from LLMLP import LLMLP
from DebateLLM import DebateLLM
from utils1 import *
from prompt_lib1 import ROLE_MAP_Code
from model_selection1 import MarkovModelSelector
from prettytable import PrettyTable
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig

# openai.api_key =
# openai.api_base =
# openai.api_type =
# openai.api_version =

PART = int(sys.argv[1])
EXP_NAME = sys.argv[2]
MODEL_NAME = sys.argv[3]

ACTIVATION = "listwise"
TYPE = "code_completion"
# ROLES = ["Assistant", "Mathematician", "Mathematician", "Assistant"]
MODEL_NAME = sys.argv[4]
# ROLES = ast.literal_eval(sys.argv[5])
# JUDGES = ast.literal_eval(sys.argv[6])
DIR_NAME = "dyLAN_code_results_" + MODEL_NAME

SUBSET = 50

DEEPSEEK_MODEL = None
DEEPSEEK_TOKENIZER = None
MODEL_PATH = "/kaggle/input/deepseek-r1-0528/transformers/deepseek-r1-0528-qwen3-8b/1"  # 根据实际情况修改
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_rd_seed(seed):
    random.seed(seed)

def init_deepseek_model(use_cpu=False):
    """初始化全局DeepSeek模型实例（仅加载一次）"""
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

def get_roles_for_model(model_type):
    """根据模型类型获取对应的角色配置"""
    if model_type == "debate":
        print("\n" + "=" * 50)
        print("SETTING UP DEBATE ROLES")
        print("=" * 50)
        roles = ["Proposer", "Opposer", "FactChecker", "Moderator"]
        role_map = ROLE_MAP_Code
        print("\nDebate Roles Configuration:")
        for role in roles:
            print(f"- {role}: {role_map[role][:100]}...")
        return roles, role_map
    else:  # cooperation
        print("\n" + "=" * 50)
        print("SETTING UP COOPERATION ROLES")
        print("=" * 50)
        roles = ["Assistant", "Assistant", "Assistant", "Assistant"]
        role_map = ROLE_MAP_Code
        print("\nCooperation Roles Configuration:")
        print(f"- All agents are configured as 'Assistant' with mathematical focus")
        return roles, role_map


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
    set_rd_seed(0)
    # assert len(ROLES) > 0
    # assert len(JUDGES) > 0
    os.makedirs(DIR_NAME, exist_ok=True)

    qa_pairs = get_human_eval_qa_pairs()

    # with open(DIR_NAME + '/' + EXP_NAME + '_' + str(len(ROLES)) + str(len(JUDGES)) + '3.json', 'w') as f:
    #     f.write("")
    # with open(DIR_NAME + '/' + EXP_NAME + '_' + str(len(ROLES)) + str(len(JUDGES)) + '3.tests', 'w') as f:
    #     f.write("")

    # with open(DIR_NAME + '/' + EXP_NAME + '_' + str(len(ROLES)) + '3.json', 'w') as f:
    #     f.write("")
    # with open(DIR_NAME + '/' + EXP_NAME + '_' + str(len(ROLES)) + '3.tests', 'w') as f:
    #     f.write("")

    results, resp_cnts, importances = [], 0, []
    completion_list = []
    tests_list = []
    total_prompt_tokens, total_completion_tokens = 0, 0

    for task_id, que, entry_point in qa_pairs:
        qid = int(task_id.split("/")[-1])
        if qid < PART * SUBSET or qid >= (PART + 1) * SUBSET:
            continue

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
        res, resp_cnt, completions, prompt_tokens, completion_tokens, tests = llmlp.forward(que, entry_point)
        imp_score = llmlp.backward(res, que, entry_point)

        completion_list.append(completions)
        results.append({"task_id": task_id, "completion": res})
        resp_cnts += resp_cnt
        importances.append(imp_score)
        tests_list.append(tests)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens

        # with open(DIR_NAME + '/' + EXP_NAME + '_' + str(len(ROLES)) + str(len(JUDGES)) + '3.json', 'a') as f:
        #     f.write(json.dumps(completions) + '\n')
        # with open(DIR_NAME + '/' + EXP_NAME + '_' + str(len(ROLES)) + str(len(JUDGES)) + '3.tests', 'a') as f:
        #     f.write(json.dumps(tests) + '\n')

        with open(DIR_NAME + '/' + EXP_NAME + '_' + str(len(ROLES)) + '3.json', 'a') as f:
            f.write(json.dumps(completions) + '\n')
        with open(DIR_NAME + '/' + EXP_NAME + '_' + str(len(ROLES)) + '3.tests', 'a') as f:
            f.write(json.dumps(tests) + '\n')

    print(results)
    print(resp_cnts)
    print(importances)
    print(total_prompt_tokens, total_completion_tokens)

    # with open(DIR_NAME + '/' + EXP_NAME + '_' + str(len(ROLES)) + str(len(JUDGES)) + '3.txt', 'w') as f:
    #     f.write(str(resp_cnts) + " " + str(resp_cnts / len(qa_pairs)) + '\n')
    #     f.write(json.dumps(importances) + '\n')
    #     f.write(json.dumps([sum(pos) / len(qa_pairs) for pos in zip(*importances)]) + '\n')
    #     f.write(str(total_prompt_tokens) + " " + str(total_completion_tokens) + '\n')
    #
    # write_jsonl(DIR_NAME + '/' + EXP_NAME + '_' + str(len(ROLES)) + str(len(JUDGES)) + '3.jsonl', results)

    with open(DIR_NAME + '/' + EXP_NAME + '_' + str(len(ROLES)) + '3.txt', 'w') as f:
        f.write(str(resp_cnts) + " " + str(resp_cnts / len(qa_pairs)) + '\n')
        f.write(json.dumps(importances) + '\n')
        f.write(json.dumps([sum(pos) / len(qa_pairs) for pos in zip(*importances)]) + '\n')
        f.write(str(total_prompt_tokens) + " " + str(total_completion_tokens) + '\n')

    write_jsonl(DIR_NAME + '/' + EXP_NAME + '_' + str(len(ROLES)) + '3.jsonl', results)


if __name__ == "__main__":
    main()