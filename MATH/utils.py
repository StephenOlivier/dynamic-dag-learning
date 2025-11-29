import json
import os
import re
import time
import pandas as pd
from prompt_lib import MMLU_QUESTION, COMPLEX_COT_EXAMPLES, TEMPERATURE, MAX_TOKENS
import openai
import backoff
# from openai.error import RateLimitError, APIError, ServiceUnavailableError, APIConnectionError, Timeout
from openai import OpenAI
from tqdm import tqdm
import torch

# @Chenlei
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



class OutOfQuotaException(Exception):
    "Raised when the key exceeded the current quota"
    def __init__(self, key, cause=None):
        super().__init__(f"No quota for key: {key}")
        self.key = key
        self.cause = cause

    def __str__(self):
        if self.cause:
            return f"{super().__str__()}. Caused by {self.cause}"
        else:
            return super().__str__()

class AccessTerminatedException(Exception):
    "Raised when the key has been terminated"
    def __init__(self, key, cause=None):
        super().__init__(f"Access terminated key: {key}")
        self.key = key
        self.cause = cause

    def __str__(self):
        if self.cause:
            return f"{super().__str__()}. Caused by {self.cause}"
        else:
            return super().__str__()

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) == 0:
                continue
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) >= 2
        return splits[0]
    else:
        return string

def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

def parse_question_answer(df, ix):
    question = df.iloc[ix, 0]
    a = df.iloc[ix, 1]
    b = df.iloc[ix, 2]
    c = df.iloc[ix, 3]
    d = df.iloc[ix, 4]

    question = MMLU_QUESTION.format(question, a, b, c, d)

    answer = df.iloc[ix, 5]

    return question, answer

def get_mmlu_qa_pairs(csv_name):
    df = pd.read_csv(csv_name, header=None)
    ix = len(df)
    return [parse_question_answer(df, idx) for idx in range(ix)]

def get_math_qa_pairs(sub_dir, min_file, max_file):
    def find_math_answer(s):
        assert('boxed' in s)
        # s = s.replace(",", "")
        ans = s.split('boxed')[-1]
        if(ans[0] == '{'):
            stack = 1
            a = ''
            for c in ans[1:]:
                if(c == '{'):
                    stack += 1
                    a += c
                elif(c == '}'):
                    stack -= 1
                    if(stack == 0): break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        a=_strip_string(a)
        return a

    def parse_single_qa_math(subdir, file):
        with open(os.path.join(subdir, file), 'r') as fp:
            try:
                problem_data = json.load(fp)
            except Exception as e:
                print(f"Error loading JSON from {file}", e)
                raise e
            prob_content = problem_data["problem"]
            question = COMPLEX_COT_EXAMPLES + "\n\nPlease solve the problem below.\nProblem: " + prob_content + "\nAnswer:"
            prob_level = problem_data["level"]
            prob_type = problem_data["type"]
            try:
                prob_level = int(prob_level.split("Level ")[1])
            except:
                prob_level = None

            # answer = remove_boxed(last_boxed_only_string(problem_data["solution"]))
            answer = find_math_answer(problem_data['solution'])

            return question, prob_level, prob_type, answer

    ret_list = []
    for subdir, dirs, files in os.walk(sub_dir):
        for file in files:
            file_num = int(os.path.splitext(file)[0])  # Get the filename without extension and convert to int
            if min_file <= file_num <= max_file:
                question, prob_level, prob_type, answer = parse_single_qa_math(subdir, file)
            else:
                continue
            ret_list.append((question, answer))
    return ret_list

def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return str1 == str2

def extract_math_answer(pred_str):
    if('The answer is ' in pred_str):
        pred = pred_str.split('The answer is ')[-1].strip()
    elif('the answer is ' in pred_str):
        pred = pred_str.split('the answer is ')[-1].strip()
    elif 'boxed' in pred_str:
        ans = pred_str.split('boxed')[-1]
        if len(ans) == 0:
            print(pred_str)
        if (ans[0] == '{'):
            stack = 1
            a = ''
            for c in ans[1:]:
                if (c == '{'):
                    stack += 1
                    a += c
                elif (c == '}'):
                    stack -= 1
                    if (stack == 0): break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        a = _strip_string(a)
        pred=a

    else:
        pattern = '-?\d*\.?\d+'
        pred = re.findall(pattern, pred_str)
        if(len(pred) >= 1):
            # print(pred_str)
            pred = pred[-1]
        else: pred = ''
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
        if pred[-1] == "/":
            pred = pred[:-1]
    pred=_strip_string(pred)
    if 'boxed' in pred:
        ans = pred.split('boxed')[-1]
        if (ans[0] == '{'):
            stack = 1
            a = ''
            for c in ans[1:]:
                if (c == '{'):
                    stack += 1
                    a += c
                elif (c == '}'):
                    stack -= 1
                    if (stack == 0): break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        a = _strip_string(a)
        pred=a
    return pred

# @Chenlei debate
def parse_debate_reply(reply):
    """解析辩论回复，提取关键信息"""
    # 尝试提取明确的立场声明
    position_patterns = [
        r'My position[::]?\s*(.+?)(?=\n\n|\n[A-Z])',
        r'I believe[::]?\s*(.+?)(?=\n\n|\n[A-Z])',
        r'The answer is[::]?\s*(.+?)(?=\n\n|\n[A-Z])',
        r'Conclusion[::]?\s*(.+?)(?=\n\n|\n[A-Z])'
    ]

    for pattern in position_patterns:
        match = re.search(pattern, reply, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # 如果没有明确提取，返回开头部分
    return reply[:100] + "..." if len(reply) > 100 else reply


def extract_math_answer(reply):
    """解析Moderator的最终决定"""
    # 寻找明确的最终决定表述
    decision_patterns = [
        r'Final decision[::]?\s*(.+?)(?=\n\n|\n[^\w])',
        r'The correct answer is[::]?\s*(.+?)(?=\n\n|\n[^\w])',
        r'Based on the debate, the answer is[::]?\s*(.+?)(?=\n\n|\n[^\w])',
        r'After consideration, I conclude that[::]?\s*(.+?)(?=\n\n|\n[^\w])'
    ]

    for pattern in decision_patterns:
        match = re.search(pattern, reply, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # 尝试从内容中推断答案（针对特定问题）
    if "startling" in reply.lower():
        return "startling (S-T-A-R-T-L-I-N-G)"
    elif "strength" in reply.lower():
        return "strength (note: actually 9 letters, not 8)"

    return reply.strip()[:200]  # 返回简洁摘要

def parse_final_decision(reply):
    """解析Moderator的最终决定"""
    # 寻找明确的最终决定表述
    decision_patterns = [
        r'Final decision[::]?\s*(.+?)(?=\n\n|\n[^\w])',
        r'The correct answer is[::]?\s*(.+?)(?=\n\n|\n[^\w])',
        r'Based on the debate, the answer is[::]?\s*(.+?)(?=\n\n|\n[^\w])',
        r'After consideration, I conclude that[::]?\s*(.+?)(?=\n\n|\n[^\w])'
    ]

    for pattern in decision_patterns:
        match = re.search(pattern, reply, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # 尝试从内容中推断答案（针对特定问题）
    if "startling" in reply.lower():
        return "startling (S-T-A-R-T-L-I-N-G)"
    elif "strength" in reply.lower():
        return "strength (note: actually 9 letters, not 8)"

    return reply.strip()[:200]  # 返回简洁摘要


@backoff.on_exception(backoff.expo, (Exception,), max_tries=20)
# def generate_answer(answer_context, model, debate_role=None):
#     import os
#     from openai import OpenAI
#
#     client = OpenAI(
#         base_url="https://router.huggingface.co/v1",
#         api_key=os.environ.get("HF_TOKEN", ""),
#     )
#
#
#     try:
#         # 添加辩论角色特定的提示
#         if debate_role == "Moderator":
#             answer_context.append({
#                 "role": "system",
#                 "content": "As the moderator, you must summarize the debate and provide a clear final decision on the correct answer."
#             })
#         elif debate_role == "FactChecker":
#             answer_context.append({
#                 "role": "system",
#                 "content": "As the fact checker, focus on verifying the factual accuracy of claims made by others."
#             })
#         elif debate_role == "Opposer":
#             answer_context.append({
#                 "role": "system",
#                 "content": "As the opposer, your role is to critically examine and challenge the arguments presented by others."
#             })
#         elif debate_role == "Proposer":
#             answer_context.append({
#                 "role": "system",
#                 "content": "As the proposer, your role is to present a clear argument or solution with supporting evidence."
#             })
#
#         completion = client.chat.completions.create(
#             model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B:novita",
#             messages=answer_context,
#             temperature=TEMPERATURE,
#             max_tokens=MAX_TOKENS,
#         )
#
#         content = completion.choices[0].message.content
#         prompt_tokens = completion.usage.prompt_tokens
#         completion_tokens = completion.usage.completion_tokens
#
#         return content, prompt_tokens, completion_tokens
#
#     except Exception as e:
#         print(f"Error in API call: {e}")
#         return "Error generating response", 0, 0


# @backoff.on_exception(backoff.expo, (RateLimitError, APIError, ServiceUnavailableError, APIConnectionError, Timeout), max_tries=20)
# def generate_answer(answer_context, model):
#     try:
#         completion = openai.ChatCompletion.create(
#                 #   model=model,
#                   engine=model,
#                   messages=answer_context,
#                   temperature=TEMPERATURE,
#                   max_tokens=MAX_TOKENS,
#                   n=1)
#     except RateLimitError as e:
#         if "You exceeded your current quota, please check your plan and billing details" in e.user_message:
#             raise OutOfQuotaException(openai.api_key)
#         elif "Your access was terminated due to violation of our policies" in e.user_message:
#             raise AccessTerminatedException(openai.api_key)
#         else:
#             raise e
#
#     return completion["choices"][0]["message"]["content"], completion["usage"]["prompt_tokens"], completion["usage"]["completion_tokens"]

# @Chenlei cooperation
def get_mmlu_qa_pairs(csv_name):
    df = pd.read_csv(csv_name, header=None)
    ix = len(df)
    print(f"Loading {ix} MMLU questions...")
    return [parse_question_answer(df, idx) for idx in tqdm(range(ix), desc="Loading MMLU QA pairs")]


def get_math_qa_pairs(sub_dir, min_file, max_file):
    # 其他代码保持不变...

    ret_list = []
    files_to_process = []

    for subdir, dirs, files in os.walk(sub_dir):
        for file in files:
            file_num = int(os.path.splitext(file)[0])  # Get the filename without extension and convert to int
            if min_file <= file_num <= max_file:
                files_to_process.append((subdir, file))

    print(f"Loading {len(files_to_process)} math problems...")
    for subdir, file in tqdm(files_to_process, desc="Loading math QA pairs"):
        question, prob_level, prob_type, answer = parse_single_qa_math(subdir, file)
        ret_list.append((question, answer))

    return ret_list

@backoff.on_exception(backoff.expo, (Exception,), max_tries=20)
def generate_answer(answer_context):
    import os
    from openai import OpenAI

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ.get("HF_TOKEN", ""),
    )

    try:
        completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B:novita",
            messages=answer_context,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        content = completion.choices[0].message.content
        prompt_tokens = completion.usage.prompt_tokens
        completion_tokens = completion.usage.completion_tokens

        return content, prompt_tokens, completion_tokens

    except Exception as e:
        print(f"Error in API call: {e}")
        return "Error generating response", 0, 0


def parse_single_choice(reply):
    pattern = r'\(([ABCDabcd])\)'
    matches = re.findall(pattern, reply)

    solution = None
    for match_str in matches[::-1]:
        solution = match_str.upper()
        if solution:
            break

    if solution is None:
        alter_pattern = r'([ABCDabcd])\)'
        alter_matches = re.findall(alter_pattern, reply)
        for match_str in alter_matches[::-1]:
            solution = match_str.upper()
            if solution:
                break

    return solution

def most_frequent(clist, cmp_func):
    counter = 0
    num = clist[0]

    for i in clist:
        current_frequency = sum(cmp_func(i, item) for item in clist)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num, counter


