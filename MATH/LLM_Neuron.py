import random
import re
from utils import *
from prompt_lib import ROLE_MAP, construct_ranking_message, construct_message, SYSTEM_PROMPT_MMLU, ROLE_MAP_MATH, SYSTEM_PROMPT_MATH, ROLE_MAP_DEBATE, SYSTEM_PROMPT_DEBATE
from tqdm import tqdm
from model_selection1 import MarkovModelSelector

#这里把 generate 改成了 以及selector.generate
#以及selector = MarkovModelSelector() 和 可视化分析结果
selector = MarkovModelSelector()

class LLMNeuron:
    
    def __init__(self, role, mtype="local-deepseek", ans_parser=parse_single_choice, qtype="single_choice"):
        self.role = role
        self.model = mtype
        self.qtype = qtype
        self.ans_parser = ans_parser
        self.reply = None
        self.answer = ""
        self.active = False
        self.importance = 0
        self.to_edges = []
        self.from_edges = []
        self.question = None


        def find_array(text):
            # Find all matches of array pattern
            matches = re.findall(r'\[\[(.*?)\]\]', text)
            if matches:
                # Take the last match and remove spaces
                last_match = matches[-1].replace(' ', '')
                def convert(x):
                    try:
                        return int(x)
                    except:
                        return 0
                # Convert the string to a list of integers
                try:
                    ret = list(map(convert, last_match.split(',')))
                except:
                    ret = []
                return ret
            else:
                return []
        self.weights_parser = find_array

        self.prompt_tokens = 0
        self.completion_tokens = 0

    def get_reply(self):
        return self.reply

    def get_answer(self):
        return self.answer

    def deactivate(self):
        self.active = False
        self.reply = None
        self.answer = ""
        self.question = None
        self.importance = 0

        self.prompt_tokens = 0
        self.completion_tokens = 0

    def activate(self, question):
        self.question = question
        self.active = True
        # get context and genrate reply
        contexts, formers = self.get_context()
        # shuffle
        original_idxs = [mess[1] for mess in formers]
        random.shuffle(formers)
        shuffled_idxs = [mess[1] for mess in formers]
        formers = [mess[0] for mess in formers]


        contexts.append(construct_message(formers, question, self.qtype))
        # 添加进度显示
        with tqdm(total=1, desc=f"Activating {self.role}", leave=False) as pbar:
            self.reply, self.prompt_tokens, self.completion_tokens = selector.generate_answer(contexts, self.model)
            pbar.update(1)
        # self.reply, self.prompt_tokens, self.completion_tokens = generate_answer(contexts, self.model)


        # parse answer
        self.answer = self.ans_parser(self.reply)
        weights = self.weights_parser(self.reply)
        if len(weights) != len(formers):
            weights = [0 for _ in range(len(formers))]

        shuffled_pairs = list(zip(shuffled_idxs, weights, formers))
        sorted_pairs = sorted(shuffled_pairs, key=lambda x: original_idxs.index(x[0]))
        weights, formers = [weight for _, weight, _ in sorted_pairs], [(former, eid) for eid, _, former in sorted_pairs]

        lp = 0
        for _, eid in formers:
            self.from_edges[eid].weight = weights[lp] / 5 if 0 < weights[lp] <= 5 else (1 if weights[lp] > 5 else 0)
            lp += 1
        # normalize weights
        total = sum([self.from_edges[eid].weight for _, eid in formers])
        if total > 0:
            for _, eid in formers:
                self.from_edges[eid].weight /= total
        else:
            for _, eid in formers:
                self.from_edges[eid].weight = 1 / len(formers)

        a = selector.compute_gradient_angle_matrix


    # # @Chenlei
    # def activate(self, question):
    #     self.question = question
    #     self.active = True
    #     # 获取上下文并生成回复
    #     contexts, formers = self.get_context()
    #
    #     # 添加辩论角色信息到上下文
    #     contexts.append(construct_message(formers, question, self.qtype, debate_role=self.role))
    #
    #     selector = MarkovModelSelector()
    #     # 生成回复
    #     with tqdm(total=1, desc=f"Activating {self.role}", leave=False) as pbar:
    #         self.reply, self.prompt_tokens, self.completion_tokens = selector.generate_answer(contexts, self.model,
    #                                                                                  debate_role=self.role)
    #         pbar.update(1)
    #
    #     # 根据角色解析答案
    #     if self.role == "Moderator":
    #         self.answer = parse_final_decision(self.reply)
    #     else:
    #         self.answer = parse_debate_reply(self.reply)
    #
    #     # 如果没有解析出答案，使用默认解析器
    #     if not self.answer:
    #         if self.qtype == "single_choice":
    #             self.answer = parse_single_choice(self.reply)
    #         elif self.qtype == "math_exp":
    #             self.answer = extract_math_answer(self.reply)
    #         else:
    #             self.answer = self.reply[:200]  # 截取前200字符作为答案
    #
    #     # 解析权重（如果适用）
    #     weights = self.weights_parser(self.reply)
    #     if len(weights) != len(formers):
    #         weights = [0 for _ in range(len(formers))]
    #
    #     # 重新排序权重以匹配原始顺序
    #     original_idxs = [mess[1] for mess in formers]
    #     shuffled_idxs = [mess[1] for mess in formers]
    #     shuffled_pairs = list(zip(shuffled_idxs, weights, formers))
    #     sorted_pairs = sorted(shuffled_pairs, key=lambda x: original_idxs.index(x[0]))
    #     weights, formers = [weight for _, weight, _ in sorted_pairs], [(former, eid) for eid, _, former in sorted_pairs]
    #
    #     # 设置边权重
    #     lp = 0
    #     for _, eid in formers:
    #         self.from_edges[eid].weight = weights[lp] / 5 if 0 < weights[lp] <= 5 else (1 if weights[lp] > 5 else 0)
    #         lp += 1
    #
    #     # 归一化权重
    #     total = sum([self.from_edges[eid].weight for _, eid in formers])
    #     if total > 0:
    #         for _, eid in formers:
    #             self.from_edges[eid].weight /= total
    #     else:
    #         for _, eid in formers:
    #             self.from_edges[eid].weight = 1 / len(formers)
    #
    #     # 可视化分析结果
    #     a = selector.compute_gradient_angle_matrix


    # def get_context(self):
    #     if self.qtype == "single_choice":
    #         sys_prompt = ROLE_MAP[self.role] + "\n" + SYSTEM_PROMPT_MMLU
    #     elif self.qtype == "math_exp":
    #         sys_prompt = ROLE_MAP_MATH[self.role] + "\n" + SYSTEM_PROMPT_MATH
    #     elif self.qtype == "open-ended":
    #         sys_prompt = ROLE_MAP[self.role] + "\n"
    #     else:
    #         raise NotImplementedError("Error init question type")
    #     contexts = [{"role": "system", "content": sys_prompt}]
    #
    #     formers = [(edge.a1.reply, eid) for eid, edge in enumerate(self.from_edges) if edge.a1.reply is not None and edge.a1.active]
    #     return contexts, formers

    # @Chenlei
    def get_context(self):
        """根据角色和问题类型获取上下文"""
        if self.role in ["Proposer", "Opposer", "FactChecker", "Moderator"]:
            # 辩论角色使用辩论特定的提示
            if self.qtype == "single_choice":
                sys_prompt = ROLE_MAP_DEBATE[self.role] + "\n" + SYSTEM_PROMPT_DEBATE
            elif self.qtype == "math_exp":
                sys_prompt = ROLE_MAP_DEBATE[self.role] + "\n" + SYSTEM_PROMPT_DEBATE
            elif self.qtype == "open-ended":
                sys_prompt = ROLE_MAP_DEBATE[self.role] + "\n" + SYSTEM_PROMPT_DEBATE
            else:
                raise NotImplementedError("Error init question type")
        else:
            # 普通角色使用原有提示
            if self.qtype == "single_choice":
                sys_prompt = ROLE_MAP[self.role] + "\n" + SYSTEM_PROMPT_MMLU
            elif self.qtype == "math_exp":
                sys_prompt = ROLE_MAP_MATH[self.role] + "\n" + SYSTEM_PROMPT_MATH
            elif self.qtype == "open-ended":
                sys_prompt = ROLE_MAP[self.role] + "\n"
            else:
                raise NotImplementedError("Error init question type")

        contexts = [{"role": "system", "content": sys_prompt}]

        formers = [(edge.a1.reply, eid) for eid, edge in enumerate(self.from_edges)
                   if edge.a1.reply is not None and edge.a1.active]
        return contexts, formers
        
    def get_conversation(self):
        if not self.active:
            return []

        contexts, formers = self.get_context()
        contexts.append(construct_message([mess[0] for mess in formers], self.question, self.qtype))
        contexts.append({"role": "assistant", "content": self.reply})
        return contexts


class LLMEdge:

    def __init__(self, a1, a2):
        self.weight = 0
        self.a1 = a1
        self.a2 = a2
        self.a1.to_edges.append(self)
        self.a2.from_edges.append(self)

    def zero_weight(self):
        self.weight = 0

def parse_ranks(completion, max_num=4):
    content = completion
    pattern = r'\[([1234567]),\s*([1234567])\]'
    matches = re.findall(pattern, content)

    try:
        match = matches[-1]
        tops = [int(match[0])-1, int(match[1])-1]
        def clip(x):
            if x < 0:
                return 0
            if x > max_num-1:
                return max_num-1
            return x
        tops = [clip(x) for x in tops]
    except:
        print("error in parsing ranks")
        tops = random.sample(list(range(max_num)), 2)

    return tops

def listwise_ranker_2(responses, question, qtype, model="local-deepseek"):
    assert 2 < len(responses)# <= 4
    message = construct_ranking_message(responses, question, qtype)
    # 添加进度显示
    with tqdm(total=1, desc="Ranking responses", leave=False) as pbar:
        completion, prompt_tokens, completion_tokens = selector.generate_answer([message], model)
        pbar.update(1)

    # completion, prompt_tokens, completion_tokens = generate_answer([message], model)
    return parse_ranks(completion, max_num=len(responses)), prompt_tokens, completion_tokens


# 在文件末尾添加以下代码
try:
    from cooperation_competition_visualizer import CooperationCompetitionVisualizer

    COOP_COMP_VIS_AVAILABLE = True
    print("成功加载合作/竞争可视化模块")
except ImportError:
    COOP_COMP_VIS_AVAILABLE = False
    print("警告: 未找到cooperation_competition_visualizer模块，无法生成合作/竞争可视化")


def visualize_debate_interaction(role_importance, num_rounds=3, output_dir="debate_visualization"):
    """
    可视化辩论交互和角色对称性

    参数:
    role_importance -- 各角色的重要性分数
    num_rounds -- 辩论轮次
    output_dir -- 输出目录

    返回:
    分析报告
    """
    if not COOP_COMP_VIS_AVAILABLE:
        print("无法生成可视化: 缺少依赖模块")
        return None

    try:
        # 创建辩论结果结构
        debate_results = {
            "role_importance": role_importance,
            "num_rounds": num_rounds
        }

        # 生成可视化
        visualizer = CooperationCompetitionVisualizer()
        report = visualizer.analyze_and_visualize(debate_results, output_dir)
        return report
    except Exception as e:
        print(f"生成可视化时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# 在DebateLLM类中添加方法 (如果尚未存在)
if 'DebateLLM' in globals():
    def analyze_and_visualize_interaction(self, result, output_dir="debate_visualization"):
        """分析辩论中的交互类型并生成可视化"""
        # 获取角色重要性
        role_importance = {}
        for idx, role in enumerate(self.agent_roles):
            importance = 0
            for rid in range(self.rounds):
                node_idx = idx + rid * self.agents
                if node_idx < len(self.nodes) and self.nodes[node_idx].active:
                    importance += self.nodes[node_idx].importance
            role_importance[role] = importance

        # 生成可视化
        return visualize_debate_interaction(role_importance, self.rounds, output_dir)


    # 将方法添加到DebateLLM类
    DebateLLM.analyze_and_visualize_interaction = analyze_and_visualize_interaction