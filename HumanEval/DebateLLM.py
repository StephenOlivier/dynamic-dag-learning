import math
import random
from LLM_Neuron import LLMNeuron, LLMEdge, listwise_ranker_2
from utils1 import parse_single_choice, most_frequent, is_equiv, extract_math_answer, parse_final_decision
from sacrebleu import sentence_bleu
from prompt_lib1 import GEN_THRESHOLD, ROLE_MAP_DEBATE, SYSTEM_PROMPT_DEBATE

ACTIVATION_MAP = {'listwise': 0, 'trueskill': 1, 'window': 2, 'none': -1}


class DebateLLM:

    def __init__(self, default_model_name, agents=4, agent_roles=[],
                 rounds=5, debate_type="structured", qtype="single_choice", mtype="gpt-3.5-turbo"):
        self.default_model_name = default_model_name
        self.agents = agents
        self.rounds = rounds
        self.debate_type = debate_type
        self.mtype = mtype

        assert len(agent_roles) == agents and agents > 0
        self.agent_roles = agent_roles
        self.qtype = qtype
        if qtype == "single_choice":
            self.cmp_res = lambda x, y: x == y
            self.ans_parser = parse_single_choice
        elif qtype == "math_exp":
            self.cmp_res = is_equiv
            self.ans_parser = extract_math_answer
        elif qtype == "open-ended":
            self.cmp_res = lambda x, y: sentence_bleu(x, [y], lowercase=True).score >= GEN_THRESHOLD * 100
            self.ans_parser = lambda x: x
        else:
            raise NotImplementedError("Error qtype")

        self.init_nn(self.debate_type, self.agent_roles)

    def init_nn(self, debate_type, agent_roles):
        self.nodes, self.edges = [], []
        for idx in range(self.agents):
            self.nodes.append(LLMNeuron(agent_roles[idx], self.mtype, self.ans_parser, self.qtype))

        agents_last_round = self.nodes[:self.agents]
        for rid in range(1, self.rounds):
            for idx in range(self.agents):
                self.nodes.append(LLMNeuron(agent_roles[idx], self.mtype, self.ans_parser, self.qtype))
            agents_last_round = self.nodes[-self.agents:]

        # 设置辩论结构
        self.setup_debate_flow(debate_type)

    def setup_debate_flow(self, debate_type):
        """设置辩论流程的通信结构"""
        self.edges = []

        if debate_type == "structured":
            # 结构化辩论: Proposer -> Opposer -> FactChecker -> Moderator -> Proposer (循环)
            for rid in range(self.rounds):
                for idx in range(self.agents):
                    current_node_idx = self.agents * rid + idx

                    # Proposer (0) -> Opposer (1)
                    # Opposer (1) -> FactChecker (2)
                    # FactChecker (2) -> Moderator (3)
                    if idx < self.agents - 1:  # Not the last role
                        next_node_idx = self.agents * rid + (idx + 1)
                        self.edges.append(LLMEdge(self.nodes[current_node_idx], self.nodes[next_node_idx]))
                    else:  # Last role (Moderator)
                        if rid < self.rounds - 1:  # Not the last round
                            next_node_idx = self.agents * (rid + 1) + 0  # Next round, first role (Proposer)
                            self.edges.append(LLMEdge(self.nodes[current_node_idx], self.nodes[next_node_idx]))

    def zero_grad(self):
        for edge in self.edges:
            edge.zero_weight()
        for node in self.nodes:
            node.importance = 0

    def set_allnodes_deactivated(self):
        for node in self.nodes:
            node.deactivate()

    def forward(self, question):
        resp_cnt = 0
        total_prompt_tokens, total_completion_tokens = 0, 0
        self.set_allnodes_deactivated()

        # 按辩论流程激活节点
        # 第一轮: Proposer -> Opposer -> FactChecker -> Moderator
        for idx in range(self.agents):
            node_idx = idx  # 第0轮
            if idx == 0 or (idx > 0 and self.nodes[node_idx - 1].active):
                self.nodes[node_idx].activate(question)
                resp_cnt += 1
                total_prompt_tokens += self.nodes[node_idx].prompt_tokens
                total_completion_tokens += self.nodes[node_idx].completion_tokens

        # 后续轮次
        for rid in range(1, self.rounds):
            # Moderator -> Proposer (新轮次)
            if self.nodes[self.agents * (rid - 1) + (self.agents - 1)].active:  # 前一轮的Moderator已激活
                self.nodes[self.agents * rid].activate(question)  # 新轮次的Proposer
                resp_cnt += 1
                total_prompt_tokens += self.nodes[self.agents * rid].prompt_tokens
                total_completion_tokens += self.nodes[self.agents * rid].completion_tokens

                # Proposer -> Opposer -> FactChecker -> Moderator
                for idx in range(1, self.agents):
                    if self.nodes[self.agents * rid + (idx - 1)].active:
                        self.nodes[self.agents * rid + idx].activate(question)
                        resp_cnt += 1
                        total_prompt_tokens += self.nodes[self.agents * rid + idx].prompt_tokens
                        total_completion_tokens += self.nodes[self.agents * rid + idx].completion_tokens

        # 获取最终答案 (由最后一轮的Moderator提供)
        moderator_idx = self.agents * (self.rounds - 1) + (self.agents - 1)
        final_answer = self.nodes[moderator_idx].get_answer()
        if not final_answer:
            # 如果Moderator没有提供明确答案，尝试解析
            final_answer = parse_final_decision(self.nodes[moderator_idx].get_reply())

        def get_completions():
            # 获取所有完成内容
            completions = [[] for _ in range(self.agents)]
            for rid in range(self.rounds):
                for idx in range(self.agents):
                    node_idx = self.agents * rid + idx
                    if self.nodes[node_idx].active:
                        completions[idx].append(self.nodes[node_idx].get_reply())
                    else:
                        completions[idx].append(None)
            return completions

        return final_answer, resp_cnt, get_completions(), total_prompt_tokens, total_completion_tokens

    def backward(self, result):
        """计算节点重要性分数，从最终决策反向传播"""
        # 初始化所有节点重要性为0
        for node in self.nodes:
            node.importance = 0

        # 设置最后一轮Moderator的重要性为1.0
        moderator_idx = self.agents * (self.rounds - 1) + (self.agents - 1)
        if self.nodes[moderator_idx].active:
            self.nodes[moderator_idx].importance = 1.0

        # 反向传播重要性
        for rid in range(self.rounds - 1, -1, -1):
            for idx in range(self.agents - 1, -1, -1):
                node_idx = self.agents * rid + idx
                current_node = self.nodes[node_idx]

                if current_node.importance == 0:  # 仅处理未设置重要性的节点
                    continue

                # 查找影响当前节点的前驱节点
                for edge in current_node.from_edges:
                    if edge.a1.active:
                        # 基于边权重分配重要性
                        edge.a1.importance += current_node.importance * edge.weight

        return [node.importance for node in self.nodes]