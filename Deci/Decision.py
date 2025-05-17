import torch
import numpy as np
from collections import defaultdict
from typing import List, Dict

import transformers


class DecisionModule:
    def __init__(
            self,
            model: torch.nn.Module,
            tokenizer: transformers.PreTrainedTokenizer,
            device: str = "cuda",
            sim_thresh: float = 0.5,
            hypoth_sim_thresh: float = 0.5,
            fusion_weights: tuple = (0.4, 0.6),
            ambiguity_thresh: float = 0.1,
            max_candidates: int = 5
    ):
        """
        完整可执行的裁决模块
        Args:
            model: 预训练编码模型
            tokenizer: 配套的分词器
            device: 计算设备
            sim_thresh: 原始相似度阈值
            hypoth_sim_thresh: 生成问题相似度阈值
            fusion_weights: 分数融合权重 (原始分, 生成分)
            ambiguity_thresh: 最大分差阈值
            max_candidates: 最大候选数量
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.sim_thresh = sim_thresh
        self.hypoth_sim_thresh = hypoth_sim_thresh
        self.fusion_weights = fusion_weights
        self.ambiguity_thresh = ambiguity_thresh
        self.max_candidates = max_candidates

    def process(
            self,
            query_emb: torch.Tensor,
            cluster_edits: List[str],
            cluster_embs: torch.Tensor,
            e2q_generator: 'E2Q'
    ) -> str:
        """
        完整处理流程
        Args:
            query_emb: 查询语句的嵌入向量 (1 x dim)
            cluster_edits: 候选簇中的edit列表
            cluster_embs: 候选edit的嵌入矩阵 (n x dim)
            e2q_generator: E2Q生成器
        Returns:
            最佳匹配的edit文本
        """
        # 阶段1：粗粒度筛选
        candidates, raw_scores = self._coarse_filter(query_emb, cluster_embs, cluster_edits)

        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        # 阶段2：精细裁决
        return self._fine_grained_judge(query_emb, candidates, raw_scores, e2q_generator)

    def _coarse_filter(
            self,
            query_emb: torch.Tensor,
            cluster_embs: torch.Tensor,
            cluster_edits: List[str]
    ) -> (List[str], List[float]):
        """粗筛候选edits"""
        # 计算余弦相似度
        similarities = torch.nn.functional.cosine_similarity(
            cluster_embs, query_emb.expand_as(cluster_embs), dim=1)

        # 双重筛选策略
        valid_idx = torch.where(similarities > self.sim_thresh)[0]
        if len(valid_idx) == 0:
            valid_idx = similarities.argsort(descending=True)[:self.max_candidates]

        # 按相似度排序
        sorted_idx = valid_idx[similarities[valid_idx].argsort(descending=True)]
        return (
            [cluster_edits[i] for i in sorted_idx[:self.max_candidates]],
            [similarities[i].item() for i in sorted_idx[:self.max_candidates]]
        )

    def _fine_grained_judge(
            self,
            query_emb: torch.Tensor,
            candidates: List[str],
            raw_scores: List[float],
            e2q_generator: 'E2Q'
    ) -> str:
        """精细裁决"""
        # 生成假设问题
        hypoth_queries = []
        for edit in candidates:
            hypoth_queries.extend(e2q_generator.generate(edit))

        # 编码假设问题
        inputs = self.tokenizer(
            hypoth_queries,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            hypoth_embs = self.model(**inputs).last_hidden_state[:, 0]

        # 计算相似度
        hypoth_sims = torch.nn.functional.cosine_similarity(
            hypoth_embs, query_emb.expand_as(hypoth_embs), dim=1)

        # 分数聚合
        score_map = defaultdict(list)
        for i, edit in enumerate([e for e in candidates for _ in range(3)]):
            score_map[edit].append(hypoth_sims[i].item())

        # 分数融合
        fused_scores = []
        for edit, raw_score in zip(candidates, raw_scores):
            hypoth_max = max(score_map[edit]) if score_map[edit] else 0
            fused = (
                    self.fusion_weights[0] * raw_score +
                    self.fusion_weights[1] * hypoth_max
            )
            fused_scores.append((edit, fused))

        # 最终裁决
        sorted_scores = sorted(fused_scores, key=lambda x: x[1], reverse=True)
        if len(sorted_scores) == 1:
            return sorted_scores[0][0]

        if (sorted_scores[0][1] - sorted_scores[1][1]) > self.ambiguity_thresh:
            return sorted_scores[0][0]

        return None


class E2Q:
    """模拟的E2Q生成器"""

    def __init__(self, device: str = "cuda"):
        self.device = device

    def generate(self, edit: str) -> List[str]:
        """生成3个假设问题（示例实现）"""
        return [
            f"What is {edit}?",
            f"Tell me about {edit.split()[0]}",
            f"How does {edit} work?"
        ]


# 测试用例
if __name__ == "__main__":
    # 初始化组件
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 示例模型和分词器
    model = transformers.AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2").to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

    # 创建测试数据
    test_edits = [
        "The capital of France is Paris",
        "Einstein developed the theory of relativity",
        "Water boils at 100 degrees Celsius"
    ]

    # 生成测试embedding
    with torch.no_grad():
        inputs = tokenizer(test_edits, padding=True, truncation=True,
                           max_length=256, return_tensors="pt").to(device)
        test_embs = model(**inputs).last_hidden_state[:, 0]

    # 创建裁决模块
    decision_module = DecisionModule(
        model=model,
        tokenizer=tokenizer,
        device=device,
        sim_thresh=0.5,
        fusion_weights=(0.4, 0.6)
    )

    # 创建测试query
    test_query = "What's the boiling point of water?"
    query_inputs = tokenizer(test_query, return_tensors="pt").to(device)
    with torch.no_grad():
        query_emb = model(**query_inputs).last_hidden_state[:, 0]

    # 执行裁决
    result = decision_module.process(
        query_emb=query_emb,
        cluster_edits=test_edits,
        cluster_embs=test_embs,
        e2q_generator=E2Q()
    )

    print(f"Input query: {test_query}")
    print(f"Matched edit: {result}")