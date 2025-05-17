import numpy as np

class Arbi:
    #初始化需要的参数为threshold,alpha, beta
    def __init__(self, threshold, alpha, beta):
        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta
    # def get_candidates(self, sorted_clusters, threshold):
    #     labels = []
    #     if not sorted_clusters:
    #         return [], []
    #
    #     top_sim, top_label = sorted_clusters[0]
    #     labels.append(top_label)
    #
    #     # 动态阈值筛选
    #     while len(sorted_clusters) >= 2:
    #         # 计算当前A（剩余最高与次高差）
    #
    #         top_sim = sorted_clusters[0][0]
    #         second_sim = sorted_clusters[1][0]
    #         A = top_sim - second_sim
    #
    #         # 计算当前B（次高与第三高差）
    #         B = 0
    #         if len(sorted_clusters) >= 3:
    #             third_sim = sorted_clusters[2][0]
    #             B = second_sim - third_sim
    #
    #         # 阈值判断
    #         if A*threshold< B:
    #             # 添加并移除当前最高
    #             current_sim, current_label = sorted_clusters[1]
    #             # current_sim, current_label = sorted_clusters.pop(0)
    #             sorted_clusters.pop(0)
    #             labels.append(current_label)
    #         else:
    #             break
    #
    #     return labels

    import numpy as np

    def get_candidates_zscore(self,sorted_clusters, z_threshold=0.5, max_candidates=3):
        """
        sorted_clusters: List of (similarity, label), sorted in descending similarity
        z_threshold: threshold for z-score (default 0.5)
        max_candidates: upper bound of selected candidates
        """

        if not sorted_clusters:
            return []

        sims = np.array([sim for sim, _ in sorted_clusters])
        labels = [label for _, label in sorted_clusters]

        mean = sims.mean()
        std = sims.std()

        if std == 0:  # all similarity scores are the same
            return labels[:max_candidates]

        z_scores = (sims - mean) / std

        candidates = []
        for z, label in zip(z_scores, labels):
            if z >= z_threshold:
                candidates.append(label)
            else:
                break  # 因为是排好序的，一旦不满足就可以提前终止

            if len(candidates) >= max_candidates:
                break

        return candidates

    # def getMaxScoreIndex(self, edits_mini_similarities, hypo_question_similarities):
    #     scores = []
    #     # 遍历每个edit_similarity元素
    #     for i in range(len(edits_mini_similarities)):
    #         e = edits_mini_similarities[i]
    #         # 获取当前edit对应的3个hypo元素起始位置
    #         start_idx = i * 3
    #         # 提取当前edit对应的3个hypo元素
    #         hypo_triplet = hypo_question_similarities[start_idx: start_idx + 3]
    #         # 计算当前edit与3个hypo的分数
    #         for h in hypo_triplet:
    #             scores.append(self.alpha * e + self.beta * h)
    #     # 找到全局最大值的索引
    #     max_index = scores.index(max(scores))/3
    #
    #     return max_index


    def get_max_score_index(self,edits_mini_similarities, hypo_question_similarities,
                            agg_method='softmax'):
        """
        计算每个 edit 与其对应 3 个 hypo 的融合得分，返回得分最高的 edit 索引。

        Parameters:
            edits_mini_similarities: List[float], 长度为 E
            hypo_question_similarities: List[float], 长度为 3E
            agg_method: str, 可选 ['max', 'mean', 'softmax']

        Returns:
            max_index: int, 最佳编辑项索引
        """



        E = len(edits_mini_similarities)
        assert len(hypo_question_similarities) == 3 * E, "Mismatch in input lengths"

        scores = []
        for i in range(E):
            e_score = edits_mini_similarities[i]
            h_triplet = hypo_question_similarities[i * 3: i * 3 + 3].cpu().numpy()


            if agg_method == 'max':
                h_score = np.max(h_triplet)
            elif agg_method == 'mean':
                h_score = np.mean(h_triplet)
            elif agg_method == 'softmax':
                weights = np.exp(h_triplet) / np.sum(np.exp(h_triplet))
                h_score = np.dot(weights, h_triplet)
            else:
                raise ValueError(f"Unknown agg_method: {agg_method}")

            total_score = self.alpha * e_score + self.beta * h_score
            scores.append(float(total_score))  # 保证是 float，不是 torch.Tensor


        max_index = int(np.argmax(scores))
        return max_index

