import torch
import numpy as np
from collections import defaultdict
from sklearn.cluster import SpectralClustering, KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity
from sklearn.neighbors import kneighbors_graph

class TextSentenceCluster:
    def __init__(self, edits,edit_emb, model, tokenizer,arbi,args):
        '''
        相比与v1主要是添加了谱聚类
        :param edits:
        :param model:
        :param tokenizer:
        :param device:
        '''
        self.edits = edits
        self.edits_emb=edit_emb
        self.device = args.device
        self.arbi=arbi
        self.model = model
        self.tokenizer = tokenizer
        self.args=args

        # 新增存储变量
        self.feature_mean = None
        self.feature_std = None
        self.cluster_centers = []
        self.cluster_edits = defaultdict(list)

    def _get_embeddings(self, batch_size=32):
        """生成句子的语义嵌入"""
        embeddings = []
        for i in range(0, len(self.edits), batch_size):
            batch = self.edits[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=128
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            batch_embeds = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_embeds)
        return np.concatenate(embeddings, axis=0)

    def _enhance_features(self, embeddings):
        """特征增强：仅使用文本统计特征"""
        text_features = np.array([
            [len(s), len(s.split())]  # 字符长度和单词数量
            for s in self.edits
        ])
        return np.concatenate([embeddings, text_features], axis=1)

    def _compute_affinity_matrix(self, features, affinity='rbf', n_neighbors=10):
        """计算亲和度矩阵，用于谱聚类"""
        if affinity == 'rbf':
            # 使用RBF核函数
            return rbf_kernel(features)
        elif affinity == 'nearest_neighbors':
            # 使用K近邻图
            connectivity = kneighbors_graph(
                features, n_neighbors=n_neighbors,
                include_self=False, mode='connectivity'
            )
            # 确保矩阵是对称的
            affinity_matrix = 0.5 * (connectivity + connectivity.T)
            return affinity_matrix.toarray()
        elif affinity == 'cosine':
            # 使用余弦相似度
            return cosine_similarity(features)
        else:
            raise ValueError(f"不支持的亲和度计算方法: {affinity}")

    def cluster(self, method='kmeans', **params):
        """执行聚类并保存必要信息"""
        base_embeds = self._get_embeddings()
        #base_embeds = self.edits_emb.cpu().numpy()
        enhanced_features = self._enhance_features(base_embeds)

        # 标准化并保存参数
        self.feature_mean = enhanced_features.mean(axis=0)
        self.feature_std = enhanced_features.std(axis=0)
        features = (enhanced_features - self.feature_mean) / self.feature_std

        # 执行聚类
        if method == 'kmeans':
            n_clusters = params.get('n_clusters', 10)
            clusterer = KMeans(n_clusters=n_clusters, n_init=10)
            labels = clusterer.fit_predict(features)
        elif method == 'dbscan':
            clusterer = DBSCAN(**params)
            labels = clusterer.fit_predict(features)
        elif method == 'spectral':
            n_clusters = params.get('n_clusters', 10)
            affinity = params.get('affinity', 'rbf')
            n_neighbors = params.get('n_neighbors', 10)
            assign_labels = params.get('assign_labels', 'kmeans')

            # 对于大型数据集，可以使用近似谱聚类
            if len(features) > 10000 and affinity == 'rbf':
                print("使用大规模近似谱聚类...")
                clusterer = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity='rbf',
                    n_jobs=-1,
                    assign_labels=assign_labels,
                    random_state=42
                )
                labels = clusterer.fit_predict(features)
            else:
                # 计算亲和度矩阵
                if affinity in ['rbf', 'nearest_neighbors', 'cosine']:
                    affinity_matrix = self._compute_affinity_matrix(
                        features, affinity=affinity, n_neighbors=n_neighbors
                    )
                    clusterer = SpectralClustering(
                        n_clusters=n_clusters,
                        affinity='precomputed',
                        assign_labels=assign_labels,
                        random_state=42
                    )
                    labels = clusterer.fit_predict(affinity_matrix)
                else:
                    # 直接使用scikit-learn内置的亲和度计算
                    clusterer = SpectralClustering(
                        n_clusters=n_clusters,
                        affinity=affinity,
                        assign_labels=assign_labels,
                        random_state=42
                    )
                    labels = clusterer.fit_predict(features)
        else:
            raise ValueError("Unsupported method")

        # 保存簇信息
        self.cluster_edits = defaultdict(list)
        for idx, label in enumerate(labels):
            self.cluster_edits[label].append(self.edits[idx])

        # 计算簇中心（排除噪声簇）
        self.cluster_centers = []
        unique_labels = np.unique(labels)
        for label in unique_labels:
            if label == -1:
                continue
            mask = (labels == label)
            cluster_feats = features[mask]
            if len(cluster_feats) > 0:
                center = cluster_feats.mean(axis=0)
                self.cluster_centers.append((label, center))

        # 评估
        if len(np.unique(labels)) > 1 and len(np.unique(labels)) < len(labels):
            try:
                silhouette = silhouette_score(features, labels)
                print(f"聚类评估 - Silhouette Score: {silhouette:.4f}")
            except:
                print("无法计算Silhouette Score")
        else:
            print("无法计算聚类指标（所有样本属于同一类或每个样本都是一个独立的类）")
        return labels

    def process_query(self, query):
        """处理查询并生成标准化特征"""
        # 生成基础嵌入
        inputs = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        base_embed = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        # 生成增强特征
        text_feat = np.array([[len(query), len(query.split())]])
        enhanced_feat = np.concatenate([base_embed, text_feat], axis=1)

        # 标准化
        normalized_feat = (enhanced_feat - self.feature_mean) / self.feature_std
        return normalized_feat

    def find_relevant_cluster(self, query):
        """找到最相关的簇"""
        query_feat = self.process_query(query)

        max_sim = -1
        best_cluster = None

        # 计算与每个簇中心的相似度
        for label, center in self.cluster_centers:
            sim = cosine_similarity(query_feat, center.reshape(1, -1))[0][0]
            if sim > max_sim:
                max_sim = sim
                best_cluster = label

        return self.cluster_edits.get(best_cluster, [])

    def find_relevant_cluster2(self, query):
        """找到符合动态阈值的相关簇序列"""
        query_feat = self.process_query(query)

        # 计算每个簇的相似度并降序排列
        cluster_similarities = []
        for label, center in self.cluster_centers:
            sim = cosine_similarity(query_feat, center.reshape(1, -1))[0][0]
            cluster_similarities.append((sim, label))

        sorted_clusters = sorted(cluster_similarities, key=lambda x: x[0], reverse=True)
        '''
        传入，排好序的，
        '''

        labels=self.arbi.get_candidates_zscore(sorted_clusters,self.args.threshold,2)


        # 收集所有相关簇的编辑记录
        relevant_edits = []
        for label in labels:
            relevant_edits.extend(self.cluster_edits.get(label, []))

        return relevant_edits


