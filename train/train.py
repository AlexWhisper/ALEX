import numpy as np
import torch
import torch.optim as optim
import transformers
from editDataset import EditDataset
from arguments import arg_parse
import json
import logging
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans  # 新增聚类模块
from torch.utils.data import Dataset, DataLoader

gpu_index = 0
device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(level=logging.INFO)
args=arg_parse()
'''
相比v1添加了聚类感知
'''
def cosine_similarity(a, b, eps=1e-8):
    """计算余弦相似度（带防除零机制）"""
    dot = torch.sum(a * b, dim=-1)
    norm_a = torch.norm(a, dim=-1)
    norm_b = torch.norm(b, dim=-1)
    return dot / (norm_a * norm_b + eps)


def batch_negative_bce_loss(edits, questions, nos, temperature=0.1, negative_number=20):
    """对比损失：正样本与负样本的相似度对抗"""
    batch_size = len(edits)
    nos_np = np.array(nos)
    loss = 0.0

    for i in range(batch_size):
        # 正样本相似度
        pos_sim = cosine_similarity(edits[i].unsqueeze(0), questions[i].unsqueeze(0))

        # 负样本选择：排除同簇样本
        neg_mask = (nos_np != nos_np[i])
        neg_indices = np.random.choice(np.where(neg_mask)[0], size=negative_number, replace=False)
        neg_sim = cosine_similarity(edits[i].unsqueeze(0), questions[neg_indices])

        # 计算交叉熵损失
        logits = torch.cat([pos_sim, neg_sim]).unsqueeze(0)
        labels = torch.zeros(1, dtype=torch.long).to(device)
        loss += torch.nn.functional.cross_entropy(logits / temperature, labels)

    return loss / batch_size


def compute_intra_cluster_similarity(edits_output, questions_output, nos, lambda_intra=0.2):
    """修正后的簇内相似度损失（基于真实聚类）"""
    unique_nos = torch.unique(nos)
    intra_loss = 0.0
    count = 0

    for no in unique_nos:
        mask = (nos == no)
        group_edits = edits_output[mask]
        group_questions = questions_output[mask]
        if len(group_edits) < 1 or len(group_questions) < 2:
            continue

        # 使用簇内所有编辑的均值作为代表
        edit_emb = group_edits.mean(dim=0).unsqueeze(0)
        sim = cosine_similarity(group_questions, edit_emb).mean()
        intra_loss += sim
        count += 1

    if count > 0:
        intra_loss = -intra_loss / count
        return lambda_intra * intra_loss
    return 0.0


def preprocess_and_cluster_edits(dataset, n_clusters=100):
    """预训练编辑嵌入并执行聚类"""
    # 1. 加载预训练模型生成编辑嵌入
    model_name = "/mnt/sda2/wmh/Models/all-mpnet-base-questions-clustering-en/"
    # model_name = "/mnt/sda2/wmh/Models/bert-base-uncased/"

    model = transformers.AutoModel.from_pretrained(model_name).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    # 提取所有唯一编辑
    unique_edits = list({item["edit"]: None for item in dataset}.keys())

    # 生成嵌入
    edit_embeddings = []
    with torch.no_grad():
        for edit in unique_edits:
            inputs = tokenizer(edit, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            emb = model(**inputs).last_hidden_state[:, 0].cpu().numpy()
            edit_embeddings.append(emb)

    # 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(np.concatenate(edit_embeddings, axis=0))

    # 映射回原始数据集
    edit_to_cluster = {edit: label for edit, label in zip(unique_edits, cluster_labels)}
    for item in dataset:
        item["cluster_no"] = int(edit_to_cluster[item["edit"]])

    return dataset


def train_model(n_clusters=50,lambda_intra=0.1):

    backbone_name = args.backbone
    save_name = args.model_name
    # 1. 数据预处理与聚类
    with open('../datasets/cls-filtered.json', 'r') as f:
        dataset = json.load(f)

    # 执行聚类（关键修正点）
    dataset = preprocess_and_cluster_edits(dataset, n_clusters)

    # 2. 划分训练集/验证集
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

    # 3. 构造数据集（使用聚类后的cluster_no）
    edit_train, questions_train, no_train = construct_dataset(train_data)
    edit_test, questions_test, no_test = construct_dataset(test_data)

    model = transformers.AutoModel.from_pretrained(backbone_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(backbone_name)

    # 5. 数据加载
    train_loader = DataLoader(EditDataset(edit_train, questions_train, no_train), batch_size=256, shuffle=True)
    val_loader = DataLoader(EditDataset(edit_test, questions_test, no_test), batch_size=512, shuffle=False)

    # 6. 训练配置
    optimizer = optim.Adam(model.parameters(), lr=2e-5, weight_decay=0.01)
    model.to(device)
      # 调整后的权重

    best_ydi = 0.0
    early_stop_counter = 0

    ydi_smooth= 0.0

    # 7. 训练循环
    for epoch in range(1000):
        model.train()
        total_loss = 0.0

        for edits, questions, nos in train_loader:
            # 编码
            edits_input = tokenizer(edits, padding=True, truncation=True, max_length=256, return_tensors='pt').to(
                device)
            questions_input = tokenizer(questions, padding=True, truncation=True, max_length=256,
                                        return_tensors='pt').to(device)

            # 获取嵌入
            edits_emb = model(**edits_input).last_hidden_state[:, 0]
            questions_emb = model(**questions_input).last_hidden_state[:, 0]

            # 损失计算
            contrastive_loss = batch_negative_bce_loss(edits_emb, questions_emb, nos)
            intra_loss = compute_intra_cluster_similarity(edits_emb, questions_emb, nos, lambda_intra)
            loss = contrastive_loss + intra_loss

            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item() * len(edits)

        print(f'Epoch {epoch}: total_loss = {total_loss}')

        # 验证步骤
        model.eval()
        recall_total, block_total, acc_total = 0, 0, 0
        with torch.no_grad():
            for edits, questions, nos in val_loader:
                edits_emb = model(
                    **tokenizer(edits, padding=True, truncation=True, max_length=256, return_tensors='pt').to(
                        device)).last_hidden_state[:, 0]
                questions_emb = model(
                    **tokenizer(questions, padding=True, truncation=True, max_length=256, return_tensors='pt').to(
                        device)).last_hidden_state[:, 0]

                acc_total += (cosine_similarity(edits_emb, questions_emb) >= 0).sum().item()
                recall, block = retrieval_metric(edits_emb, questions_emb, nos)
                recall_total += recall
                block_total += block

        ydi = (recall_total + block_total) / len(val_loader.dataset)
        print(f'Validation YDI = {ydi}')





        # 早停与保存
        if ydi > best_ydi:
            best_ydi = ydi
            early_stop_counter = 0
            model.save_pretrained(f"detector-checkpoint/{save_name}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= 5:
                print("Early stopping triggered.")
                break




def construct_dataset(input_data):
    """使用聚类后的cluster_no构造数据集"""
    input_edit = []
    input_questions = []
    input_no = []
    for item in input_data:
        for question in item["questions"]:
            input_edit.append(item["edit"])
            input_questions.append(question)
            input_no.append(item["cluster_no"])  # 使用聚类后的编号
    return input_edit, input_questions, input_no


def retrieval_metric(edits, questions, nos):
    """检索评估指标"""
    instance_num = len(questions)
    nos = np.array(nos)
    retrieval_num, block_num = 0, 0

    for i in range(instance_num):
        idxs = nos != nos[i]
        sim = cosine_similarity(edits, questions[i].unsqueeze(0))

        if sim[i] > sim[idxs].max():
            retrieval_num += 1
        if sim[idxs].max() < 0:
            block_num += 1

    return retrieval_num, block_num



if __name__ == '__main__':
    train_model()


