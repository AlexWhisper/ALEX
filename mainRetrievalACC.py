from arguments import arg_parse
import torch
import csv
import testRetrievalACC

args = arg_parse()

if args.retraining:
    from train.train import train_model

    train_model()

llm = "deepseek"
model_name = args.model_name
dataset_name = args.dataset_name

if args.buildE2Q:
    from edit2Query import E2Q

    e2q = E2Q(dataset_name, args.device)
    e2q.build()

datasets = [
    "MQuAKE-CF-3k-v2",
    "MQuAKE-CF-3k",
    "MQuAKE-T",
    "MQuAKE-hard",
    "MQuAKE-2002"
]

n_cluster = [ 7, 10, 12, 15, 18, 20]

# 初始化显存记录字典
memory_usage = {}

for n in n_cluster:
    args.n_clusters = n
    memory_usage[n] = {}

    for dataset_name in datasets:
        # 清理缓存并重置统计
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # 运行测试
        testRetrievalACC.runAlex(llm, model_name, dataset_name, args)

        # 获取峰值显存并转换单位
        max_mem = torch.cuda.max_memory_allocated()  # 单位：字节
        max_mem_mb = max_mem / (1024 ** 2)  # 转换为MB

        # 记录结果
        memory_usage[n][dataset_name] = max_mem_mb
        print(f"n_clusters: {n}, Dataset: {dataset_name}, Memory Usage: {max_mem_mb:.2f} MB")

# 将结果写入CSV文件
with open('memory_usage_report.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Cluster', 'Dataset', 'Memory Usage (MB)'])

    for n in n_cluster:
        for dataset in datasets:
            writer.writerow([
                n,
                dataset,
                f"{memory_usage[n][dataset]:.2f}"
            ])

print("显存使用报告已保存至 memory_usage_report.csv")