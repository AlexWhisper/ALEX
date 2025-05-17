import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='Alex Args')
    parser.add_argument('--retraining', action='store_true')
    parser.add_argument('--buildE2Q', action='store_true')
    parser.add_argument('--backbone_model', dest='backbone_model_name', type=str, default="/mnt/sda2/wmh/Models/all-mpnet-base-questions-clustering-en/",
                        help='model')
    parser.add_argument('--model_name', dest='model_name', type=str, default="./train/all-mpnet-base-questions-clustering-en",help='model')
    parser.add_argument('--llm_name', dest='llm_name', type=str, default="Llama-2-7b-chat-hf",help='llm')
    parser.add_argument('--api_key', dest='api_key', type=str, default="sk-55aab06bd7fc452e8b5202c6f3f19b29",help='api_key')
    parser.add_argument('--n_clusters', dest='n_clusters', type=int, default="12",help='model')
    parser.add_argument('--dataset', dest='dataset_name', type=str, default="MQuAKE-CF-3k-v2",help='dataset name')
    parser.add_argument('--threshold', dest='threshold', type=float, default="0.8",help='threshold')
    parser.add_argument('--alpha', dest='alpha', type=float, default="0.5",help='alpha')
    parser.add_argument('--beta', dest='beta', type=float, default="0.5",help='beta')
    parser.add_argument('--device', dest='device', type=str, default="cuda:0",help='device')


    return parser.parse_args()

