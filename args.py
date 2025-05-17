import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='Alex Args')
    parser.add_argument('--retraining', action='store_true')
    parser.add_argument('--backbone_model', dest='backbone_model_name', type=str, default="/mnt/sda2/wmh/Models/all-mpnet-base-questions-clustering-en/",
                        help='model')
    parser.add_argument('--model_name', dest='model_name', type=str, default="best_model",
                        help='model')
    parser.add_argument('--dataset', dest='dataset_name', type=str, default="MQuAKE-CF-3k",
                        help='dataset name')

    parser.add_argument('--edited-num', dest='edited_num', type=int, default=1,
                        help='Number of edited instances')

    return parser.parse_args()
