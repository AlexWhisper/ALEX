from arguments import arg_parse
import torch

import eval2


args = arg_parse()

#是否重新训练
if args.retraining:
    from train.train import train_model
    train_model()


#加载模型和数据集
llm=args.llm_name
model_name=args.model_name
dataset_name=args.dataset_name

#构建E2Q缓存
if args.buildE2Q:
    from edit2Query import E2Q
    e2q = E2Q(dataset_name,args.device)
    e2q.build()

#评估

torch.cuda.empty_cache()
# dataset_name="MQuAKE-hard"
llm="deepseek"
datasets = ["MQuAKE-CF-3k-v2",
             #   "MQuAKE-CF-3k",
                #"MQuAKE-T",
                #"MQuAKE-2002"
                ]
for dataset_name in datasets:
    eval2.runAlex(llm,model_name,dataset_name,args)
