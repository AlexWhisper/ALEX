import json
import time

import transformers
from tqdm import tqdm
import logging
from Deci import Arbitration
from AlexDataLoader import DataLoader
import torch
from Call_LLM import Load_LLM
from Cluster.TextSentenceCluster import TextSentenceCluster
from edit2Query import E2Q
logging.basicConfig(filename=f'./logs/3kv2_failedCase.logs', level=logging.WARNING,
                    format='%(asctime)s - %(levelname)s - %(message)s')
def runAlex(llm,model_name,dataset_name,args):
    #加载大模型
    if llm=='Llama-2-7b-chat-hf':
        llm_name="/mnt/sda2/Shared_Model/Llama-2-7b-chat-hf/"
        llmModel = transformers.AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.float16).to(args.device)
        llmtokenizer = transformers.AutoTokenizer.from_pretrained(llm_name)

    elif llm=='llama3':

        model_name='gpt-4'

    arbi=Arbitration.Arbi(args.threshold,args.alpha,args.beta)

    #加载Alex模型
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model=transformers.AutoModel.from_pretrained(model_name).to(args.device)
    #time.sleep(100000)
    #加载call_gpt方法

    call_gpt = Load_LLM.method_map[llm]
    dataloader=DataLoader(dataset_name,batch_size=1,device=args.device)
    dataset_batch=dataloader.getAllData()
    edit=dataloader.getAllEdit()
    edit_emb=dataloader.getDataEmb(model,tokenizer)

    #加载E2Q
    e2q = E2Q(dataset_name,args.device)
    print(f'开始评估{dataset_name}')

    #聚类处理
    print(f'当前聚类簇数量为：{args.n_clusters}')
    clusterer = TextSentenceCluster(edit,edit_emb, model, tokenizer,arbi,args)
    labels = clusterer.cluster(method='kmeans', n_clusters=args.n_clusters)

    qa_pairs = {}
    qa_pairs = dataloader.loadQuery()



    with open('prompts/KGprompt.txt', 'r') as f:
        KGprompt = f.read()
    cor = 0
    ver_cor = 0
    cor_list = [0, 0, 0]
    ver_cor_list = [0, 0, 0]
    tot = 0
    fail_cases = {}
    
    sizePerQuery=[]
    gen_failed_count_1=0
    gen_failed_count_2=0
    def verify_subquestion_path(prompt, hops, path):
        # Checking the correctness of reasoning path i.e. Calculating Hop-Acc
        # subquestion verification
        if not len(prompt.strip().split('Subquestion:')) == hops + 1:
            return 1

        # reasoning path verification
        sub_prompt = prompt.strip().split('Subquestion:')
        for idx in range(1, len(sub_prompt)):

            inter_ans = sub_prompt[idx].strip().split(': ')[-1]
            # print(inter_ans)
            if inter_ans != path[idx - 1]["answer"] and inter_ans not in path[idx - 1]["answer_alias"]:
                return 2

        return False

    total=0
    cluster_ACC=0
    Retrieval_ACC=0
    for query, ans in tqdm(qa_pairs.items()):
        total += 1

        # relevant_edits = clusterer.find_relevant_cluster(query)


        subquestion = query

        # 子问题嵌入
        with torch.no_grad():
            subquestion_input = tokenizer(subquestion, padding=True, truncation=True, max_length=256,
                                          return_tensors='pt').to(args.device)
            subquestion_emb = model(**subquestion_input).last_hidden_state[:, 0]

        relevant_edits = clusterer.find_relevant_cluster2(subquestion)  #已经能找到多个簇了
        
        sizePerQuery.append(len(relevant_edits))
    #print(sizePerQuery)
    print(sum(sizePerQuery)/len(sizePerQuery))
        
#         if ans not in relevant_edits:
#             continue
#         else:
#             cluster_ACC+=1



#         #原代码
#         # with torch.no_grad():
#         #     facts_input = tokenizer(relevant_edits, padding=True, truncation=True, max_length=256,
#         #                             return_tensors='pt').to(
#         #         args.device)
#         #     facts_emb = model(**facts_input).last_hidden_state[:, 0]

#         #改进代码
#         with torch.no_grad():
#             # 分批处理策略
#             batch_size = 8
#             facts_emb_list = []

#             # 修正点1：使用新的autocast API
#             with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
#                 for i in range(0, len(relevant_edits), batch_size):
#                     batch_edits = relevant_edits[i:i + batch_size]

#                     # Tokenization（修正点2：保持long类型）
#                     batch_inputs = tokenizer(
#                         batch_edits,
#                         padding='max_length',
#                         truncation=True,
#                         max_length=128,
#                         return_tensors='pt'
#                     )

#                     # 修正点3：保持long类型
#                     for key in batch_inputs:
#                         batch_inputs[key] = batch_inputs[key].to(args.device)  # 默认保持long类型

#                     # 前向计算
#                     outputs = model(**batch_inputs)
#                     batch_emb = outputs.last_hidden_state[:, 0].detach().clone()
#                     facts_emb_list.append(batch_emb)

#                     # 显存回收
#                     del batch_inputs, outputs
#                     torch.cuda.empty_cache()

#             facts_emb = torch.cat(facts_emb_list, dim=0)


#         #判断的是簇中edit和子问题的相似度
#         edit_similarities=torch.nn.functional.cosine_similarity(facts_emb, subquestion_emb.expand_as(facts_emb),dim=1)

#         if edit_similarities.max() < 0.6:  # 概率最大的都小于0.5，那应该没啥关系
#             continue
#             # prompt = prompt + gen + "Generated answer"
#         else:#找到簇中和子问题相似度大于0.5的edit
#             idxs = torch.where(edit_similarities > 0.6)[0].cpu().tolist()#找到relevant_edits中，所有相似度大于0.5元素的索引
#             #edits_mini和selected_similarities是对应的元素，相似度
#             # selected_similarities = [edit_similarities[i] for i in idxs].cpu().numpy() #所有相似度大于0.5具体相似性值

#             selected_similarities = edit_similarities[edit_similarities > 0.6].cpu().numpy() #所有相似度大于0.5具体相似性值
#             #edits_mini = [relevant_edits[i] for i in idxs] #簇中相似度大于0.5的edit元素

#             myinput = [(s.item(), int(idx)) for s, idx in zip(selected_similarities, idxs)]


#             #找到簇中比较closest的edit，需要传入元组列表(相似度，序号)
#             sorted_clusters = sorted(myinput, key=lambda x: x[0], reverse=True)
#             #这里threshold用来控制选择edit的数量，threhold越大，选择的edit数量越少
#             #后续可能要调小threhold，因为如果选择edit的数量过少，那么准确率会降低
#             relevant_edits_labels=arbi.get_candidates_zscore(sorted_clusters,args.threshold,5)

#             edits_mini = [relevant_edits[j] for j in relevant_edits_labels]
#             #现在要取到edit_mini集合中元素的相似度
#             edits_mini_similarities = [edit_similarities[j] for j in relevant_edits_labels]


#             if len(edits_mini) == 1:
#                 if  edits_mini[0]==ans:
#                     Retrieval_ACC+=1

#                 #prompt = prompt + gen + 'Generated answer: ' + edits_mini[0] + '.'



#             elif len(edits_mini) > 1: #这里以后要优化，判断simmax-simsecmax<0.1执行



#                 # 找到过滤后仍然有多个的候选的情况，edits_mini中的元素要执行E2Q
#                 hypo_querys = []

#                 for edit in edits_mini:
#                     hypo_query = e2q.get(edit)
#                     hypo_querys.extend(hypo_query)

#                 with torch.no_grad():
#                     hypo_querys_input = tokenizer(hypo_querys, padding=True, truncation=True, max_length=256,
#                                                   return_tensors='pt').to(
#                         args.device)
#                     hypo_querys_emb = model(**hypo_querys_input).last_hidden_state[:, 0]

#                 # 使用余弦相似度计算语义相似性
#                 hypo_similarities = torch.nn.functional.cosine_similarity(hypo_querys_emb,
#                                                                      subquestion_emb.expand_as(hypo_querys_emb),
#                                                                      dim=1)



#                 max_similarity_index = arbi.get_max_score_index(edits_mini_similarities, hypo_similarities)

#                 if ans  == edits_mini[int(max_similarity_index)]:
#                     Retrieval_ACC+=1
#         # current_mem = torch.cuda.memory_allocated() / 1024**2
#         # print(f"[Memory] After processing sample: {current_mem:.2f}MB")
#     print(f'cluter ACC = {cluster_ACC / total}')
#     print(f'Retrieval ACC = {Retrieval_ACC / total}')











