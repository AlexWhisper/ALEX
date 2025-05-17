import json

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
        llmModel.eval()

    elif llm=='llama3':
        llm_name= "/mnt/sda2/wmh/Models/llama3.1-8b/"
        llamatokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
        llamamodel = transformers.AutoModelForCausalLM.from_pretrained(model_dir,torch_dtype=torch.float16).to(device)
        
        if llamatokenizer.pad_token_id is None:
            llamatokenizer.pad_token_id = llamatokenizer.eos_token_id if llamatokenizer.eos_token_id is not None else 0  
        if llamatokenizer.eos_token_id is None:
            llamatokenizer.eos_token_id = llamatokenizer.pad_token_id if llamatokenizer.pad_token_id is not None else 0 
        llamamodel.eval()
        
    elif llm=="deepseek":
        llmtokenizer = ""
        llmModel = ""


    arbi=Arbitration.Arbi(args.threshold,args.alpha,args.beta)

    #加载Alex模型
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model=transformers.AutoModel.from_pretrained(model_name).to(args.device)
    #加载call_gpt方法

    call_gpt = Load_LLM.method_map[llm]
    dataloader=DataLoader(dataset_name,batch_size=1,device=args.device)
    dataset_batch=dataloader.getAllData()
    edit=dataloader.getAllEdit()
    edit_emb=dataloader.getDataEmb(model,tokenizer)

    #加载E2Q
    e2q = E2Q(dataset_name,args.device)

    #聚类处理
    clusterer = TextSentenceCluster(edit,edit_emb, model, tokenizer,arbi,args)
    labels = clusterer.cluster(method='kmeans', n_clusters=args.n_clusters)

    with open('prompts/KGprompt.txt', 'r') as f:
        KGprompt = f.read()
    cor = 0
    ver_cor = 0
    cor_list = [0, 0, 0]
    ver_cor_list = [0, 0, 0]
    tot = 0
    fail_cases = {}

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


    for d in tqdm(dataset_batch[417:]):
        tot += 1
        have_cor = False

        #添加use_kgprompt
        for q in d["questions"]:

            new_answer = d["new_answer"]
            new_answer_alias = d["new_answer_alias"]


            found_ans = False
            #考虑换为
            # prompt = KGprompt + "\n\nQuestion: " + q + "\n" + "Entity of Question: "
            prompt = KGprompt + "\n\nQuestion: " + q

            subqa_trace = []

            for i in range(5):
                # prompt the model to identify the subquestion
                if llm=="deepseek":
                    truncation = ["Generated answer:", "\n\n"]
                else:
                    truncation = len(prompt)
                gen = call_gpt(llmModel,llmtokenizer,prompt, truncation,args.device)

                last_sent = gen.strip().split('\n')[-1]
                # if final answer is there, get the answer and exit
                if last_sent.startswith('Final answer: '):
                    found_ans = True
                    ans = last_sent[len("Final answer: "):]
                    prompt = prompt + gen
                    break
                # otherwise, extract the generated subquestion
                if len(gen.strip().split('\n')) < 1 or len(gen.strip().split('\n')) > 3:
                    prompt = prompt + gen
                    gen_failed_count_1+=1
                    break  # failed case
                subquestion = gen.strip().split('\n')[-1]
                if not subquestion.startswith('Subquestion: '):
                    prompt = prompt + gen
                    gen_failed_count_2 += 2
                    break  # failed case
                subquestion = subquestion[len("Subquestion: "):]

                subqa_trace.append(subquestion)
                # 子问题嵌入
                with torch.no_grad():
                    subquestion_input = tokenizer(subquestion, padding=True, truncation=True, max_length=256,
                                                  return_tensors='pt').to(args.device)
                    subquestion_emb = model(**subquestion_input).last_hidden_state[:, 0]

                relevant_edits = clusterer.find_relevant_cluster2(subquestion)  #已经能找到多个簇了

                with torch.no_grad():
                    facts_input = tokenizer(relevant_edits, padding=True, truncation=True, max_length=256,
                                            return_tensors='pt').to(
                        args.device)
                    facts_emb = model(**facts_input).last_hidden_state[:, 0]

                #判断的是簇中edit和子问题的相似度
                edit_similarities=torch.nn.functional.cosine_similarity(facts_emb, subquestion_emb.expand_as(facts_emb),dim=1)

                if edit_similarities.max() < 0.6:  # 概率最大的都小于0.5，那应该没啥关系
                    prompt = prompt + gen + "Generated answer"
                else:#找到簇中和子问题相似度大于0.5的edit
                    idxs = torch.where(edit_similarities > 0.6)[0].cpu().tolist()#找到relevant_edits中，所有相似度大于0.5元素的索引
                    #edits_mini和selected_similarities是对应的元素，相似度
                    # selected_similarities = [edit_similarities[i] for i in idxs].cpu().numpy() #所有相似度大于0.5具体相似性值

                    selected_similarities = edit_similarities[edit_similarities > 0.6].cpu().numpy() #所有相似度大于0.5具体相似性值
                    #edits_mini = [relevant_edits[i] for i in idxs] #簇中相似度大于0.5的edit元素

                    myinput = [(s.item(), int(idx)) for s, idx in zip(selected_similarities, idxs)]




                    #找到簇中比较closest的edit，需要传入元组列表(相似度，序号)
                    sorted_clusters = sorted(myinput, key=lambda x: x[0], reverse=True)
                    #这里threshold用来控制选择edit的数量，threhold越大，选择的edit数量越少
                    #后续可能要调小threhold，因为如果选择edit的数量过少，那么准确率会降低
                    relevant_edits_labels=arbi.get_candidates_zscore(sorted_clusters,args.threshold,5)

                    edits_mini = [relevant_edits[j] for j in relevant_edits_labels]
                    #现在要取到edit_mini集合中元素的相似度
                    edits_mini_similarities = [edit_similarities[j] for j in relevant_edits_labels]


                    if len(edits_mini) == 1:
                        prompt = prompt + gen + 'Generated answer: ' + edits_mini[0] + '.'



                    elif len(edits_mini) > 1: #这里以后要优化，判断simmax-simsecmax<0.1执行



                        # 找到过滤后仍然有多个的候选的情况，edits_mini中的元素要执行E2Q
                        hypo_querys = []

                        for edit in edits_mini:
                            hypo_query = e2q.get(edit)
                            hypo_querys.extend(hypo_query)

                        with torch.no_grad():
                            hypo_querys_input = tokenizer(hypo_querys, padding=True, truncation=True, max_length=256,
                                                          return_tensors='pt').to(
                                args.device)
                            hypo_querys_emb = model(**hypo_querys_input).last_hidden_state[:, 0]

                        # 使用余弦相似度计算语义相似性
                        hypo_similarities = torch.nn.functional.cosine_similarity(hypo_querys_emb,
                                                                             subquestion_emb.expand_as(hypo_querys_emb),
                                                                             dim=1)


                        # #疑似需要删除！！！！！！！！！！，因为如果生成hypoth—_query可信度太低，会导致不使用任何edit，哪怕edit中有超高相似度的edit
                        # if hypo_similarities.max() < 0.5: #概率最大的都小于0.5，那应该没啥关系,,疑似需要删除
                        #     count_hypoSimBlow05 += 1
                        #     prompt = prompt + gen + 'Generated answer'+edits_mini[0] + '.'
                        #
                        # else:
                        #     #Arbitration.Arbi.getMaxScoreIndex(similarities, similarities)
                        #     max_similarity_index=arbi.getMaxScoreIndex(edits_mini_similarities, hypo_similarities)
                        #
                        #     #max_similarity_index = hypo_similarities.argmax().item() / 3
                        #     # prompt = prompt + gen[:-remove_length] + edip ts_batch[no][index] + '.\nIntermediate answer:'
                        #     prompt = prompt + gen + 'Generated answer: ' + edits_mini[int(max_similarity_index)] + '.'


                        max_similarity_index = arbi.get_max_score_index(edits_mini_similarities, hypo_similarities)


                        prompt = prompt + gen + 'Generated answer: ' + edits_mini[int(max_similarity_index)] + '.'

                    else:
                        #这里应该不会执行，因为edits_mini必定包含和query相似性最大的edit
                        prompt = prompt + gen + 'Generated answer'

            if not found_ans:
                continue
            prompt = prompt.strip().split('\n\n')[-1]

            # if the answer is correct -> positive instance for Acc
            if ans == d["new_answer"] or ans in d["new_answer_alias"]:
                instance_type = verify_subquestion_path(prompt, len(d["single_hops"]), d["new_single_hops"])
                if not have_cor:
                    cor += 1
                    cor_list[len(d["single_hops"]) - 2] += 1
                    have_cor = True
                if not instance_type:  # positive instance for Hop-Acc
                    ver_cor += 1
                    ver_cor_list[len(d["single_hops"]) - 2] += 1
                    #print('verification passed sum:{} hop:{}'.format(ver_cor, ver_cor_list[len(d["single_hops"]) - 2]))

                    break
        if not have_cor:
            for r in d["requested_rewrite"]:
                fail_edit=f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}'
                if fail_edit not in fail_cases:
                    fail_cases[fail_edit]=r['question']
                    try:
                        with open("fail_cases.json", 'w') as f:
                            json.dump(fail_cases, f, indent=2)
                    except Exception as e:
                        pass
                    #qa_pair[r['question']] = f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}'





        print(f'Acc = {cor / tot} ({cor} / {tot})')
        print(f'Hop-Acc = {ver_cor / tot} ({ver_cor} / {tot})')
        print(f'2-hop = {cor_list[0]}')
        print(f'3-hop = {cor_list[1]}')
        print(f'4-hop = {cor_list[2]}')
        print(f'gen_failed_count_1= {gen_failed_count_1}')
        print(f'gen_failed_count_2= {gen_failed_count_2}')





