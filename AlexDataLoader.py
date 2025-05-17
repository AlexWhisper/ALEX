import json
import torch
import torch.nn.functional
from sklearn.model_selection import StratifiedKFold

# loading dataset
class DataLoader:
    def __init__(self, dataset_name, batch_size, device="cuda:0"):

        self.dataset_name = dataset_name
        self.batch_size=batch_size
        self.device = device
        with open(f'./datasets/{self.dataset_name}.json', 'r') as f:
            self.dataset = json.load(f)
        self.edited_num=len(self.dataset)
        self.samples = self.edited_num
    def getAllEdit(self):
        # construct edit batches of different sizes
        dataset_splits = self.samples // self.edited_num
        hop_labels = [i // 1000 for i in range(3000)]
        if self.edited_num == 1 or self.edited_num == self.samples:
            hop_labels = [0 for i in range(self.samples)]

        if self.edited_num == 1:
            skf = StratifiedKFold(n_splits=dataset_splits, shuffle=False)
        elif self.edited_num != self.samples:
            skf = StratifiedKFold(n_splits=dataset_splits, shuffle=True, random_state=42)

        subsets = []

        if self.edited_num == self.samples:
            subsets.append([i for i in range(self.samples)])
        else:
            for _, test_index in skf.split(self.dataset, hop_labels):
                subsets.append(test_index)

        dataset_batch = []
        edits_batch = []
        embs_batch = []

        for batch in subsets:
            sub_dataset = [self.dataset[index] for index in batch]
            new_facts = set()
            for d in sub_dataset:
                for r in d["requested_rewrite"]:
                    rel=r["prompt"].replace("{}", "")
                    fac=(r["subject"],rel,r["target_new"]["str"] )
                    # if fac not in new_facts:
                    #     new_facts.append(fac)
                    if f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}' not in new_facts:
                        new_facts.add(f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}')

            new_facts = list(new_facts)

            # with torch.no_grad():
            #     facts_input = tokenizer(new_facts, padding=True, truncation=True, max_length=256, return_tensors='pt').to(
            #         device)
            #     facts_emb = classifier(**facts_input).last_hidden_state[:, 0]

            dataset_batch.append(sub_dataset)
            #embs_batch.append(facts_emb)
            edits_batch.append(new_facts)

        return edits_batch[0]

    def loadQuery(self):
        qa_pair={}
        for d in self.dataset:
            for r in d["requested_rewrite"]:
                qa_pair[r['question']]=f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}'
        return qa_pair

    def getDataEmb(self,classifier,tokenizer):
        new_facts=self.getAllEdit()
        with torch.no_grad():
            facts_input = tokenizer(new_facts, padding=True, truncation=True, max_length=256, return_tensors='pt').to(
                self.device)
            facts_emb = classifier(**facts_input).last_hidden_state[:, 0]
        return facts_emb

    def getAllData(self):
        return self.dataset

