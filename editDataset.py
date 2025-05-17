from torch.utils.data import Dataset, DataLoader

class EditDataset(Dataset):
    def __init__(self,edits,questions,nos):
        '''

        :param edits:
        :param questions:
        :param nos:
        '''
        self.edits = edits
        self.questions = questions
        self.nos = nos

    def __getitem__(self, index):
        
        return self.edits[index],self.questions[index],self.nos[index]

    def __len__(self):
        return len(self.edits)


class EditSeqDataset(Dataset):
    '''
    在类括号中表示继承父类torch.utils.data.Dataset
    '''
    def __init__(self,data):
        '''
        data=[
            {"edit": "Edit_A", "questions": ["Q1", "Q2", "Q3"]},
            {"edit": "Edit_B", "questions": ["Q4", "Q5", "Q6"]},
                ...
            ]
        :param data:
        '''
        self.pairs = data

    def __getitem__(self, index):
        #在外面肯定有迭代调用本方法获取data里面的数据
        #返回的是一个元组，("Edit_A"，["Q1", "Q2", "Q3"])
        return self.pairs[index]["edit"],self.pairs[index]["questions"]

    def __len__(self):
        return len(self.pairs)