import torch
from torch.utils.data import Dataset

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class MyDataset(Dataset):
    def __init__(self, features, mode):
        self.nums = len(features)

        self.input_ids = [torch.tensor(example.input_ids).to(device).long() for example in features]
        self.input_mask = [torch.tensor(example.input_mask).to(device).float() for example in features]
        self.segment_ids = [torch.tensor(example.segment_ids).to(device).long() for example in features]

        self.label_id = None
        if mode != 'test':
            self.label_id = [torch.tensor(example.label_id).to(device) for example in features]

    def __getitem__(self, index):
        data = {'input_ids': self.input_ids[index],
                'input_mask': self.input_mask[index],
                'segment_ids': self.segment_ids[index]}

        if self.label_id is not None:
            data['label_id'] = self.label_id[index]

        return data

    def __len__(self):
        return self.nums
