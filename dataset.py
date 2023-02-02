from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, text, labels):
        '''
        Dataset object for base model
        :param data:
        '''
        self.text = np.array(text)
        self.label = np.array(labels)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx], self.label[idx]

