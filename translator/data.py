import random
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, Subset


class TranslationDataset(Dataset):
    def __init__(self, split: str):
        ds = load_dataset("wmt/wmt19", "zh-en")
        self.dataset = ds[split]
        self.src_lang = "en"
        self.tgt_lang = "zh"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        src_sentence = sample['translation'][self.src_lang]
        tgt_sentence = sample['translation'][self.tgt_lang]
        return {
            "src": src_sentence,
            "tgt": tgt_sentence
        }

def get_dataloader(split: str, batch_size: int, shuffle: bool = True):
    dataset = TranslationDataset(split)
    sample_size = int(0.05 * len(dataset))
    

    sampled_indices = random.sample(range(len(dataset)), sample_size)
    
    
    sampled_dataset = Subset(dataset, sampled_indices)
    dataloader = DataLoader(sampled_dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

class TranslationDatasetV2(Dataset):
    def __init__(self, src_path: str, tgt_path: str):
        with open(src_path, "r", encoding="utf-8") as f:
            self.src_data = f.read().splitlines()
        with open(tgt_path, "r", encoding="utf-8") as f:
            self.tgt_data = f.read().splitlines()
        assert len(self.src_data) == len(self.tgt_data), "The number of source and target sentences must be the same"
        
        length = len(self.src_data)
        self.dataset = [None for _ in range(length)]
        for i in range(length):
            self.dataset[i] = {
                "src": self.src_data[i],
                "tgt": self.tgt_data[i]
            }
    
    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.src_data)
    

def get_dataloaderV2(src_path: str, tgt_path: str, batch_size: int, shuffle: bool = True):
    dataset = TranslationDatasetV2(src_path, tgt_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader