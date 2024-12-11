import random
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, Subset

ds = load_dataset("wmt/wmt19", "zh-en")

class TranslationDataset(Dataset):
    def __init__(self, split: str):
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

if __name__ == "__main__":
    print(ds)
    loader = get_dataloader("train", 5, True)
    for batch in loader:
        print(batch['src'])
        print(batch['tgt'])
        break
