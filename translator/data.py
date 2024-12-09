from datasets import load_dataset

ds = load_dataset("wmt/wmt19", "zh-en")
print(ds[0])