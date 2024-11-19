import torch
from torch import nn
import numpy as np
def PositionEmbedding(seq_len, d,n):

    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in range(d):
            
            denominator = np.power(n, 2*i/d)
            if i%2==0:
                P[k, i] = np.sin(k/denominator)
            else:
                P[k, i] = np.cos(k/denominator)
    return P


 
   

    def forward(self, x):
        seq_len = x.size(1)  # 获取输入序列长度
        return x + self.encoding[:, :seq_len, :].to(x.device)  # 将位置编码加到输入上

if __name__ == '__main__':
    P = PositionEmbedding(seq_len=4, d=4, n=10000)
    print(P)
    embedding = nn.Embedding(10, 2) # 一共有10个词，每个词用2维词向量表示
    input = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]]) # 两个句子，每个句子有4个词
    output = embedding(input)
    print(output.size()) # sentence_num * seq_len * embedding_dim
    # 实际使用中，每次输入的sentence_num是预设的batch_size