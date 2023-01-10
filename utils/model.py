import torch.nn as nn

from utils.constants import EMBED_DIMENSION, EMBED_MAX_NORM


from torch import Tensor
from torch.nn import functional as F

class DynamicDimsionEmbedding(nn.Embedding):
    def __init__(self, *args, **kwargs):
        super(DynamicDimsionEmbedding, self).__init__(*args, **kwargs)
        self.conv1d_1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv1d_2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride=3, padding=1)
        self.conv1d_3 = nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1)
        self.conv1d_4 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=3, stride=2, padding=1)
        
        self.weight = nn.Parameter(self.weight.unsqueeze(dim=1))
        print(">>>>", self.weight.shape)
        self.weight = nn.Parameter(self.conv1d_1(self.weight))
        self.weight = nn.Parameter(self.conv1d_2(self.weight))
        self.weight = nn.Parameter(self.conv1d_3(self.weight))
        self.weight = nn.Parameter(self.conv1d_4(self.weight))
        self.weight = nn.Parameter(self.weight.squeeze(dim=1))
            
    # def forward(self, input: Tensor) -> Tensor:
    #     x =  F.embedding(
    #         input, self.weight, self.padding_idx, self.max_norm,
    #         self.norm_type, self.scale_grad_by_freq, self.sparse)
    #     return x
class CBOW_Model(nn.Module):
    """
    Implementation of CBOW model described in paper:
    https://arxiv.org/abs/1301.3781
    """
    def __init__(self, vocab_size: int):
        super(CBOW_Model, self).__init__()
        self.embeddings = DynamicDimsionEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM,
        )
        
        reduced_embed_dimention = self.embeddings.weight.shape[-1]

        self.linear = nn.Linear(
            in_features=reduced_embed_dimention,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x


class SkipGram_Model(nn.Module):
    """
    Implementation of Skip-Gram model described in paper:
    https://arxiv.org/abs/1301.3781
    """
    def __init__(self, vocab_size: int):
        super(SkipGram_Model, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM,
        )
        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = self.linear(x)
        return x