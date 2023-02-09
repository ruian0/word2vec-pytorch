from typing import List

import pandas as pd
import torch
import torch.nn as nn

from utils.constants import EMBED_DIMENSION, EMBED_MAX_NORM

embedding_cache_path = f"/mnt/ruian/gpt3/embeddings/example_embeddings_cache_40831.pkl"
try:
    print("loading pre-saved gpt3 embeddings into pandas")
    embedding_cache = pd.read_pickle(embedding_cache_path)
except:
    print("loading pre-saved gpt3 embeddings into pandas is broken...")
    embedding_cache = {}

def embedding_from_string(
    string: str,
    engine: str = "text-embedding-ada-002",
    embedding_cache=embedding_cache,
    embedding_cache_path=embedding_cache_path
) -> List:
    """Return embedding of given string, using a cache to avoid recomputing."""
    if (string, engine) not in embedding_cache.keys():
        print("Adding to embedding cache, contact Rui An for more")
        print((string, engine))
        embedding_cache[(string, engine)] = get_embedding(string, engine)
#         with open(embedding_cache_path, "wb") as embedding_cache_file:
#             pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(string, engine)]

class CBOW_Model(nn.Module):
    """
    Implementation of CBOW model described in paper:
    https://arxiv.org/abs/1301.3781
    """
    def __init__(self, vocab):
        super(CBOW_Model, self).__init__()
        vocab_size = len(vocab)

        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM,
        )
        
        with torch.no_grad():
            for skill, idx in vocab.get_stoi().items():
                emd = embedding_from_string(skill)
                self.embeddings.weight[idx] = torch.tensor(emd,  requires_grad=False)
                
        self.embeddings = self.embeddings.to('cpu')
                
        self.features = nn.Sequential(
            nn.Flatten(start_dim=0, end_dim=1),
            nn.Unflatten(1, (1, 1536)),
            nn.Conv1d(1, 16, 3, stride=2), 
            nn.Conv1d(16, 16, 3, stride=2), 
            nn.Conv1d(16, 1, 3, stride=2), 
            nn.Linear(191, 100),
            # nn.Linear(1536, 768),
            # nn.Linear(768, 384),
            # nn.Linear(384, 100),
            nn.Flatten(start_dim=1, end_dim=2),
            nn.Unflatten(0, (-1, 8)),
        )

        self.linear = nn.Linear(
            in_features=100,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = self.features(x) 
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
