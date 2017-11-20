import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class Model(nn.Module):


    def __init__(self, vocab_size, embedding_dim):
        super(Model, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)


    def forward(self, inputs):
        em = self.embeddings(inputs)
        embeds = torch.sum(em, dim = 1)
        out = self.linear1(embeds)
        out = self.linear2(out)
        log_probs = F.log_softmax(out)
        return log_probs
