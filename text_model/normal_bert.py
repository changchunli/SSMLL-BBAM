import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_transformers import *


class ClassificationBertBase(nn.Module):
    def __init__(self, num_labels=2):
        super(ClassificationBertBase, self).__init__()
        # Load pre-trained bert model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.linear = nn.Sequential(nn.Linear(768, 256), nn.GELU())
        self.classifier = nn.Linear(256, num_labels)

    def forward(self, x, flag=0, length=256):
        # Encode input text
        all_hidden, pooler = self.bert(x)

        pooled_output = torch.mean(all_hidden, 1)
        # Use linear layer to do the predictions
        feature = self.linear(pooled_output)

        output = self.classifier(feature)

        return output


class ClassificationBertNormalization(nn.Module):
    def __init__(self, num_labels=2):
        super(ClassificationBertNormalization, self).__init__()
        # Load pre-trained bert model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.linear = nn.Sequential(nn.Linear(768, 256), nn.GELU())
        self.weight = nn.Parameter(torch.FloatTensor(num_labels, 256))
        # self.bias = nn.Parameter(torch.FloatTensor(num_labels))
        # k = math.sqrt(1.0 / 256)

        # nn.init.uniform_(self.weight, -k, k)
        # nn.init.uniform_(self.bias, -k, k)
        nn.init.xavier_normal_(self.weight)

    def forward(self, x, flag=0, length=256):
        # Encode input text
        all_hidden, pooler = self.bert(x)

        pooled_output = torch.mean(all_hidden, 1)
        # Use linear layer to do the predictions
        feature = self.linear(pooled_output)

        cosine = F.linear(F.normalize(feature), F.normalize(self.weight))
        if flag == 1:
            output = (cosine, feature)
        else:
            # output = F.linear(feature, self.weight)
            output = cosine

        return output

