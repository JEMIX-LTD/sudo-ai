#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""@author: Aymen Jemi (jemix) <jemiaymen@gmail.com>

Copyright (c) 2021 Aymen Jemi SUDO-AI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import DEVICE, MAX_WORDS

"""Hybrid Attention for Extreme Multi-Label Text Classification """


class ExtremMutliLabelTextClassification(nn.Module):

    def __init__(self,
                 n_class=3714,
                 vocab_size=30001,
                 embedding_size=300,
                 hidden_size=256,
                 d_a=256,
                 multiclass=False):

        super().__init__()
        self.embedding_size = embedding_size
        self.num_labels = n_class
        self.max_seq = MAX_WORDS
        self.hidden_size = hidden_size
        self.multiclass = multiclass

        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)

        self.lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # interaction-attention layer
        self.key_layer = torch.nn.Linear(2*self.hidden_size, self.hidden_size)
        self.query_layer = torch.nn.Linear(self.hidden_size, self.hidden_size)

        # self-attn layer
        self.linear_first = torch.nn.Linear(2*self.hidden_size, d_a)
        self.linear_second = torch.nn.Linear(d_a, self.num_labels)

        # weight adaptive layer
        self.linear_weight1 = torch.nn.Linear(2*self.hidden_size, 1)
        self.linear_weight2 = torch.nn.Linear(2*self.hidden_size, 1)

        # shared for all attention component
        self.linear_final = torch.nn.Linear(
            2*self.hidden_size, self.hidden_size)
        self.output_layer = torch.nn.Linear(self.hidden_size, 1)

        label_embedding = torch.FloatTensor(self.num_labels, self.hidden_size)
        nn.init.xavier_normal_(label_embedding)
        self.label_embedding = nn.Parameter(
            label_embedding,
            requires_grad=False
        )

    def init_hidden(self, batch_size):
        return (torch.zeros(2, batch_size, self.hidden_size, device=DEVICE),
                torch.zeros(2, batch_size, self.hidden_size, device=DEVICE))

    def forward(self, input):

        emb = self.word_embeddings(input)

        hidden_state = self.init_hidden(emb.size(0))
        output, hidden_state = self.lstm(
            emb, hidden_state)  # [batch,seq,2*hidden]

        # get attn_key
        attn_key = self.key_layer(output)  # [batch,seq,hidden]
        attn_key = attn_key.transpose(1, 2)  # [batch,hidden,seq]
        # get attn_query
        label_emb = self.label_embedding.expand((attn_key.size(0), self.label_embedding.size(
            0), self.label_embedding.size(1)))  # [batch,L,label_emb]
        label_emb = self.query_layer(label_emb)  # [batch,L,label_emb]

        # attention
        similarity = torch.bmm(label_emb, attn_key)  # [batch,L,seq]
        similarity = F.softmax(similarity, dim=2)

        out1 = torch.bmm(similarity, output)  # [batch,L,label_emb]

        # self-attn output
        self_attn = torch.tanh(
            self.linear_first(output))  # [batch,seq,d_a]
        self_attn = self.linear_second(self_attn)  # [batch,seq,L]
        self_attn = F.softmax(self_attn, dim=1)
        self_attn = self_attn.transpose(1, 2)  # [batch,L,seq]
        out2 = torch.bmm(self_attn, output)  # [batch,L,hidden]

        factor1 = torch.sigmoid(self.linear_weight1(out1))
        factor2 = torch.sigmoid(self.linear_weight2(out2))
        factor1 = factor1/(factor1+factor2)
        factor2 = 1-factor1

        out = factor1*out1+factor2*out2

        out = F.relu(self.linear_final(out))

        if self.multiclass is True:
            out = torch.sigmoid(self.output_layer(
                out).squeeze(-1))  # [batch,L]
        else:
            out = torch.log_softmax(self.output_layer(
                out).squeeze(-1), 0, torch.float64)

        return out.requires_grad_(True)
