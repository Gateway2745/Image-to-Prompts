import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch.utils.data import DataLoader
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint


class EnsembleModel(nn.Module):
    def __init__(self, CFG):
        super(EnsembleModel, self).__init__()
        
        self.linear1 = nn.Linear(512,384)
        self.linear2 = nn.Linear(768,384)

        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=384,nhead=CFG.num_heads)
        self.tranformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer,num_layers=CFG.num_layers)
        self.linear_out = nn.Linear(384,384)


    def forward(self, x):
        emb1 = self.leaky_relu(self.linear1(x[0]))
        emb2 = self.leaky_relu(self.linear2(x[1]))
        emb3 = x[2]
        
        emb_inp = torch.stack([emb1,emb2,emb3],dim=1)
        out = self.tranformer_encoder(emb_inp)
        out = self.linear_out(out)

        return out[:,0,:]