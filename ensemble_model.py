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
    def __init__(self):
        super(EnsembleModel, self).__init__()
        
        self.linear1 = nn.Linear(768,512)
        self.relu = nn.ReLU()
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=512,nhead=8)
        self.tranformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer,num_layers=3)
        self.linear2 = nn.Linear(512,384)


    def forward(self, x):
        emb1 = x[0]
        emb2 = self.relu(self.linear1(x[1]))
        
        emb_inp = torch.stack([emb1,emb2],dim=1)
        out = self.tranformer_encoder(emb_inp)
        out = self.linear2(out)

        return out[:,0,:]

    
    
    