import numpy as np
import os
import sys
import yaml

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

from ensemble_model import EnsembleModel
from image_to_text_dataset import ImageToTextDataModule


class ImageToText(pl.LightningModule):
    def __init__(self, CFG):
        super(ImageToText, self).__init__()
        self.model = EnsembleModel(CFG)
        self.loss_fn = nn.CosineEmbeddingLoss()
        self.cs = nn.CosineSimilarity()
        self.CFG = CFG
        self.y = torch.tensor([1]).cuda()

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        inps = batch[:-1]
        y = batch[-1]
        out = self.model(inps)
        loss = self.loss_fn(out, y, target=self.y)
        self.log("train_loss",loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inps = batch[:-1]
        y = batch[-1]
        out = self.model(inps)
        loss = self.loss_fn(out, y, target=self.y).item()
        sim = self.cs(out,y).mean(0).item()
        
        self.log("val_loss_epoch", loss, logger=True, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_cosine_sim_epoch", sim, logger=True, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        inps = batch[:-1]
        y = batch[-1]
        out = self.model(inps)
        loss = self.loss_fn(out, y, target=self.y).item()
        sim = self.cs(out,y).mean(0).item()
        
        self.log("test_loss_epoch", loss)
        self.log("test_cosine_sim_epoch", sim)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.CFG.learning_rate)
        return optimizer

if __name__ == "__main__":

    CONFIG_FILE = sys.argv[1]
    configs = yaml.safe_load(open(CONFIG_FILE, 'r'))

    class cfg(object):
      def __init__(self, d):    
          for key in d:
              setattr(self, key, d[key])

    CFG = cfg(configs)

    seed_everything(CFG.seed, workers=True)

    paths = CFG.embedding_paths
    
    gt_path = CFG.gt_path
    
    dm = ImageToTextDataModule(CFG,paths,gt_path)
    dm.prepare_data()
    dm.setup()

    model = ImageToText(CFG)

    logger = pl_loggers.TensorBoardLogger("tb_logs", name=CFG.exp_name)
    checkpoint_callback = ModelCheckpoint(dirpath="./model_ckpts/"+CFG.exp_name,
                                        monitor='val_loss_epoch',
                                        save_top_k=1,
                                        save_last=True,
                                        save_weights_only=True,
                                        filename='{epoch:02d}-{val_loss_epoch:.4f}',
                                        verbose=False,
                                        mode='min')

    trainer = Trainer(
        max_epochs=CFG.num_epochs,
        precision=32,
        accelerator="cuda",
        devices=1,
        callbacks=[checkpoint_callback],
        logger = logger,
        log_every_n_steps=4,
        deterministic=True,
    )

    trainer.fit(model,dm)
