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

from ensemble_model import EnsembleModel
from image_to_text_dataset import ImageToTextDataModule
from train import ImageToText

if __name__ == "__main__":
    class CFG:
        seed = 42
        batch_size = 32
        learning_rate = 1e-3
        num_epochs = 50

    seed_everything(CFG.seed, workers=True)

    paths = ["/content/drive/MyDrive/CSE 252D Project/img_embeddings_clip_vit32.npy",
            "/content/drive/MyDrive/CSE 252D Project/img_embeddings_convnext.npy"]

    gt_path = "/content/drive/MyDrive/CSE 252D Project/prompt_embeddings.npy"

    dm = ImageToTextDataModule(CFG,paths,gt_path)
    dm.prepare_data()
    dm.setup()

    model = ImageToText(CFG)

    logger = pl_loggers.TensorBoardLogger("tb_logs", name="my_model")
    checkpoint_callback = ModelCheckpoint(dirpath="./model_ckpts",
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

    trainer.test(model=model,
                 datamodule=dm,
                 ckpt_path="/content/drive/MyDrive/CSE 252D Project/model_ckpts/epoch=99-val_loss_epoch=0.0015.ckpt")