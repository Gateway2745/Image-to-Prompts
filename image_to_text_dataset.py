import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class ImageToTextDataset(torch.utils.data.Dataset):
    def __init__(self, paths, gt_path, indices):
        super(ImageToTextDataset, self).__init__()
        self.paths = paths
        self.embeddings = [np.load(f)[indices] for f in paths]
        self.gt_embeddings = np.load(gt_path)[indices]

    def __len__(self): 
        return self.embeddings[0].shape[0]

    def __getitem__(self, index):
        embeddings_list_tensor = [torch.tensor(e[index],dtype=torch.float32) for e in self.embeddings]
        embeddings_list_tensor.append(torch.tensor(self.gt_embeddings[index]))
        return embeddings_list_tensor
    
    
class ImageToTextDataModule(pl.LightningDataModule):
    def __init__(self,CFG,paths,gt_path):
        super(ImageToTextDataModule, self).__init__()
        self.CFG=CFG
        self.paths = paths
        self.gt_path = gt_path
        self.num_examples = CFG.num_examples
        self.train_fraction = CFG.train_fraction

    def prepare_data(self):
        self.permuted_indices = torch.randperm(self.num_examples)
        self.train_indices = self.permuted_indices[:int(self.num_examples*self.train_fraction)]
        self.test_indices = self.permuted_indices[int(self.num_examples*self.train_fraction):]

    def setup(self,stage=None):
        self.train_dataset = ImageToTextDataset(self.paths, self.gt_path, self.train_indices)
        self.test_dataset = ImageToTextDataset(self.paths, self.gt_path, self.test_indices)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size=self.CFG.batch_size)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=self.CFG.batch_size,shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=self.CFG.batch_size,shuffle=False)

    def teardown(self,stage=None):
        import gc
        del self.train_dataset
        del self.test_dataset
        gc.collect()
