import util
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from model import Encoder

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # TODO: Define model architecture
        self.model = Encoder()
        
    def forward(self, x):
        # TODO: Implement forward pass
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        # TODO: Implement training step
        x, y = batch
        y_hat = self(x)
        loss = None # Define loss
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        # TODO: Implement validation step
        x, y = batch
        y_hat = self(x)
        loss = None # Define loss
        self.log('val_loss', loss)
        
    def configure_optimizers(self):
        # TODO: Define optimizer
        optimizer = optim.Adam(self.parameters())
        return optimizer

def train():
    # TODO: Initialize data
    train_dataset = None
    val_dataset = None
    
    train_loader = DataLoader(train_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize model and trainer
    model = LitModel()
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )
    
    logger = TensorBoardLogger("lightning_logs", name="my_model")
    
    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[checkpoint_callback],
        logger=logger,
        accelerator='auto'
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    train()