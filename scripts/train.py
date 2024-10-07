import pickle
import matplotlib.pyplot as plt
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import torch
torch.manual_seed(40)
import pdb
import copy
import re
import numpy as np
from torchmetrics.text import CharErrorRate, WordErrorRate
import torchvision.transforms as transforms
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import glob
import os.path
import hashlib
from kraken.lib.vgsl import TorchVGSLModel
from torchvision.models import resnet18
import argparse
import logging
from utils import LineImageDataset, MyNN, LatinTranscriber

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="source")
    parser.add_argument("--lines_dir", type=str, default="output")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--output", type=str, default="model.pkl")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    all_chars = " -.ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuvwxyz¶"
    
    char_to_num = {}
    num_to_char = {}
    for i in range(len(all_chars)):
        num_to_char[i+1] = all_chars[i]
        char_to_num[all_chars[i]] = i+1

    train_transform = transforms.Compose(
        [
            transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
            transforms.RandomAffine(0.7, translate=(0.01, 0.02), scale=(0.98, 1.02)),
            transforms.RandomChoice([
            transforms.RandomAdjustSharpness(2, p=0.5),
            transforms.GaussianBlur(21, (1,6))
            ]),      
            transforms.Normalize(0.15, 0.38)
        ])

    val_transform = transforms.Compose(
        [  
            transforms.Normalize(0.15, 0.38)
        ])

    #train_dataset = LineImageDataset("data/", char_to_num, num_to_char, data_type="train", transform=train_transform)
    #val_dataset = LineImageDataset("data/", char_to_num, num_to_char, data_type="val", transform=val_transform)
    dataset = LineImageDataset(args.input_dir, args.lines_dir, char_to_num, num_to_char, data_type="all", transform=val_transform)
    #train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_indices, val_indices = torch.utils.data.random_split(range(len(dataset)), [train_size, val_size])

    train_dataset = LineImageDataset(args.input_dir, args.lines_dir, char_to_num, num_to_char, data_type="all", transform=train_transform)
    val_dataset = LineImageDataset(args.input_dir, args.lines_dir, char_to_num, num_to_char, data_type="all", transform=val_transform)

    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    print(len(train_dataset))
    #plt.imshow(torch.tensor(np.load("line_0_JUST1-734m5d.npy")))
    #idx = 10
    #print(val_dataset[idx]["image"][0].mean(), val_dataset[idx]["image"][0].std())
    #plt.figure(figsize=(16,6))
    #plt.imshow(train_dataset[idx]["image"][0], cmap="gray")


    all_chars = " -.ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuvwxyz¶"

    net = MyNN()

    transcriber = LatinTranscriber(net, num_to_char)


    #plt.figure(figsize=(16,6))
    #idx = 10
    #print(torch.min(val_dataset[1]["image"]), torch.max(val_dataset[1]["image"]))
    #plt.imshow(val_dataset[0]["image"], cmap="gray")
    #plt.imshow(val_dataset[0][0], cmap="gray")

    train_loader = utils.data.DataLoader(train_dataset, num_workers=4)
    valid_loader = utils.data.DataLoader(val_dataset, num_workers=4)


    checkpoint_callback = ModelCheckpoint(
        monitor="val_word_acc", mode="max", dirpath=args.checkpoint_dir, filename="ocr"
    )

    trainer = L.Trainer(accumulate_grad_batches=1, max_epochs=args.max_epochs, enable_progress_bar=True, callbacks=[checkpoint_callback], devices=[1])

    trainer.fit(transcriber, train_loader, valid_loader)
    
    trainer.validate(ckpt_path="best", dataloaders=valid_loader)

    # save the model
    torch.save(net.state_dict(), args.output)
