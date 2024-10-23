import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import numpy as np
import torchvision.transforms as transforms
import argparse
import utils
import os
import logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="source")
    parser.add_argument("--lines_dir", type=str, default="output")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--output", type=str, default="model.pkl")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--train_proportion", type=float, default=0.9)
    parser.add_argument("--gpu_devices", type=int, nargs="+", default=[0])
    parser.add_argument("--data_cutoff", type=int, default=-1)
    args = parser.parse_args()

    logging_file = args.output.split(".")[0] + ".log"
    logging.basicConfig(filename=logging_file, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filemode='w')

    logging.info(args)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)


    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    if not torch.cuda.is_available():
        raise ValueError("CUDA not available. Exiting.")
    else:
        for device_idx in args.gpu_devices:
            device = torch.device(f"cuda:{device_idx}")
            free_mem, total_mem = torch.cuda.mem_get_info(device)
            used_MB = (total_mem - free_mem)/1024**2
            if used_MB > 3000:
                raise ValueError(f"Device {device_idx} is not empty, it is {used_MB} MB. Exiting.")


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

    dataset = utils.LineImageDataset(args.input_dir, args.lines_dir, char_to_num, num_to_char, data_type="all", transform=val_transform)
 
    dataset = utils.LineImageDataset(
        dirname=args.input_dir,
        lines_dir=args.lines_dir,
        char_to_num=char_to_num,
        num_to_char=num_to_char,
        data_type="all",
        transform=val_transform
    )

    train_size = int(args.train_proportion * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    net = utils.MyNN()

    transcriber = utils.LatinTranscriber(net, num_to_char)

    if args.data_cutoff > 0:
        logging.info(f"Cutting off data at {args.data_cutoff}")
        train_dataset = torch.utils.data.Subset(train_dataset, range(args.data_cutoff))
    logging.info(f"Training on {len(train_dataset)} examples")
    logging.info(f"Validating on {len(val_dataset)} examples")
    

    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(val_dataset, num_workers=4)


    checkpoint_callback = ModelCheckpoint(
        monitor="val_word_acc", mode="max", dirpath=args.checkpoint_dir, filename="ocr"
    )
    logging.info("Starting training")
    trainer = L.Trainer(accumulate_grad_batches=1, max_epochs=args.max_epochs, enable_progress_bar=True, callbacks=[checkpoint_callback], devices=args.gpu_devices)

    trainer.fit(transcriber, train_loader, valid_loader)
    
    trainer.validate(ckpt_path="best", dataloaders=valid_loader)
    logging.info("Finished training")
    torch.save(net.state_dict(), args.output)
