import argparse
from termcolor import colored
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import CharErrorRate, WordErrorRate
import logging
from utils import LineImageDataset, MyNN, highlight_differences
from tqdm import tqdm
import json
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--lines_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--train_proportion", type=float, default=0.9)
    parser.add_argument("--gpu_devices", nargs="+", default=[0])
    args = parser.parse_args()

    logger_name = args.output_dir + ".log"
    logging.basicConfig(filename=logger_name, level=logging.INFO)
    logger = logging.getLogger(__name__)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "train"), exist_ok=True)

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    if not torch.cuda.is_available():
        raise ValueError("CUDA not available. Exiting.")
    else:
        for device_idx in args.gpu_devices:
            device = torch.device(f"cuda:{device_idx}")
            free_mem, total_mem = torch.cuda.mem_get_info(device)
            used_MB = (total_mem - free_mem)/1024**2
            if used_MB > 500:
                raise ValueError(f"Device {device_idx} is not empty. Exiting.")


    all_chars = " -.ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuvwxyz¶"
    char_to_num = {char: idx + 1 for idx, char in enumerate(all_chars)}
    num_to_char = {idx + 1: char for idx, char in enumerate(all_chars)}

    val_transform = transforms.Compose([
        transforms.Normalize(0.15, 0.38)
    ])

    dataset = LineImageDataset(
        dirname=args.input_dir,
        lines_dir=args.lines_dir,
        char_to_num=char_to_num,
        num_to_char=num_to_char,
        data_type="all",
        transform=val_transform,
        return_filenames=True
    )

    train_size = int(args.train_proportion * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    val_loader = DataLoader(val_dataset, shuffle=False, num_workers=4)
    train_loader = DataLoader(train_dataset, shuffle=False, num_workers=4)
    splits = {
        "train": train_loader,
        "val": val_loader
    }

    model = MyNN()
    model.load_state_dict(torch.load(args.model))
    model.eval()

    # only one device needed
    device = torch.device(f"cuda:{args.gpu_devices[0]}")
    model.to(device)

    all_predictions = []
    all_truths = []
    all_images = []
    all_filenames = []

    for split_name, split_loader in splits.items():
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(split_loader)):

                images = batch['image'].to(device)
                targets = batch['target']
                texts = batch['text']
                filename = batch['filename']

                outputs = model(images)

                outputs = outputs.permute(2, 0, 1)
                decoded_outputs = torch.argmax(outputs, dim=2).squeeze(1).cpu().numpy()

                predictions = ""
                prev_char = None
                for idx in decoded_outputs:
                    if idx != 0 and idx != prev_char:
                        predictions += num_to_char.get(idx, '')
                    prev_char = idx

                truth = texts[0]
                filename = filename[0]

                json_filename = filename.split('/')[-1]
                json_filename = json_filename.split('.')[0] + ".json"
                # join output dir with a dir that has the split name
                json_filename = os.path.join(args.output_dir, split_name, json_filename)
                with open(json_filename, 'w+') as f:
                    json.dump({
                        "filename": filename,
                        "truth": truth,
                        "prediction": predictions
                    }, f)

            
        

