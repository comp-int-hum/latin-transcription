import argparse
from termcolor import colored
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import CharErrorRate, WordErrorRate
import argparse
import difflib
from utils import LineImageDataset, MyNN, LatinTranscriber


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="work/source")
    parser.add_argument("--lines_dir", type=str, default="work/output")
    parser.add_argument("--model_path", type=str, default="work/model.pkl")
    args = parser.parse_args()

    # Load the character mappings
    all_chars = " -.ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuvwxyz¶"
    char_to_num = {char: idx + 1 for idx, char in enumerate(all_chars)}
    num_to_char = {idx + 1: char for idx, char in enumerate(all_chars)}

    # Define the same transforms used during validation
    val_transform = transforms.Compose([
        transforms.Normalize(0.15, 0.38)
    ])

    # Load the validation dataset
    val_dataset = LineImageDataset(
        dirname=args.input_dir,
        char_to_num=char_to_num,
        num_to_char=num_to_char,
        data_type="all",  # Use 'all' or 'val' depending on your split
        transform=val_transform
    )

    val_size = int(0.1 * len(val_dataset))
    # TODO make sure to use the same split as during training
    val_indices = torch.utils.data.Subset(val_dataset, range(val_size))

    val_loader = DataLoader(val_indices, batch_size=1, shuffle=False, num_workers=4)

    # Load the trained model
    model = MyNN()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize error metrics
    cer_metric = CharErrorRate()
    wer_metric = WordErrorRate()

    # Lists to store results for analysis
    all_predictions = []
    all_truths = []
    all_images = []

    # Iterate over the validation data
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            images = batch['image'].to(device)
            targets = batch['target']
            texts = batch['text']

            # Forward pass
            outputs = model(images)

            # Get predictions
            outputs = outputs.permute(2, 0, 1)  # (seq_len, batch, num_classes)
            decoded_outputs = torch.argmax(outputs, dim=2).squeeze(1).cpu().numpy()

            # Decode predictions and targets
            predictions = ""
            prev_char = None
            for idx in decoded_outputs:
                if idx != 0 and idx != prev_char:
                    predictions += num_to_char.get(idx, '')
                prev_char = idx

            truth = texts[0]  # Since batch_size=1

            # Update metrics
            cer_metric.update([predictions], [truth])
            wer_metric.update([predictions], [truth])

            # Store for analysis
            all_predictions.append(predictions)
            all_truths.append(truth)
            all_images.append(images.cpu())

            # Optional: Print progress
            if batch_idx % 100 == 0:
                print(f"Processed {batch_idx} samples")

    # Compute overall metrics
    cer = cer_metric.compute()
    wer = wer_metric.compute()
    print(f"Character Error Rate (CER): {cer:.4f}")
    print(f"Word Error Rate (WER): {wer:.4f}")

    # Error analysis
    # Identify misclassified samples
    misclassified_indices = [
        i for i, (pred, truth) in enumerate(zip(all_predictions, all_truths)) if pred != truth
    ]

    print(f"Total misclassified samples: {len(misclassified_indices)}")

    # Function to highlight differences between prediction and truth
    def highlight_differences(pred, truth, level='char'):
        if level == 'char':
            seqm = difflib.SequenceMatcher(None, truth, pred)
            output = []
            for opcode, a0, a1, b0, b1 in seqm.get_opcodes():
                if opcode == 'equal':
                    output.append(truth[a0:a1])
                elif opcode == 'insert':
                    inserted_text = pred[b0:b1]
                    output.append(colored(inserted_text, 'green', attrs=['bold', 'underline']))
                elif opcode == 'delete':
                    deleted_text = truth[a0:a1]
                    output.append(colored(deleted_text, 'red', attrs=['bold', 'underline']))
                elif opcode == 'replace':
                    replaced_text = pred[b0:b1]
                    output.append(colored(replaced_text, 'yellow', attrs=['bold', 'underline']))
            return ''.join(output)
        elif level == 'word':
            truth_words = truth.split()
            pred_words = pred.split()
            seqm = difflib.SequenceMatcher(None, truth_words, pred_words)
            output = []
            for opcode, a0, a1, b0, b1 in seqm.get_opcodes():
                if opcode == 'equal':
                    output.extend(truth_words[a0:a1])
                elif opcode == 'insert':
                    inserted_words = ' '.join(pred_words[b0:b1])
                    output.append(colored(inserted_words, 'green', attrs=['bold', 'underline']))
                elif opcode == 'delete':
                    deleted_words = ' '.join(truth_words[a0:a1])
                    output.append(colored(deleted_words, 'red', attrs=['bold', 'underline']))
                elif opcode == 'replace':
                    replaced_words = ' '.join(pred_words[b0:b1])
                    output.append(colored(replaced_words, 'yellow', attrs=['bold', 'underline']))
            return ' '.join(output)
        else:
            raise ValueError("Invalid level: choose 'char' or 'word'")

    # Display misclassified examples with highlighted differences
    for idx in misclassified_indices:
        image = all_images[idx][0][0]  # Get the image tensor
        pred = all_predictions[idx]
        truth = all_truths[idx]

        # Highlight differences
        highlighted_pred_char = highlight_differences(pred, truth, level='char')
        highlighted_pred_word = highlight_differences(pred, truth, level='word')

        # Print results
        print(f"\nSample Index: {idx}")
        print(f"Ground Truth: {truth}")
        print(f"Prediction (char-level differences): {highlighted_pred_char}")
        print(f"Prediction (word-level differences): {highlighted_pred_word}")

        # Display the image
        plt.figure(figsize=(10, 4))
        plt.imshow(image, cmap='gray')
        plt.title(f"Sample Index: {idx}")
        plt.axis('off')
        plt.show()

