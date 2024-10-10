import argparse
from tqdm import tqdm
import json
import os
from torchmetrics.text import CharErrorRate, WordErrorRate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for filename in os.listdir(args.input_dir):
        with open(os.path.join(args.input_dir, filename), 'r') as f:
            data = json.load(f)
            results.append(data)

    cer = CharErrorRate()
    wer = WordErrorRate()

    for result in tqdm(results):
        if result['prediction'] != result['truth']:
            filename = os.path.basename(result['filename'])
            filename = filename.replace('.npy', '.json')
            with open(os.path.join(args.output_dir, filename), 'w') as f:
                json.dump(result, f)


    