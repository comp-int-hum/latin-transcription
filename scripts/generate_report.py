import argparse
from tqdm import tqdm
import json
import os
from torchmetrics.text import CharErrorRate, WordErrorRate


# Generate report:
# Generates a report of the character error rate and word error rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    results = []
    for filename in os.listdir(args.input_dir):
        with open(os.path.join(args.input_dir, filename), 'r') as f:
            results.append(json.load(f))

    cer = CharErrorRate()
    wer = WordErrorRate()

    for result in tqdm(results):
        cer.update(result['prediction'], result['truth'])
        wer.update(result['prediction'], result['truth'])

    character_error_rate = float(cer.compute())
    word_error_rate = float(wer.compute())

    print(f"Character error rate: {character_error_rate}")
    print(f"Word error rate: {word_error_rate}")

    with open(args.output, 'w') as f:
        json.dump({
            'character_error_rate': character_error_rate,
            'word_error_rate': word_error_rate
        }, f)


    