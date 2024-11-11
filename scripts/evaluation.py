import argparse
from tqdm import tqdm
import json
import os
from torchmetrics.text import CharErrorRate, WordErrorRate
import jiwer
import Levenshtein
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd


# Error evaluation:
# Evaluates errors across train and test datasets

if __name__ == "__main__":
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_results_dir", type=str, required=True)
    parser.add_argument("--val_results_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    train_results = args.train_results_dir
    val_results = args.val_results_dir

    results = {"train": [], "val": []}

    for (directory, name) in [(train_results, "train"), (val_results, "val")]:
        for filename in os.listdir(directory):
            with open(os.path.join(directory, filename), 'r') as f:
                results[name].append(json.load(f))
    
    word_error_stats = {"train": {}, "val": {}}

    for directory in results.keys():
        for i in tqdm(range(len(results[directory]))):
            reference = transformation(results[directory][i]["truth"])
            hypothesis = transformation(results[directory][i]["prediction"])
            truth_words = reference.split(" ")
            comparison = jiwer.process_words(reference, hypothesis)
            for alignment in comparison.alignments[0]:
                for index in range(alignment.ref_start_idx, alignment.ref_end_idx):
                    word_ = truth_words[index]
                    type_ = alignment.type
                    if word_ not in word_error_stats[directory]:
                        word_error_stats[directory][word_] = {}
                    if type_ not in word_error_stats[directory][word_]:
                        word_error_stats[directory][word_][type_] = 0
                    if "total" not in word_error_stats[directory][word_]:
                        word_error_stats[directory][word_]["total"] = 0
                    word_error_stats[directory][word_][type_] += 1
                    word_error_stats[directory][word_]["total"] += 1

    val_words = set(word_error_stats["val"].keys())
    train_words = set(word_error_stats["train"].keys())
    val_not_in_train = val_words - train_words

    val_df = pd.DataFrame.from_dict(word_error_stats["val"], orient='index')
    train_df = pd.DataFrame.from_dict(word_error_stats["train"], orient='index')
    val_not_in_train_df = pd.DataFrame.from_dict({word: word_error_stats["val"][word] for word in val_not_in_train}, orient='index')

    val_df["accuracy"] = val_df["equal"] / val_df["total"]
    train_df["accuracy"] = train_df["equal"] / train_df["total"]
    val_not_in_train_df["accuracy"] = val_not_in_train_df["equal"] / val_not_in_train_df["total"]

    total_errors_train = train_df["substitute"].sum() + train_df["delete"].sum()
    total_errors_val = val_df["substitute"].sum() + val_df["delete"].sum()
    total_errors_val_not_in_train = val_not_in_train_df["substitute"].sum() + val_not_in_train_df["delete"].sum()

    result_json = {
        "total_errors_train": total_errors_train,
        "total_errors_val": total_errors_val,
        "total_errors_val_not_in_train": total_errors_val_not_in_train,
        "train_accuracy": 1 - total_errors_train/train_df["total"].sum(),
        "val_accuracy": 1 - total_errors_val/val_df["total"].sum(),
        "val_not_in_train_accuracy": 1 - total_errors_val_not_in_train/val_not_in_train_df["total"].sum(),
        "percentage_of_errors_val_not_in_train": total_errors_val_not_in_train/total_errors_val,
    }
    path = "/".join(args.output.split("/")[:-1])
    plt.scatter(val_df["accuracy"], val_df["total"])
    plt.xlabel("accuracy")
    plt.ylabel("total")
    plt.savefig(path + "/val_accuracy_vs_total.png")
    plt.close()

    plt.scatter(train_df["accuracy"], train_df["total"])
    plt.xlabel("accuracy")
    plt.ylabel("total")
    plt.savefig(path + "/train_accuracy_vs_total.png")
    plt.close()

    distance_mapping = {}
    for word in val_not_in_train:
        distances = {train_word: Levenshtein.distance(word, train_word) for train_word in train_words}
        closest_word = min(distances, key=distances.get)
        closest_distance = distances[closest_word]
        if closest_distance not in distance_mapping:
            distance_mapping[closest_distance] = {"total": 0, "correct": 0}
        distance_mapping[closest_distance]["total"] += val_not_in_train_df.loc[word]["total"].item()
        equal = val_not_in_train_df.loc[word]["equal"].item()
        if not math.isnan(equal):
            distance_mapping[closest_distance]["correct"] += equal

    word_distances = list(distance_mapping.keys())
    word_accuracies = [distance_mapping[distance]["correct"] / distance_mapping[distance]["total"] for distance in word_distances]
    plt.scatter(word_distances, word_accuracies)
    plt.xlabel("Levenstein distance")
    plt.ylabel("accuracy")
    plt.savefig(path + "/val_not_in_train_accuracy_vs_distance.png")
    plt.close()
    with open(args.output, 'w') as f:
        json.dump(result_json, f)




