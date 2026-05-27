import argparse
import json
import os
import random
import yaml

from collections import Counter


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_papers(papers, entries_per_paper):
    for i, paper in enumerate(papers):
        if len(paper) != entries_per_paper:
            raise ValueError(f"Paper group {i} has {len(paper)} entries, expected {entries_per_paper}")

        scores = sorted([entry["score"] for entry in paper])
        if scores != [0, 1, 2, 3, 4]:
            raise ValueError(f"Paper group {i} has invalid score set: {scores}")


def print_distribution(name, split):
    scores = Counter([entry["score"] for entry in split])
    print(f"\n{name} score distribution:")
    for score in sorted(scores.keys()):
        print(f"  Score {score}: {scores[score]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML file"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    raw_path = config["dataset"]["raw_path"]
    train_path = config["dataset"]["train_path"]
    val_path = config["dataset"]["val_path"]
    test_path = config["dataset"]["test_path"]

    entries_per_paper = config["dataset"]["entries_per_paper"]
    split_seed = config["dataset"]["split_seed"]
    train_ratio = config["dataset"]["train_ratio"]
    val_ratio = config["dataset"]["val_ratio"]

    print("=" * 60)
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Raw dataset: {raw_path}")
    print("=" * 60)

    with open(raw_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total entries: {len(data)}")

    if len(data) % entries_per_paper != 0:
        raise ValueError(
            f"Dataset length {len(data)} is not divisible by entries_per_paper={entries_per_paper}"
        )

    papers = [
        data[i:i + entries_per_paper]
        for i in range(0, len(data), entries_per_paper)
    ]

    print(f"Total papers: {len(papers)}")

    validate_papers(papers, entries_per_paper)

    random.seed(split_seed)
    random.shuffle(papers)

    train_end = int(len(papers) * train_ratio)
    val_end = int(len(papers) * (train_ratio + val_ratio))

    train_papers = papers[:train_end]
    val_papers = papers[train_end:val_end]
    test_papers = papers[val_end:]

    train = [entry for paper in train_papers for entry in paper]
    val = [entry for paper in val_papers for entry in paper]
    test = [entry for paper in test_papers for entry in paper]

    print(f"\nTrain: {len(train_papers)} papers = {len(train)} entries")
    print(f"Val:   {len(val_papers)} papers = {len(val)} entries")
    print(f"Test:  {len(test_papers)} papers = {len(test)} entries")

    print_distribution("Train", train)
    print_distribution("Val", val)
    print_distribution("Test", test)

    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train, f, indent=2, ensure_ascii=False)

    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val, f, indent=2, ensure_ascii=False)

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test, f, indent=2, ensure_ascii=False)

    print("\nSaved splits:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")


if __name__ == "__main__":
    main()