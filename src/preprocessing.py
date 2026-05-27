import json
import random
from collections import Counter

# ============================================================
# Load Dataset
# ============================================================
with open("data/raw/dataset_final_v2.json", "r") as f:
    data = json.load(f)

print(f"Total entries: {len(data)}")

# ============================================================
# Split by Papers
# ============================================================
random.seed(42)

# Group entries by paper (every 5 = 1 paper)
papers = [data[i:i+5] for i in range(0, len(data), 5)]
print(f"Total papers: {len(papers)}")

random.shuffle(papers)

train_end = int(len(papers) * 0.8)
val_end  = int(len(papers) * 0.9)

train_papers = papers[:train_end]
val_papers   = papers[train_end:val_end]
test_papers  = papers[val_end:]

train = [e for p in train_papers for e in p]
val   = [e for p in val_papers   for e in p]
test  = [e for p in test_papers  for e in p]

print(f"Train: {len(train_papers)} papers = {len(train)} entries")
print(f"Val:   {len(val_papers)}   papers = {len(val)}   entries")
print(f"Test:  {len(test_papers)}  papers = {len(test)}  entries")

# ============================================================
# Check Score Distribution
# ============================================================
for split_name, split in [("Train", train), ("Val", val), ("Test", test)]:
    scores = Counter([e["score"] for e in split])
    print(f"\n{split_name} score distribution:")
    for score in sorted(scores.keys()):
        print(f"  Score {score}: {scores[score]}")

# ============================================================
# Save Splits
# ============================================================
with open("data/splits/train.json", "w") as f:
    json.dump(train, f, indent=2)

with open("data/splits/val.json", "w") as f:
    json.dump(val, f, indent=2)

with open("data/splits/test.json", "w") as f:
    json.dump(test, f, indent=2)

print("\n✅ Saved train.json, val.json, test.json to data/")
