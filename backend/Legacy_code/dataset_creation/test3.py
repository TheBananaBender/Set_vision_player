import os
import json
import cv2
import random
import shutil
from pathlib import Path

# Input folders
image_folder = "sort_input5"
json_folder = "boxes_json5"

# Output dataset structure
dataset_root = "dataset"
img_out = {
    "train": os.path.join(dataset_root, "train", "images"),
    "val": os.path.join(dataset_root, "val", "images")
}
lbl_out = {
    "train": os.path.join(dataset_root, "train", "labels"),
    "val": os.path.join(dataset_root, "val", "labels")
}

# Create output folders
for path in [*img_out.values(), *lbl_out.values()]:
    os.makedirs(path, exist_ok=True)

# Collect matching image/json pairs
samples = []
for json_file in os.listdir(json_folder):
    if not json_file.endswith(".json"):
        continue
    base_name = os.path.splitext(json_file)[0]
    for ext in [".jpg", ".png"]:
        image_path = os.path.join(image_folder, base_name + ext)
        if os.path.exists(image_path):
            samples.append((image_path, os.path.join(json_folder, json_file)))
            break

# Shuffle and split
random.shuffle(samples)
split = int(len(samples) * 0.7)
train_samples = samples[:split]
val_samples = samples[split:]

# Function to convert JSON polygon to YOLOv8-seg
def convert_and_save(image_path, json_path, label_out_path):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    with open(json_path, "r") as f:
        data = json.load(f)

    cards = data.get("cards", [])
    lines = []
    for card in cards:
        coords = []
        for x, y in card:
            coords.append(f"{x / w:.6f} {y / h:.6f}")
        lines.append("0 " + " ".join(coords))  # Class ID 0

    with open(label_out_path, "w") as f:
        f.write("\n".join(lines))

# Process all samples
for split_name, split_samples in [("train", train_samples), ("val", val_samples)]:
    for img_path, json_path in split_samples:
        base_name = Path(img_path).stem
        new_img_path = os.path.join(img_out[split_name], base_name + ".jpg")
        new_lbl_path = os.path.join(lbl_out[split_name], base_name + ".txt")

        # Convert and copy
        convert_and_save(img_path, json_path, new_lbl_path)
        shutil.copy(img_path, new_img_path)

print(f"✅ Dataset prepared: {len(train_samples)} train / {len(val_samples)} val")

# Save dataset.yaml
dataset_yaml = f"""path: {dataset_root}
train: train/images
val: val/images
names:
  0: card
"""
with open(os.path.join(dataset_root, "dataset.yaml"), "w") as f:
    f.write(dataset_yaml)

print("✅ dataset.yaml created")
