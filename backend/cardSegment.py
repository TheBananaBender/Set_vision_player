import os
from PIL import Image
from lang_sam import LangSAM
import cv2
import numpy as np

# Define base folders
input_dir = "input6"
output_dir = "res6"

# === Create the output folder ===
os.makedirs(output_dir, exist_ok=True)

# Initialize model
model = LangSAM()

# Supported image formats
image_extensions = (".jpg", ".jpeg", ".png")
cnt = 0

# Loop over all images
for filename in os.listdir(input_dir):
    if not filename.lower().endswith(image_extensions):
        continue

    cnt += 1
    print("\nImage number:", cnt)

    image_path = os.path.join(input_dir, filename)
    image_name = os.path.splitext(filename)[0]
    print(f"🔍 Processing: {filename}")

    # Load image and run prediction
    image_pil = Image.open(image_path).convert("RGB")
    results = model.predict(
        images_pil=[image_pil],
        texts_prompt=["White Cards"],
        box_threshold=0.2,
        text_threshold=0.15,
    )

    # If no masks returned, skip
    if not results or "masks" not in results[0] or results[0]["masks"].size == 0:
        print(f"⚠️  No masks found for {filename}")
        continue

    scores = results[0]["scores"]
    median = np.median(scores)
    std = np.std(scores)
    binary_mask = np.where(np.abs(scores - median) <= std, 1, 0).astype(np.uint8)

    all_masks = results[0]["masks"]  # shape: (N, H, W)
    filtered_masks = [all_masks[i] for i in range(len(all_masks)) if binary_mask[i]]
    if not filtered_masks:
        print(f"⚠️  No masks passed filtering for {filename}")
        continue

    final_mask = np.any(filtered_masks, axis=0).astype(np.uint8) * 255

    output_path = os.path.join(output_dir, f"{image_name}_mask.png")
    cv2.imwrite(output_path, final_mask)
    print(f"✅ Saved: {output_path}")
