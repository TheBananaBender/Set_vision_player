import os
from PIL import Image
from lang_sam import LangSAM
import cv2
import numpy as np

input_dir = "input_failed"
base_output_dir = "res_failed"
os.makedirs(base_output_dir, exist_ok=True)

# Create subfolders for different threshold values
x_values = np.round(np.arange(0.05, 1.05, 0.05), 2)  # 0.05, 0.1, ..., 0.95, 1
output_dirs = {}
for x in x_values:
    folder_name = f"x_{x:.2f}".replace('.', '_')
    folder_path = os.path.join(base_output_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    output_dirs[x] = folder_path

model = LangSAM()
image_extensions = (".jpg", ".jpeg", ".png")
cnt = 0

for filename in os.listdir(input_dir):
    if not filename.lower().endswith(image_extensions):
        continue

    cnt += 1
    print(f"\n📷 Image {cnt}: {filename}")
    image_path = os.path.join(input_dir, filename)
    image_name = os.path.splitext(filename)[0]

    image_pil = Image.open(image_path).convert("RGB")
    results = model.predict(
        images_pil=[image_pil],
        texts_prompt=["White Cards"],
        box_threshold=0.2,
        text_threshold=0.15,
    )

    if not results or "masks" not in results[0] or results[0]["masks"].size == 0:
        print(f" No masks found for {filename}")
        continue

    all_masks = results[0]["masks"]
    scores = results[0]["scores"]
    median = np.median(scores)
    std = np.std(scores)

    for x in x_values:
        binary_mask = np.where(np.abs(scores - median) <= x * std, 1, 0).astype(np.uint8)
        filtered_masks = [all_masks[i] for i in range(len(all_masks)) if binary_mask[i]]
        if not filtered_masks:
            print(f" No masks passed filtering for x={x:.2f} on {filename}")
            continue

        final_mask = np.any(filtered_masks, axis=0).astype(np.uint8) * 255
        output_path = os.path.join(output_dirs[x], f"{image_name}_mask.png")
        cv2.imwrite(output_path, final_mask)
        print(f"Saved for x={x:.2f}: {output_path}")
