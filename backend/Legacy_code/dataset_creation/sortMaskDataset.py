import os
import shutil
import cv2
import numpy as np

input_dir = "input7"
res_dir = "res7"
output_input_dir = "sort_input7"
output_res_dir = "sort_res7"

os.makedirs(output_input_dir, exist_ok=True)
os.makedirs(output_res_dir, exist_ok=True)

image_extensions = (".jpg", ".jpeg", ".png")
all_images = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)])
total = len(all_images)

# List all res_failed/x_XX subfolders
threshold_folders = sorted([
    os.path.join(res_dir, d) for d in os.listdir(res_dir)
    if os.path.isdir(os.path.join(res_dir, d)) and d.startswith("x_")
])

for idx, filename in enumerate(all_images, start=1):
    base_name = os.path.splitext(filename)[0]
    image_path = os.path.join(input_dir, filename)
    image = cv2.imread(image_path)

    if image is None:
        print(f"⚠️ Failed to load image: {image_path}")
        continue

    print(f"\n📷 [{idx}/{total}] {filename} — Review masks:")

    # Go over all masks for this image in different threshold folders
    for folder in threshold_folders:
        mask_path = os.path.join(folder, f"{base_name}_mask.png")
        if not os.path.exists(mask_path):
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        stacked = cv2.hconcat([image, mask_colored])
        imS = cv2.resize(stacked, (960, 540))

        cv2.imshow("Image | Mask", imS)
        print(f"🔎 From: {folder} — [y] accept | [n] skip | [b] black mask | [q] quit")

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('y'):
                shutil.copy(image_path, os.path.join(output_input_dir, filename))
                shutil.copy(mask_path, os.path.join(output_res_dir, f"{base_name}_mask.png"))
                print(f"✅ Accepted: {filename}")
                break  # Move to next image
            elif key == ord('b'):
                black_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.imwrite(os.path.join(output_res_dir, f"{base_name}_mask.png"), black_mask)
                shutil.copy(image_path, os.path.join(output_input_dir, filename))
                print(f"⬛ Black mask assigned to: {filename}")
                break  # Move to next image
            elif key == ord('n'):
                print("⏭️ Skipping this mask...")
                break  # Try next mask for the same image
            elif key == ord('q'):
                print("🛑 Quitting.")
                cv2.destroyAllWindows()
                exit()
            else:
                print("❓ Press y / n / b / q")

        if key in [ord('y'), ord('b')]:  # If we accepted or assigned black mask — skip remaining masks
            break

cv2.destroyAllWindows()
