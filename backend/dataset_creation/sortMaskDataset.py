import os
import shutil
import cv2
import numpy as np

input_dir = "input5"
res_dir = "res5"
output_input_dir = "sort_input5"
output_res_dir = "sort_res5"

os.makedirs(output_input_dir, exist_ok=True)
os.makedirs(output_res_dir, exist_ok=True)

image_extensions = (".jpg", ".jpeg", ".png")

# Filter and sort image files for consistent order
all_images = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]
all_images.sort()
total = len(all_images)

for idx, filename in enumerate(all_images, start=1):
    base_name = os.path.splitext(filename)[0]
    mask_filename = f"{base_name}_mask.png"

    image_path = os.path.join(input_dir, filename)
    mask_path = os.path.join(res_dir, mask_filename)

    if not os.path.exists(mask_path):
        print(f"❌ No mask found for: {filename}")
        continue

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"⚠️ Failed to load image: {image_path}")
        continue
    if mask is None:
        print(f"⚠️ Failed to load mask: {mask_path}")
        continue

    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    stacked = cv2.hconcat([image, mask_colored])
    imS = cv2.resize(stacked, (960, 540))
    cv2.imshow("Image | Mask", imS)
    print(f"✅ {idx}/{total}: {filename} — [y] accept | [n] skip | [b] black mask | [q] quit")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('y'):
            shutil.copy(image_path, os.path.join(output_input_dir, filename))
            shutil.copy(mask_path, os.path.join(output_res_dir, mask_filename))
            print(f"✅ Accepted: {filename}")
            break
        elif key == ord('b'):
            black_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.imwrite(os.path.join(output_res_dir, mask_filename), black_mask)
            shutil.copy(image_path, os.path.join(output_input_dir, filename))
            print(f"⬛ Black mask assigned to: {filename}")
            break
        elif key == ord('n'):
            print(f"⏭️ Skipped: {filename}")
            break
        elif key == ord('q'):
            print("🛑 Exiting.")
            cv2.destroyAllWindows()
            exit()
        else:
            print("❓ Press y / n / b / q")

cv2.destroyAllWindows()
