import os
import shutil
import cv2

input_dir = "input2"
res_dir = "res2"
output_input_dir = "sort_input2"
output_res_dir = "sort_res2"

os.makedirs(output_input_dir, exist_ok=True)
os.makedirs(output_res_dir, exist_ok=True)

image_extensions = (".jpg", ".jpeg", ".png")

for filename in os.listdir(input_dir):
    if not filename.lower().endswith(image_extensions):
        continue

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

    # Resize mask to image if needed
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Stack side by side
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    stacked = cv2.hconcat([image, mask_colored])
    imS = cv2.resize(stacked, (960, 540))
    cv2.imshow("Image | Mask", imS)
    print(f"✅ {filename} — [y] accept | [n] skip | [q] quit")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('y'):
            shutil.copy(image_path, os.path.join(output_input_dir, filename))
            shutil.copy(mask_path, os.path.join(output_res_dir, mask_filename))
            print(f"✅ Accepted: {filename}")
            break
        elif key == ord('n'):
            print(f"⏭️ Skipped: {filename}")
            break
        elif key == ord('q'):
            print("🛑 Exiting.")
            cv2.destroyAllWindows()
            exit()
        else:
            print("❓ Press y / n / q")

cv2.destroyAllWindows()
