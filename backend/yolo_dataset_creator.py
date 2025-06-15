import os
import json
import cv2
import numpy as np

input_dir = "sort_input5"
mask_dir = "sort_res5"
output_json_dir = "boxes_json5"
os.makedirs(output_json_dir, exist_ok=True)

image_extensions = (".jpg", ".jpeg", ".png")

for filename in os.listdir(input_dir):
    if not filename.lower().endswith(image_extensions):
        continue

    base_name = os.path.splitext(filename)[0]
    mask_filename = f"{base_name}_mask.png"
    image_path = os.path.join(input_dir, filename)
    mask_path = os.path.join(mask_dir, mask_filename)

    if not os.path.exists(mask_path):
        print(f"❌ Mask not found for {filename}")
        continue

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"⚠️ Failed to load mask: {mask_path}")
        continue

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    card_corners = []

    for contour in contours:
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            # Found a quadrilateral
            corners = [[int(p[0][0]), int(p[0][1])] for p in approx]
            card_corners.append(corners)
        else:
            # If not 4 points, try to get minAreaRect as fallback
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)  # returns 4 corner points
            box = np.int0(box)
            corners = [[int(p[0]), int(p[1])] for p in box]
            card_corners.append(corners)

    json_data = {
        "image": filename,
        "cards": card_corners
    }

    json_path = os.path.join(output_json_dir, f"{base_name}.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"✅ Saved {json_path} with {len(card_corners)} cards.")
