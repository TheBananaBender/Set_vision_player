import os
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
import torch.nn as nn
from torchvision.transforms import v2

# === Label maps ===
inv_COLOR   = {0: 'Red',    1: 'Green',   2: 'Purple'}
inv_SHAPE   = {0: 'Diamond',1: 'Squiggle',2: 'Oval'}
inv_NUMBER  = {0: 'One',    1: 'Two',     2: 'Three'}
inv_SHADING = {0: 'Solid',  1: 'Striped', 2: 'Open'}

# === Multi-head model ===
class MultiHeadEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.efficientnet_b0(weights=None)
        self.features = base.features
        self.pool = base.avgpool
        self.flatten = nn.Flatten()
        self.dropout = base.classifier[0]
        in_features = base.classifier[1].in_features
        self.head_color   = nn.Linear(in_features, 3)
        self.head_shape   = nn.Linear(in_features, 3)
        self.head_number  = nn.Linear(in_features, 3)
        self.head_shading = nn.Linear(in_features, 3)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return (
            self.head_color(x),
            self.head_shape(x),
            self.head_number(x),
            self.head_shading(x)
        )

# === Preprocessing for the model ===
preprocess = v2.Compose([
    v2.Resize((256, 256), antialias=True),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
])

#detect_cards_opencv
def adaptive_blur(gray):
    stddev = np.std(gray)
    print(f"[DEBUG] Image stddev: {stddev:.2f}")
    meandev = np.mean(gray)
    print(f"[DEBUG] Image meandev: {meandev:.2f}")

    if stddev < 10:
        ksize = 3
    elif stddev < 25:
        ksize = 9
    elif stddev < 35:
        ksize = 15
    elif stddev < 35:
        ksize = 33
    else:
        ksize = 43

    if ksize % 2 == 0:
        ksize += 1

    return cv2.GaussianBlur(gray, (ksize, ksize), 0)

def detect_cards_opencv(image, debug=True, scale=0.5):
    def show(title, img):
        if debug:
            h, w = img.shape[:2]
            resized = cv2.resize(img, (int(w * scale), int(h * scale)))
            cv2.imshow(title, resized)

    # Step 1: Convert to grayscale and apply adaptive blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = adaptive_blur(gray)
    show("Blured", blur)
    normalized_image = cv2.normalize(blur, None, alpha=50, beta=200, norm_type=cv2.NORM_MINMAX)
    show("normalized_image", normalized_image)

    # Step 2: Canny edge detection
    edges = cv2.Canny(normalized_image, 5, 25)
    show("1 - Canny", edges)


    # Step 3: Dilate to connect lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    show("2 - Dilated", dilated)

    # Step 4: Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"[INFO] Contours found: {len(contours)}")

    card_boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000 or area > 500000:
            continue

        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) < 4 or len(approx) > 12:
            continue

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        w, h = rect[1]
        if w == 0 or h == 0:
            continue

        aspect = max(w, h) / min(w, h)
        if aspect < 1 or aspect > 3.0:
            continue

        if w * h < 5000:  # reject small interior shapes
            continue

        card_boxes.append(box)

    # Step 5: Draw final results
    final = image.copy()
    for box in card_boxes:
        cv2.polylines(final, [box], True, (0, 255, 0), 2)
    show("3 - Final Detected Cards", final)

    if debug:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return card_boxes




# === Classify single card (warped) ===
def classify_card(cropped_bgr_image, model, device):
    image_pil = Image.fromarray(cv2.cvtColor(cropped_bgr_image, cv2.COLOR_BGR2RGB))
    tensor = preprocess(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out_color, out_shape, out_number, out_shading = model(tensor)
    pred = (
        torch.argmax(out_color, dim=1).item(),
        torch.argmax(out_shape, dim=1).item(),
        torch.argmax(out_number, dim=1).item(),
        torch.argmax(out_shading, dim=1).item()
    )
    return f"{inv_COLOR[pred[0]]}_{inv_SHAPE[pred[1]]}_{inv_NUMBER[pred[2]]}_{inv_SHADING[pred[3]]}"

# === Process full image ===
def process_image(image_path, model, device):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    output = image.copy()
    boxes = detect_cards_opencv(image)

    for box in boxes:
        src_pts = np.array(box, dtype="float32")
        dst_pts = np.array([[0, 0], [255, 0], [255, 255], [0, 255]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(output, M, (256, 256))
        cv2.imshow("Warped Card", warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        label = classify_card(warped, model, device)

        # Draw box and label
        cv2.polylines(output, [box], isClosed=True, color=(0, 255, 0), thickness=2)
        x, y = box[0]
        cv2.putText(output, label, (int(x), int(y)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show result
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Detected SET Cards")
    plt.show()

# === Main function ===
if __name__ == '__main__':
    image_path = "C:\\Users\\galha\\Downloads\\picfortest.jpg"  # 🔁 CHANGE to your image path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiHeadEfficientNet()
    model.load_state_dict(torch.load("set_card_model.pth", map_location=device))
    model.to(device)
    model.eval()

    process_image(image_path, model, device)
