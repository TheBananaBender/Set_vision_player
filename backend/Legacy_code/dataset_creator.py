import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import v2

# -------------------------------
# 1. Define the classifier model
# -------------------------------
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

# ------------------------------------
# 2. Utility functions for detection
# ------------------------------------
def order_box_points(pts):
    """
    Given 4 points of a rotated rectangle (unordered), return them in order:
    top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]       # top-left
    rect[2] = pts[np.argmax(s)]       # bottom-right
    rect[1] = pts[np.argmin(diff)]    # top-right
    rect[3] = pts[np.argmax(diff)]    # bottom-left
    return rect

def detect_cards_in_frame(frame):
    """
    Detects quadrilateral contours that correspond to cards in the frame.
    Returns a list of 4×2 integer numpy arrays (boxes).
    """
    height, width = frame.shape[:2]
    scale = 600.0 / width
    resized = cv2.resize(frame, (600, int(height * scale)))
    ratio = width / 600.0

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    card_boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 2000:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            rect = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rect)
            box = order_box_points(np.array(box, dtype="float32")) * ratio
            w, h = rect[1]
            if w == 0 or h == 0:
                continue
            ar = max(w, h) / min(w, h)
            if ar < 1.1 or ar > 2.0:
                continue
            card_boxes.append(box.astype(int))
    return card_boxes

def warp_card(image, box, output_size=(256, 256)):
    """
    Given a BGR image and a 4×2 box, warp the quadrilateral to a rectangle of size output_size.
    """
    dst_pts = np.array([
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1],
        [0, output_size[1] - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(np.float32(box), dst_pts)
    return cv2.warpPerspective(image, M, output_size)

def preprocess_for_model(img):
    """
    Convert a cropped BGR image (256×256) to a normalized tensor for the model.
    """
    transform = v2.Compose([
        v2.ToImage(),                           # BGR→CHW float in [0,255]
        v2.ToDtype(torch.float32, scale=True),  # [0,1]
        v2.Resize((256, 256), antialias=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transform(img_rgb)
    return tensor.unsqueeze(0)  # shape (1,3,256,256)

def decode_prediction(preds):
    """
    Given a tuple of four logits tensors, each shape (1,3), return a string label.
    """
    color_map   = ['Red', 'Green', 'Purple']
    shape_map   = ['Oval', 'Diamond', 'Squiggle']
    number_map  = ['One', 'Two', 'Three']
    shading_map = ['Open', 'Stripe', 'Solid']
    c, s, n, sh = [torch.argmax(p, dim=1).item() for p in preds]
    return f"{color_map[c]}_{shape_map[s]}_{number_map[n]}_{shading_map[sh]}"

# ----------------------------------------
# 3. Define the sequence of all 81 cards
# ----------------------------------------
colors   = ['Red', 'Green', 'Purple']
shapes   = ['Oval', 'Diamond', 'Squiggle']
numbers  = ['One', 'Two', 'Three']
shadings = ['Open', 'Stripe', 'Solid']

card_sequence = []
for color in colors:
    for shape in shapes:
        for number in numbers:
            for shading in shadings:
                card_sequence.append((color, shape, number, shading))

# --------------------------------------------
# 4. Main loop: capture, detect, crop, save
# --------------------------------------------
def main():
    # 4.1 Create directories for saving if they don’t exist
    os.makedirs("collected_cards", exist_ok=True)

    # 4.2 Initialize camera and model
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiHeadEfficientNet().to(device)
    model.load_state_dict(torch.load("set_card_model.pth", map_location=device))
    model.eval()

    idx = 0  # index in card_sequence
    counters = {card: 0 for card in card_sequence}
    predictions = []  # to store last predictions (unused in dataset collection)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Make a pristine copy for cropping—do not draw on this one:
        raw = frame.copy()

        # Now detect and draw on a separate display‐frame:
        display_frame = frame.copy()
        boxes = detect_cards_in_frame(display_frame)

        # Draw green boxes onto display_frame
        for box in boxes:
            cv2.polylines(display_frame, [box], True, (0, 255, 0), 2)

        # Overlay your UI text on display_frame
        target = card_sequence[idx]
        cv2.putText(display_frame,
                    f"Target: {target[0]} {target[1]} {target[2]} {target[3]}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(display_frame,
                    "Press 's' to Save, 'n' to Next, 'q' to Quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Dataset Collection", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('n'):
            idx = (idx + 1) % len(card_sequence)

        elif key == ord('s'):
            if len(boxes) == 0:
                print("No card detected to save.")
                continue

            # Use raw (un‐drawn) for cropping:
            for box in boxes:
                cropped = warp_card(raw, box)  # raw instead of display_frame
                label = "_".join(target)
                counters[target] += 1
                count = counters[target]
                filename = f"{label}_{count}.png"
                save_path = os.path.join("collected_cards", filename)
                cv2.imwrite(save_path, cropped)
                print(f"Saved: {save_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
