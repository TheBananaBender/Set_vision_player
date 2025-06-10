import cv2
import numpy as np
import time
import torch
import torch.nn.functional as F
from torchvision import models
import torch.nn as nn
from torchvision.transforms import v2
import Set_game_mechanics


class MultiHeadMobileNetV3(nn.Module):
    """
    Multi-head MobileNetV3 model for SET card detection.
    This model uses the MobileNetV3 architecture and has four heads for predicting each attribute of the SET card:
    color, shape, number, and shading.
    Each head outputs a probability distribution over three classes for its respective attribute.
    The model is designed to be lightweight and efficient, suitable for real-time applications.
    """
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v3_large(weights=None)
        self.features = base.features
        self.pool = base.avgpool
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)
        in_features = base.classifier[0].in_features
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


def order_box_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def detect_cards_in_frame(frame):
    """"
    ###IRRELEVANT method ###
    Detect SET cards in a given frame using contour detection. 
    with classical image processing methods
    Returns a list of bounding boxes for detected cards.
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
    dst_pts = np.array([
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1],
        [0, output_size[1] - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(np.float32(box), dst_pts)
    return cv2.warpPerspective(image, M, output_size)

def preprocess_for_model(img):
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((256, 256), antialias=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])

    ])
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = transform(img_rgb)
    return pil_img.unsqueeze(0)



def decode_prediction(preds):
    color_map = ['Red', 'Green', 'Purple']
    shape_map = ['Diamond', 'Squiggle', 'Oval']
    number_map = ['One', 'Two', 'Three']
    shading_map = ['Solid', 'Striped', 'Open']
    c, s, n, sh = [torch.argmax(p, dim=1).item() for p in preds]
    return f"{color_map[c]} {shape_map[s]} {number_map[n]} {shading_map[sh]}"


def predictions_to_set_cards(preds):
    """
    Convert model predictions to a list of SET card objects.
    arguments:
    preds: tuple of tensors (color, shape, number, shading)
    where each tensor has shape (batch_size, 3)
    """
    COLOR = {0: "Red", 1: "Green", 2: "Purple"}
    NUMBER = {0: "One", 1: "Two", 2: "Three"}
    SHADING = {0: "Solid", 1: "Striped", 2: "Open"}
    SHAPE = {0: "Diamond", 1: "Squiggle", 2: "Oval"}
    color, shape, number, shading = preds
    cards = []
    for i in range(color.shape[0]):
        c = color[i].argmax().item()
        s = shape[i].argmax().item()
        n = number[i].argmax().item()
        sh = shading[i].argmax().item()
        cards.append(Set_game_mechanics.card(
            color=COLOR[c],
            number=NUMBER[n],
            shading=SHADING[sh],
            shape=SHAPE[s]
        ))
    return cards



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiHeadMobileNetV3().to(device)
    model.load_state_dict(torch.load("mobilenetv3_set_card.pth", map_location=device))
    model.eval()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    last_prediction_time = 0
    predictions = []  # list of tuples: (box, label)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        boxes = detect_cards_in_frame(frame)
        updated_predictions = []

        if current_time - last_prediction_time >= 1.0:
            for box in boxes:
                cropped = warp_card(frame, box)
                inp = preprocess_for_model(cropped).to(device)
                with torch.no_grad():
                    preds = model(inp)
                label = decode_prediction(preds)
                updated_predictions.append((box, label))

            predictions = updated_predictions
            last_prediction_time = current_time

        # Draw boxes and labels (use latest predictions even if not updated)
        for box, label in predictions:
            cv2.polylines(frame, [box], isClosed=True, color=(0, 255, 0), thickness=2)
            cx, cy = np.mean(box, axis=0).astype(int)
            cv2.putText(frame, label, (cx - 60, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.imshow("SET Card Detection + Prediction", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
