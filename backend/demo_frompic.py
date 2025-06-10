import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from torchvision import models
import torch.nn as nn
from torchvision.transforms import v2
import matplotlib.pyplot as plt


# --- הגדרת המודל לסיווג ---
class MultiHeadMobileNetV3(nn.Module):
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


# --- פונקציות עזר ---
def order_box_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

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
    return transform(img_rgb).unsqueeze(0)

def decode_prediction(preds):
    color_map = ['Red', 'Green', 'Purple']
    shape_map = ['Diamond', 'Squiggle', 'Oval']
    number_map = ['One', 'Two', 'Three']
    shading_map = ['Solid', 'Striped', 'Open']
    c, s, n, sh = [torch.argmax(p, dim=1).item() for p in preds]
    return f"{color_map[c]} {shape_map[s]} {number_map[n]} {shading_map[sh]}"


# --- פונקציה ראשית ---
def main():
    image_path = './input2/photo_46_2025-06-08_18-34-19.jpg'
    yolo_model_path = './best.pt'
    classifier_model_path = './mobilenetv3_set_card.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    yolo_model = YOLO(yolo_model_path)
    classifier = MultiHeadMobileNetV3().to(device)
    classifier.load_state_dict(torch.load(classifier_model_path, map_location=device))
    classifier.eval()

    # Load image
    image_pil = Image.open(image_path).convert('RGB')
    image_np = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Run YOLO detection
    results = yolo_model.predict(source=image_path, save=False, imgsz=640, conf=0.3)

    draw = ImageDraw.Draw(image_pil)
    for result in results:
        if result.masks is None:
            continue
        for mask in result.masks.xy:
            mask = np.array(mask).astype(int)
            epsilon = 0.02 * cv2.arcLength(mask, True)
            approx = cv2.approxPolyDP(mask, epsilon, True)
            if len(approx) == 4:
                quad = [(int(p[0][0]), int(p[0][1])) for p in approx]
                ordered_box = order_box_points(np.array(quad, dtype='float32'))
                warped = warp_card(image_bgr, ordered_box)
                inp = preprocess_for_model(warped).to(device)

                with torch.no_grad():
                    preds = classifier(inp)
                label = decode_prediction(preds)
                words = label.split()
                label_with_newline = ' '.join(words[:2]) + '\n' + ' '.join(words[2:])

                draw.polygon(quad, outline='red')
                cx, cy = np.mean(quad, axis=0).astype(int)
                font = ImageFont.truetype("arial.ttf", size=24)
                draw.text((cx - 40, cy), label_with_newline, fill='yellow', font=font)

    # Show result
    plt.figure(figsize=(12, 8))
    plt.imshow(image_pil)
    plt.axis('off')
    plt.title("Detected SET Cards with Classification")
    plt.show()


if __name__ == '__main__':
    main()
