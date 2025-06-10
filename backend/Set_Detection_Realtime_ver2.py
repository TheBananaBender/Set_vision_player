import cv2
import numpy as np
import time
import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import v2
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
FPS = 10


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = MultiHeadMobileNetV3().to(device)
    classifier.load_state_dict(torch.load("mobilenetv3_set_card.pth", map_location=device))
    classifier.eval()

    detector = YOLO("./best.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    last_prediction_time = 0
    predictions = []  # list of tuples: (box, label)

    font = ImageFont.truetype("arial.ttf", 24)  # Use a system font or provide path

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        updated_predictions = []

        if current_time - last_prediction_time >= 1/FPS:
            # Run YOLO detection
            results = detector.predict(source=frame, imgsz=640, conf=0.3, device=0 if torch.cuda.is_available() else 'cpu')
            for result in results:
                if result.masks is None:
                    continue
                for mask in result.masks.xy:
                    mask = np.array(mask).astype(int)
                    epsilon = 0.02 * cv2.arcLength(mask, True)
                    approx = cv2.approxPolyDP(mask, epsilon, True)
                    if len(approx) == 4:
                        quad = [(int(p[0][0]), int(p[0][1])) for p in approx]
                        box = order_box_points(np.array(quad, dtype='float32'))
                        cropped = warp_card(frame, box)
                        inp = preprocess_for_model(cropped).to(device)

                        with torch.no_grad():
                            preds = classifier(inp)
                        label = decode_prediction(preds)

                        # Line break after two words
                        words = label.split()
                        label = ' '.join(words[:2]) + '\n' + ' '.join(words[2:])
                        updated_predictions.append((box.astype(int), label))

            predictions = updated_predictions
            last_prediction_time = current_time

        # Draw overlays using PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_img)

        for box, label in predictions:
            draw.polygon([tuple(p) for p in box], outline="green", width=3)
            cx, cy = np.mean(box, axis=0).astype(int)
            draw.text((cx - 60, cy), label, fill="yellow", font=font)

        # Convert back to OpenCV for display
        frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.imshow("SET Card Realtime Detection", frame_bgr)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()