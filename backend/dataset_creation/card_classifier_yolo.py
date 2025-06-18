import os
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from torchvision import models
import torch.nn as nn
from torchvision.transforms import v2


# --- Model Definition ---
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


# --- Utility Functions ---
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
    return f"{color_map[c]}_{shape_map[s]}_{number_map[n]}_{shading_map[sh]}"

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


# --- Batch Segment and Classify ---
def segment_classify_and_save(
    input_folder: str,
    output_folder: str,
    yolo_model_path: str,
    classifier_model_path: str
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models once
    yolo_model = YOLO(yolo_model_path)
    classifier = MultiHeadMobileNetV3().to(device)
    classifier.load_state_dict(torch.load(classifier_model_path, map_location=device))
    classifier.eval()

    # Get image paths
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_paths = [
        os.path.join(input_folder, f) for f in os.listdir(input_folder)
        if f.lower().endswith(image_extensions)
    ]
    os.makedirs(output_folder, exist_ok=True)

    for img_path in image_paths:
        try:
            image_pil = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"❌ Error loading {img_path}: {e}")
            continue

        image_np = np.array(image_pil)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        results = yolo_model.predict(source=image_bgr, save=False, imgsz=640, conf=0.3)

        for result_idx, result in enumerate(results):
            if result.masks is None:
                continue

            for mask_idx, mask in enumerate(result.masks.xy):
                polygon = np.array(mask).astype(np.int32)

                if len(polygon) < 3:
                    continue

                # Use approxPolyDP to get a clean quadrilateral (if possible)
                epsilon = 0.05 * cv2.arcLength(polygon, True)
                approx = cv2.approxPolyDP(polygon, epsilon, True)

                if len(approx) != 4:
                    continue  # skip non-quads

                quad = np.array([point[0] for point in approx], dtype="float32")
                ordered_box = order_box_points(quad)
                warped = warp_card(image_bgr, ordered_box)

                if warped.size == 0:
                    continue

                inp = preprocess_for_model(warped).to(device)
                with torch.no_grad():
                    preds = classifier(inp)
                label = decode_prediction(preds)

                label_folder = os.path.join(output_folder, label)
                os.makedirs(label_folder, exist_ok=True)

                base_name = Path(img_path).stem
                card_filename = f"{base_name}_card{result_idx}_{mask_idx}.jpg"
                cv2.imwrite(os.path.join(label_folder, card_filename), warped)

    print(f"\n✅ Done. Results saved in: {output_folder}")


# --- Run ---
if __name__ == "__main__":
    input_dir = "./input"
    output_dir = "classified_cards"
    yolo_weights = "C:\\Users\\galha\\Desktop\\Set_vision_player\\backend\\models\\SET_yolo_model\\best.pt"
    classifier_weights = "C:\\Users\\galha\\Desktop\\Set_vision_player\\backend\\models\\classification model\\mobilenetv3_set_card.pth"

    segment_classify_and_save(
        input_folder=input_dir,
        output_folder=output_dir,
        yolo_model_path=yolo_weights,
        classifier_model_path=classifier_weights
    )
