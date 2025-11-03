import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import models
import torch.nn as nn
from pathlib import Path
from torchvision.transforms import v2
import timm
import os 

# Base dir of THIS file, regardless of where the process was launched
THIS_DIR = Path(__file__).resolve().parent

# Point to weights relative to this module
YOLO_MODEL_PATH = Path(os.getenv("YOLO_MODEL_PATH", THIS_DIR / "SET_yolo_model" / "best.pt"))
CLASSIFIER_MODEL_PATH = THIS_DIR / "classification model" / "best_mobilenetv4_set_card_finetuned_ver4.pth" 

from ultralytics import YOLO


class Pipeline():
    """
    The vision pipeline used to segment and detect the set cards
    the "detect_and_classify_from_array" is the "important function"
    """

    def __init__(self):
        self.yolo_model, self.classifier = self.load_models()
        self.device = next(self.classifier.parameters()).device


    def load_models(self):
        """loads the pth and pt files (paramaters) of the models"""
        _yolo_model = None
        _classifier = None
        if _yolo_model is None:
            _yolo_model = YOLO(YOLO_MODEL_PATH)
        if _classifier is None:
            _classifier = MultiHeadMobileNetV4()
            _classifier.load_state_dict(
                torch.load(CLASSIFIER_MODEL_PATH, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
            _classifier.eval()
            _classifier.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return _yolo_model, _classifier


    # --- Main Inference Function ---
    def detect_and_classify_from_array(self, image_bgr):
        
        """
        Args:
            image_bgr (np.ndarray): BGR image (OpenCV format).

        Returns:
            List of tuples: [(quad_points, color, number, shade, shape), ...]
        """
        results = self.yolo_model.predict(source=image_bgr, save=False, imgsz=640, conf=0.3,verbose=False)

        # Collect all valid quads and their warped cards first
        quads = []
        warped_cards = []
        
        for result in results:
            if result.masks is None:
                continue
            for mask in result.masks.xy:
                mask = np.array(mask).astype(int)
                x, y, w, h = cv2.boundingRect(mask)
                epsilon = 0.02 * cv2.arcLength(mask, True)
                approx = cv2.approxPolyDP(mask, epsilon, True)
                if len(approx) == 4:
                    quad = [(int(p[0][0]), int(p[0][1])) for p in approx]
                    ordered_box = order_box_points(np.array(quad, dtype='float32'))
                    
                    warped = warp_card(image_bgr, ordered_box)
                    quads.append(quad)
                    warped_cards.append(warped)

        # Batch classification if we have cards
        if not warped_cards:
            return []

        try:
            # Preprocess all cards into a batch tensor
            batch_tensor = preprocess_batch(warped_cards).to(self.device)
            
            # Single forward pass for all cards
            with torch.no_grad():
                preds_batch = self.classifier(batch_tensor)
        except Exception as e:
            print(f"[Pipeline] Error in batch classification: {e}")
            import traceback
            traceback.print_exc()
            return []  # Return empty on error
        
        # Decode predictions for all cards
        labels = []
        # preds_batch is a tuple of 4 tensors, each of shape [B, 3]
        # where B is batch size (number of cards)
        for i, quad in enumerate(quads):
            try:
                # Extract predictions for this card - each pred is [B, 3], extract row i
                color_pred = preds_batch[0][i:i+1]  # Keep as [1, 3] for decode_prediction
                shape_pred = preds_batch[1][i:i+1]  # Keep as [1, 3] for decode_prediction
                number_pred = preds_batch[2][i:i+1]  # Keep as [1, 3] for decode_prediction
                shade_pred = preds_batch[3][i:i+1]  # Keep as [1, 3] for decode_prediction
                
                card_preds = (color_pred, shape_pred, number_pred, shade_pred)
                color, number, shade, shape = decode_prediction(card_preds)
                labels.append((quad, color, number, shade, shape))
            except Exception as e:
                print(f"[Pipeline] Error decoding prediction for card {i}: {e}")
                # Skip this card if there's an error
                continue
        
        return labels

    


# --- Model Definition ---
class MultiHeadMobileNetV4(nn.Module):
    def __init__(self):
        super().__init__()
        base = timm.create_model('mobilenetv4_conv_medium', pretrained=True, num_classes=0)
        self.backbone = base
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)
        in_features = base.num_features

        self.head_color   = nn.Linear(in_features, 3)
        self.head_shape   = nn.Linear(in_features, 3)
        self.head_number  = nn.Linear(in_features, 3)
        self.head_shading = nn.Linear(in_features, 3)

    def forward(self, x):
        x = self.backbone.forward_features(x)  # shape [B, C, H, W]
        x = self.pool(x)                       # shape [B, C, 1, 1]
        x = self.flatten(x)                    # shape [B, C]
        x = self.dropout(x)
        return (
            self.head_color(x),
            self.head_shape(x),
            self.head_number(x),
            self.head_shading(x)
        )

# --- Utility Functions ---
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

def preprocess_batch(images):
    """
    Preprocess a batch of images for the classifier model.
    
    Args:
        images: List of numpy arrays (BGR images from OpenCV)
    
    Returns:
        torch.Tensor: Batched tensor of shape [B, 3, 256, 256]
    """
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((256, 256), antialias=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
    ])
    
    batch_tensors = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = transform(img_rgb)
        batch_tensors.append(tensor)
    
    # Stack into batch: [B, 3, 256, 256]
    return torch.stack(batch_tensors, dim=0)


def decode_prediction(preds):
    color_map = ['Red', 'Green', 'Purple']
    shape_map = ['Diamond', 'Squiggle', 'Oval']
    number_map = ['One', 'Two', 'Three']
    shading_map = ['Solid', 'Striped', 'Open']
    color, shape, number, shade = [torch.argmax(p, dim=1).item() for p in preds]
    return color, number, shade, shape


# --- Lazy loading of models ---

