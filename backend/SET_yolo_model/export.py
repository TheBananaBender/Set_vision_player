# export.py
from ultralytics import YOLO
import argparse

def export_model(weights_path, export_format="onnx", imgsz=640, dynamic=False):
    print(f"Loading model: {weights_path}")
    model = YOLO(weights_path)

    print(f"Exporting to {export_format.upper()} format...")
    model.export(
        format=export_format,
        imgsz=imgsz,
        dynamic=dynamic,
        optimize=True  # try to fuse layers if possible
    )
    print(f"Export complete → check output folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="card_polygon_runs/yolov8n_seg_card/weights/best.pt", help="Trained .pt model path")
    parser.add_argument("--format", type=str, default="onnx", help="Export format: onnx, engine, torchscript, etc.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--dynamic", action="store_true", help="Use dynamic input shapes (for ONNX/engine)")
    args = parser.parse_args()

    export_model(args.weights, args.format, args.imgsz, args.dynamic)
