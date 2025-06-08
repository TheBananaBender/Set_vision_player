# train.py
from ultralytics import YOLO
import argparse

def train_yolo_seg(model_size, data_yaml, epochs=50, imgsz=640, batch=16):
    print(f"Starting training with model: {model_size}")
    model = YOLO(f"yolov8{model_size}-seg.pt")  # e.g. 'n' for nano

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project="card_polygon_runs",
        name=f"yolov8{model_size}_seg_card",
        exist_ok=True,
        save=True,
        save_period=5,  # Save every 5 epochs
        cache=True,
        device=0  # Change if using multiple GPUs
    )

    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="n", help="YOLOv8 size: n, s, m, l, x")
    parser.add_argument("--data", type=str, default="card_dataset/data.yaml", help="Path to dataset YAML")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    train_yolo_seg(args.model_size, args.data, args.epochs, args.imgsz, args.batch)
