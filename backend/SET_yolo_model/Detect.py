# detect.py
import cv2
import torch
from ultralytics import YOLO
import argparse
import time

def draw_polygon(frame, polygon, color=(0, 255, 0), thickness=2):
    polygon = polygon.reshape(-1, 2).astype(int)
    cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=thickness)

def run_detection(model_path, source=0, conf=0.25, show=True):
    print(f"Loading model from {model_path}")
    model = YOLO(model_path)  # Make sure this is a YOLOv8n-seg model (.pt)

    cap = cv2.VideoCapture(source)
    assert cap.isOpened(), f"Cannot open source: {source}"

    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Failed to grab frame")
            break

        t1 = time.time()
        results = model(frame, conf=conf)[0]
        t2 = time.time()

        for mask in results.masks.xy if results.masks else []:
            draw_polygon(frame, mask)

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf_score = box.conf[0]
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            label = f"{model.names[cls_id]} {conf_score:.2f}"
            cv2.rectangle(frame, tuple(xyxy[:2]), tuple(xyxy[2:]), (255, 0, 0), 2)
            cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        fps = 1 / (t2 - t1)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if show:
            cv2.imshow("YOLOv8n Polygon Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="card_model.pt", help="Path to trained YOLOv8n-seg model")
    parser.add_argument("--source", type=str, default="0", help="Video source: 0 for webcam or path to video/image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--noshow", action="store_true", help="Run without display window")
    args = parser.parse_args()

    video_source = int(args.source) if args.source.isdigit() else args.source
    run_detection(args.model, source=video_source, conf=args.conf, show=not args.noshow)
