# train.py
import os
from ultralytics import YOLO
from tqdm import tqdm
import torch



def main():
    # ⚙️ Configs
    model_type = 'yolov11n-seg.pt'
    data_yaml = 'card_dataset/data.yaml'
    epochs = 5
    imgsz = 640
    batch = 16
    save_interval = 5  # Save every 5 epochs
    device = 0 if torch.cuda.is_available() else 'cpu'

    print(f" Using device: {device}")
    print(f" Loading model: {model_type}")
    model = YOLO(model_type)

    # 🧠 Training loop using .train() does not expose internals, so we use callbacks
    print(f" Training for {epochs} epochs...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,  
        imgsz=imgsz,
        batch=batch,
        device=device,
        save=True,
        save_period=save_interval,
        project='runs',
        name='yolo_card_seg',
        exist_ok=True,
        verbose=True
    )

    # 🔍 Print summary stats
    print("\n Training finished.")
    if results:
        metrics = results.metrics
        print(f" Final Results:\n{metrics}")

if __name__ == '__main__':
    main()
