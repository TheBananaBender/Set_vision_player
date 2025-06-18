import sys
import os

sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything"))

from grounded_sam import load_model, grounded_sam_predict

model = load_model()
image_path = "your_image.jpg"
text_prompt = "playing card"
results = grounded_sam_predict(model, image_path, text_prompt)