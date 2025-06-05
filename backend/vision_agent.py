import time
from utils import decode_base64_image

def process_frame(image_b64, settings: dict) -> str:
    image = decode_base64_image(image_b64)
    time.sleep(settings.get("delay", 1))  # simulate compute time
    # TODO: AI detection here
    return "SET found at A1, B2, C3"  # stub hint
