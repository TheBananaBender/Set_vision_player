from PIL import Image
from io import BytesIO
import base64

def decode_base64_image(base64_str):
    header, encoded = base64_str.split(',', 1)
    img_bytes = base64.b64decode(encoded)
    image = Image.open(BytesIO(img_bytes))
    return image
