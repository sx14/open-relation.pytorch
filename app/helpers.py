import base64
from io import BytesIO

from PIL import Image


def base64_encode_image(image):
    img_buffer = BytesIO()
    image.save(img_buffer, format='JPEG')
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    return base64_str


def base64_decode_image(base64_str):
    image = base64.b64decode(base64_str)
    image = BytesIO(image)
    image = Image.open(image)
    return image
