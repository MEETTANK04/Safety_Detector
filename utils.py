from PIL import Image
import numpy as np
import io
from tensorflow.keras.preprocessing import image

def preprocess_image(img_file):
    # Case 1: Streamlit uploaded file
    if isinstance(img_file, io.BytesIO) or hasattr(img_file, "read"):
        img = Image.open(img_file)  # No need to wrap in BytesIO again
    # Case 2: File path
    elif isinstance(img_file, str):
        img = image.load_img(img_file, target_size=(224, 224))
        img = img.convert('RGB')  # Ensure 3 channels
    else:
        raise ValueError("Unsupported image input format")

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array
