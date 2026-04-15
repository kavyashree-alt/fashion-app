import torch
import open_clip
from PIL import Image
import numpy as np

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
model.eval()

labels = [
    "men formal shirt",
    "men casual shirt",
    "men t-shirt",
    "women top",
    "jeans",
    "pants",
    "shoes"
]

with torch.no_grad():
    text_tokens = open_clip.tokenize(labels)
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

def extract_features(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    with torch.no_grad():
        features = model.encode_image(image)
        features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().flatten()

def detect_color(image_path):
    img = Image.open(image_path).resize((50, 50))
    arr = np.array(img)
    r, g, b = arr.mean(axis=(0, 1))

    if b > r and b > g: return "blue"
    if r > g and r > b: return "red"
    if g > r and g > b: return "green"
    return "dark"

def classify_full(image_path):
    feat = extract_features(image_path)

    sims = [np.dot(feat, tf.cpu().numpy()) for tf in text_features]
    best = labels[np.argmax(sims)]

    category = "shirt" if "shirt" in best or "top" in best else ("pants" if "pants" in best or "jeans" in best else "shoes")
    gender = "men" if "men" in best else "women"
    style = "formal" if "formal" in best else "casual"
    color = detect_color(image_path)

    return category, gender, style, color