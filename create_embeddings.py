import os
import numpy as np
import pickle
from feature_extractor import extract_features

image_folder = "dataset/images"

features = []
image_paths = []

print("🔄 Creating embeddings...")

for file in os.listdir(image_folder):
    if file.endswith(".jpg"):
        path = os.path.join(image_folder, file)

        try:
            feat = extract_features(path)
            features.append(feat)
            image_paths.append(path)
        except:
            continue

features = np.array(features)

np.save("features.npy", features)

with open("image_paths.pkl", "wb") as f:
    pickle.dump(image_paths, f)

print("✅ Embeddings created!")