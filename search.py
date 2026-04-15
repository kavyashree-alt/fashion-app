import numpy as np
import faiss
import json
from feature_extractor import extract_features

features = np.load("features.npy")

with open("products.json") as f:
    products = json.load(f)

d = features.shape[1]
index = faiss.IndexFlatL2(d)
index.add(features)

def compute_score(q, db, product, category, gender, style, color):
    sim = np.dot(q, db)
    score = sim

    title = product["title"].lower()

    if category in title: score += 0.3
    else: score -= 0.5

    if gender in title: score += 0.2
    if style in title: score += 0.3
    if color in title: score += 0.4

    return score

def search_filtered(image_path, category, gender, style, color, top_k=6):
    query = extract_features(image_path)

    results = []

    for i, product in enumerate(products):
        db_feat = features[i]
        score = compute_score(query, db_feat, product, category, gender, style, color)
        results.append((score, product))

    results = sorted(results, key=lambda x: x[0], reverse=True)
    return [r[1] for r in results[:top_k]]