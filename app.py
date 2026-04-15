import streamlit as st
from PIL import Image
import tempfile
import cv2
import numpy as np

from transformers import pipeline
from segment_anything import sam_model_registry, SamPredictor

from feature_extractor import classify_full
from search import search_filtered

# ---------------- LOAD MODELS ----------------
detector = pipeline(
    task="zero-shot-object-detection",
    model="IDEA-Research/grounding-dino-base"
)

sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth")
predictor = SamPredictor(sam)

# ---------------- UI ----------------
st.title("🛍️ AI Fashion Search (FINAL 🔥)")

uploaded_file = st.file_uploader("Upload Image")

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        img_path = tmp.name

    image = Image.open(img_path)
    st.image(image, caption="Input Image")

    # ---------------- DETECTION ----------------
    candidate_labels = ["shirt", "pants", "shoes"]

   # ---------------- DETECTION ----------------
    results = detector(image, candidate_labels=candidate_labels)

# ---------------- FILTER ----------------
    filtered = {}

    for r in results:
        label = r["label"]
        score = r["score"]

        if label not in filtered or score > filtered[label]["score"]:
            filtered[label] = r

# ✅ NOW create filtered_results
    filtered_results = list(filtered.values())

# ✅ NOW you can limit
    filtered_results = filtered_results[:2]

    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    predictor.set_image(img_cv)

    items = []

    # ---------------- SEGMENTATION ----------------
    for r in filtered_results:
        box = r["box"]

        x1, y1, x2, y2 = int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])

        masks, _, _ = predictor.predict(
            box=np.array([x1, y1, x2, y2]),
            multimask_output=False
        )

        mask = masks[0]
        seg = img_cv * mask[:, :, None]

        path = f"{r['label']}.png"
        cv2.imwrite(path, seg)

        items.append(path)

        st.image(path, caption=r["label"], width=200)

    # ---------------- SEARCH ----------------
    if st.button("🔍 Find Similar Products"):

        for path in items:

            category, gender, style, color = classify_full(path)

            st.subheader(f"{color} {gender} {style} {category}")

            results = search_filtered(path, category, gender, style, color)

            cols = st.columns(3)

            for i, item in enumerate(results):
                with cols[i % 3]:
                    st.image(item["image"], width=150)
                    st.write(item["title"])
                    st.write(f"₹{item['price']}")
                    st.markdown(f"[View 🔗]({item['link']})")