import json
import os
import requests

# ---------------- CONFIG ----------------
JSON_FOLDER = "json_data"   # folder containing all your JSON files
IMAGE_FOLDER = "dataset/images"

os.makedirs(IMAGE_FOLDER, exist_ok=True)

products = []
img_id = 0

# ---------------- CATEGORY DETECTION ----------------
def detect_category(filename):
    name = filename.lower()

    if "shirt" in name or "tshirt" in name:
        return "shirt"
    elif "pant" in name:
        return "pants"
    elif "shoe" in name:
        return "shoes"
    elif "kurti" in name:
        return "shirt"
    else:
        return "other"

print("🔄 Processing JSON files...")

# ---------------- LOOP THROUGH FILES ----------------
for file in os.listdir(JSON_FOLDER):

    if not file.endswith(".json"):
        continue

    filepath = os.path.join(JSON_FOLDER, file)
    category = detect_category(file)

    print(f"📂 Reading: {file}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ---------------- PROCESS EACH ITEM ----------------
    for item in data:

        try:
            # ✅ FIXED IMAGE KEY
            img_url = item.get("thumbnailImage")

            if not img_url:
                print("❌ No image found, skipping...")
                continue

            print("⬇️ Downloading:", img_url)

            img_path = f"{IMAGE_FOLDER}/{img_id}.jpg"

            # download image
            response = requests.get(img_url, timeout=5)
            with open(img_path, "wb") as f:
                f.write(response.content)

            # ✅ FIXED PRICE EXTRACTION
            price_data = item.get("price", {})
            price = price_data.get("value", "N/A")

            # create product entry
            product = {
                "id": img_id,
                "title": item.get("title", "Unknown"),
                "image": img_path,
                "price": price,
                "brand": item.get("brand", "Unknown"),
                "link": item.get("url", ""),
                "category": category
            }

            products.append(product)
            img_id += 1

        except Exception as e:
            print("⚠️ Error:", e)
            continue

# ---------------- SAVE FINAL DATASET ----------------
with open("products.json", "w", encoding="utf-8") as f:
    json.dump(products, f, indent=4)

print(f"\n✅ Dataset created successfully with {len(products)} products!")