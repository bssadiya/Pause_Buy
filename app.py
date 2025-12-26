from flask import Flask, render_template, request, redirect, jsonify
from flask_cors import CORS
import os, json, requests
from uuid import uuid4
from PIL import Image
import imagehash

from image_labeler import (
    detect_category,
    detect_color,
    is_clothing_item
)

# =================================================
# APP SETUP
# =================================================
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "static/uploads"
DATA_FILE = "data/closet.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("data", exist_ok=True)

# =================================================
# CATEGORY GROUPS
# =================================================
CATEGORY_GROUPS = {
    "top": ["top", "t-shirt", "shirt", "crop", "hoodie", "sweater", "jacket", "kurti"],
    "bottom": ["pants", "jeans", "trousers", "palazzo", "shorts", "skirt"],
    "dress": ["dress", "saree", "lehenga", "jumpsuit"]
}

def resolve_group(category):
    if not category:
        return "unknown"
    category = category.lower()
    for group, keys in CATEGORY_GROUPS.items():
        if any(k in category for k in keys):
            return group
    return "unknown"

def same_category(cat1, cat2):
    return resolve_group(cat1) == resolve_group(cat2)

# =================================================
# STORAGE
# =================================================
def load_closet():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def save_closet(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

# =================================================
# ROUTES
# =================================================
@app.route("/")
def home():
    return render_template("index.html", closet=load_closet())

@app.route("/closet")
def closet():
    return render_template("closet.html", closet=load_closet())

# =================================================
# ADD ITEM MANUALLY
# =================================================
@app.route("/add", methods=["POST"])
def add_manual():
    img = request.files.get("image")
    if not img:
        return redirect("/")

    path = os.path.join(UPLOAD_FOLDER, f"manual_{uuid4().hex}.jpg")
    img.save(path)

    #  Do not reject too aggressively
    if not is_clothing_item(path):
        print(" Forcing clothing for manual add")

    raw_category = detect_category(path)
    color = detect_color(path, raw_category)
    group = resolve_group(raw_category)

    closet = load_closet()

    closet.append({
        "image": "/" + path,
        "category": raw_category,
        "group": group,
        "color": color,
        "text": f"{color} {raw_category}"
    })

    save_closet(closet)
    return redirect("/closet")

# =================================================
# MANUAL CHECK — "CAN I BUY THIS?"
# =================================================
@app.route("/check", methods=["POST"])
def check_manual():
    closet = load_closet()
    image = request.files.get("item_image")

    if not image:
        return render_template(
            "index.html",
            closet=closet,
            result=" Please upload an image.",
            matches=[]
        )

    path = os.path.join(UPLOAD_FOLDER, f"check_{uuid4().hex}.jpg")
    image.save(path)

    if not is_clothing_item(path):
        print(" Forcing clothing for check")

    product_category = detect_category(path)

    category_matches = [
        c for c in closet
        if same_category(c.get("category"), product_category)
    ]

    os.remove(path)

    if not category_matches:
        return render_template(
            "index.html",
            closet=closet,
            result=" Safe to buy. You don’t own this category.",
            matches=[]
        )

    return render_template(
        "index.html",
        closet=closet,
        result=" You already own items in this category.",
        matches=category_matches
    )

# =================================================
# FLIPKART CHECK (BUY / ADD TO CART)
# =================================================
@app.route("/check-product", methods=["POST"])
def check_product():
    try:
        url = request.json.get("image")
        if not url:
            return jsonify({"ignore": False})

        temp_path = f"{UPLOAD_FOLDER}/temp_{uuid4().hex}.jpg"
        r = requests.get(url, timeout=10)
        with open(temp_path, "wb") as f:
            f.write(r.content)

        if not is_clothing_item(temp_path):
            print(" Forcing clothing for Flipkart")

        product_category = detect_category(temp_path)
        product_group = resolve_group(product_category)

        closet = load_closet()

        matches = [
            c for c in closet
            if resolve_group(c.get("category")) == product_group
        ]

        os.remove(temp_path)

        if not matches:
            return jsonify({
                "is_clothing": True,
                "found": False,
                "matches": []
            })

        return jsonify({
            "is_clothing": True,
            "found": True,
            "matches": matches
        })

    except Exception as e:
        print("check-product error:", e)
        return jsonify({"ignore": False})

# =================================================
# CLEAR CLOSET
# =================================================
@app.route("/clear-closet", methods=["POST"])
def clear_closet():
    save_closet([])
    for f in os.listdir(UPLOAD_FOLDER):
        if f.startswith(("order_", "temp_", "manual_", "check_")):
            try:
                os.remove(os.path.join(UPLOAD_FOLDER, f))
            except:
                pass
    return jsonify({"status": "cleared"})

# =================================================
# RUN
# =================================================
if __name__ == "__main__":
    app.run(debug=True)
