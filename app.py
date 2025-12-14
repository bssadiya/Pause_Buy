from flask import Flask, render_template, request, redirect, url_for
import os, json
from werkzeug.utils import secure_filename

from similarity import check_similarity
from image_labeler import detect_category, detect_color, is_clothing_item

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
DATA_FILE = "data/closet.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("data", exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ---------- STORAGE ----------
def load_closet():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def save_closet(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ---------- FRONT PAGE ----------
@app.route("/")
def home():
    return render_template("index.html", closet=load_closet())


# ---------- CLOSET PAGE ----------
@app.route("/closet")
def closet_page():
    return render_template("closet.html", closet=load_closet())


# ---------- ADD TO CLOSET ----------
@app.route("/add", methods=["POST"])
def add_item():
    image = request.files.get("image")

    if not image:
        return redirect(url_for("home"))

    filename = secure_filename(image.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image.save(path)

    # Reject non-clothing uploads
    if not is_clothing_item(path):
        return render_template(
            "index.html",
            closet=load_closet(),
            message="❌ Please upload clothing items only."
        )

    category = detect_category(path)
    color = detect_color(path, category)
    text_label = f"{color} {category}"

    closet = load_closet()

    # 🔒 PREVENT DUPLICATE ITEMS
    for c in closet:
        if c["text"] == text_label:
            return render_template(
                "index.html",
                closet=closet,
                message="ℹ️ This item already exists in your closet."
            )

    item = {
        "image": "/" + path,
        "category": category,
        "color": color,
        "text": text_label
    }

    closet.append(item)
    save_closet(closet)

    return render_template(
        "index.html",
        closet=closet,
        message=f"✅ Added {color} {category} to your closet."
    )


# ---------- CHECK BEFORE BUYING ----------
@app.route("/check", methods=["POST"])
def check_item():
    text = request.form.get("item_text", "").strip()
    image = request.files.get("item_image")

    if not text and not image:
        return render_template(
            "index.html",
            closet=load_closet(),
            result="❌ Please enter text or upload an image."
        )

    query = text

    if image and image.filename:
        filename = secure_filename(image.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        image.save(path)

        category = detect_category(path)
        color = detect_color(path, category)
        query = f"{query} {color} {category}".strip()

    closet = load_closet()
    if not closet:
        return render_template(
            "index.html",
            closet=[],
            result="ℹ️ Closet empty. Add items first."
        )

    owned = [c["text"] for c in closet]
    found, raw_matches = check_similarity(query, owned)

    # 🔁 REMOVE DUPLICATE MATCHES
    matches = []
    seen = set()

    for t, s in raw_matches:
        if t in seen:
            continue
        seen.add(t)

        for c in closet:
            if c["text"] == t:
                matches.append({
                    "image": c["image"],
                    "label": t,
                    "score": s
                })
                break   # stop after first match

    result = "⚠️ You already own similar items." if found else "✅ Safe to buy."

    return render_template(
        "index.html",
        closet=closet,
        result=result,
        matches=matches,
        item=query
    )


if __name__ == "__main__":
    app.run(debug=True)
