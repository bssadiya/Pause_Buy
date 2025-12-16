from PIL import Image
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans

# =================================================
# LOAD CLIP MODEL (ONCE)
# =================================================
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# =================================================
# ALLOWED CLOTHING CATEGORIES
# =================================================
CATEGORIES = [
    "crop top",
    "t-shirt",
    "shirt",
    "top",
    "hoodie",
    "sweater",
    "jacket",
    "kurti",
    "shorts",
    "skirt",
    "pants",
    "jeans",
    "trousers",
    "palazzo",
    "dress",
    "saree",
    "lehenga",
    "jumpsuit"
]

# =================================================
# SAFE IMAGE OPEN (FIXES PIL ERRORS)
# =================================================
def load_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        return img
    except Exception as e:
        print("❌ Image load failed:", e)
        return None

# =================================================
# LENGTH DETECTION (PRODUCT IMAGES)
# =================================================
def detect_length(image_path):
    image = load_image(image_path)
    if image is None:
        return "unknown"

    w, h = image.size
    arr = np.array(image)
    gray = np.mean(arr, axis=2)

    fabric_rows = np.where(gray < 235)[0]
    if len(fabric_rows) == 0:
        return "unknown"

    top = fabric_rows[0]
    bottom = fabric_rows[-1]
    ratio = (bottom - top) / h

    if ratio < 0.45:
        return "mini"
    elif ratio < 0.85:
        return "knee-length"
    else:
        return "long"

# =================================================
# SHAPE RULES
# =================================================
def is_crop_top(image_path):
    image = load_image(image_path)
    if image is None:
        return False
    w, h = image.size
    return (h / w) < 0.9

def is_flared(image_path):
    image = load_image(image_path)
    if image is None:
        return False
    w, h = image.size
    upper = image.crop((0, int(h * 0.25), w, int(h * 0.45)))
    lower = image.crop((0, int(h * 0.55), w, h))
    return np.array(lower).std() > np.array(upper).std() * 1.25

def is_saree(image_path, predicted):
    image = load_image(image_path)
    if image is None:
        return False
    w, h = image.size
    if predicted not in ["saree", "dress"]:
        return False
    if h / w < 2.4:
        return False
    lower = image.crop((0, int(h * 0.55), w, h))
    return np.array(lower).std() > 55

# =================================================
# CATEGORY DETECTION
# =================================================
def detect_category(image_path):
    image = load_image(image_path)
    if image is None:
        return "unknown"

    inputs = processor(
        text=CATEGORIES,
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)

    predicted = CATEGORIES[probs.argmax().item()]
    length = detect_length(image_path)

    if is_saree(image_path, predicted):
        return "saree"

    if predicted == "skirt" and length == "long" and is_flared(image_path):
        return "lehenga"

    if predicted in ["top", "shirt", "t-shirt"] and is_crop_top(image_path):
        return "crop top"

    if predicted == "skirt":
        return f"{length} skirt"

    if predicted == "dress":
        return f"{length} dress"

    if predicted == "kurti":
        return f"{length} kurti"

    return predicted

# =================================================
# COLOR DETECTION
# =================================================
def detect_color(image_path, category=None):
    image = load_image(image_path)
    if image is None:
        return "unknown"

    w, h = image.size

    if category in ["saree", "lehenga"]:
        crop = image.crop((int(w * 0.2), int(h * 0.35),
                           int(w * 0.8), int(h * 0.65)))
    else:
        crop = image.crop((int(w * 0.25), int(h * 0.25),
                           int(w * 0.75), int(h * 0.75)))

    crop = crop.resize((120, 120))
    pixels = np.array(crop).reshape(-1, 3)

    kmeans = KMeans(n_clusters=3, n_init=10)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_

    dominant = centers[np.bincount(labels).argmax()]
    return rgb_to_name(*dominant)

# =================================================
# RGB → COLOR NAME (YOUR LOGIC, FIXED)
# =================================================
def rgb_to_name(r, g, b):
    r, g, b = int(r), int(g), int(b)
    brightness = (r + g + b) / 3

    if brightness < 45:
        return "black"
    if brightness > 230:
        return "white"

    if abs(r - g) < 15 and abs(g - b) < 15:
        return "grey" if brightness < 180 else "silver"

    if r > 170 and g > 150 and b < 120:
        return "yellow"
    if r > 180 and g > 110 and b < 90:
        return "orange"
    if r > 120 and g < 100 and b < 80:
        return "brown"
    if r > g and r > b:
        return "maroon" if r < 150 else "red"
    if g > r and g > b:
        return "olive" if g < 140 else "green"
    if b > r and b > g:
        return "navy" if b < 100 else "blue"
    if r > 140 and b > 140:
        return "lavender" if g > 140 else "purple"
    if r > 180 and g > 130 and b > 150:
        return "pink"
    if g > 120 and b > 120 and r < 100:
        return "teal"

    return "multicolour"

# =================================================
# CLOTHING / NON-CLOTHING CHECK (FIXED)
# =================================================
def is_clothing_item(image_path):
    image = load_image(image_path)
    if image is None:
        return False

    check_labels = [
        "a garment",
        "fashion apparel",
        "a person wearing clothes",
        "electronics",
        "furniture",
        "food item",
        "footwear",
        "bag"
    ]

    inputs = processor(
        text=check_labels,
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)

    top_label = check_labels[probs.argmax().item()]
    confidence = probs.max().item()

    print("🧠 CLIP:", top_label, round(confidence, 3))

    if top_label in [
        "a garment",
        "fashion apparel",
        "a person wearing clothes"
    ] and confidence > 0.30:
        return True

    return False

