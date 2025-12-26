from PIL import Image
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans
import cv2

# =================================================
# LOAD CLIP MODEL (ONCE)
# =================================================
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# =================================================
# FINE-GRAINED CLOTHING CATEGORIES
# =================================================
CATEGORIES = [
    "crop top", "t-shirt", "shirt", "top",
    "hoodie", "sweater", "jacket", "kurti",
    "shorts", "skirt", "pants", "jeans",
    "trousers", "palazzo", "dress",
    "saree", "lehenga", "jumpsuit"
]

# =================================================
# CATEGORY NORMALIZATION
# =================================================
def normalize_category(cat: str):
    if not cat:
        return "unknown"
    cat = cat.lower()

    if any(k in cat for k in ["top", "t-shirt", "shirt", "crop", "hoodie", "sweater", "jacket", "kurti"]):
        return "top"
    if any(k in cat for k in ["pants", "jeans", "trousers", "palazzo", "shorts", "skirt"]):
        return "bottom"
    if any(k in cat for k in ["dress", "lehenga", "saree", "jumpsuit"]):
        return "dress"

    return "unknown"

# =================================================
# SAFE IMAGE LOAD
# =================================================
def load_image(path):
    try:
        return Image.open(path).convert("RGB")
    except:
        return None

# =================================================
# CATEGORY DETECTION (CLIP)
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
        probs = model(**inputs).logits_per_image.softmax(dim=1)

    return CATEGORIES[probs.argmax().item()]

# =================================================
# COLOR HELPERS
# =================================================
def brightness(r, g, b):
    return (r + g + b) / 3

COLOR_PALETTE = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "grey": (130, 130, 130),
    "brown": (150, 100, 60),
    "red": (180, 30, 30),
    "pink": (230, 150, 170),
    "orange": (220, 140, 40),
    "yellow": (240, 220, 90),
    "green": (70, 140, 70),
    "olive": (120, 130, 50),
    "teal": (60, 130, 130),
    "blue": (70, 100, 180),
    "navy": (40, 60, 120),
    "purple": (130, 80, 160),
}

COLOR_FAMILY_MAP = {
    "maroon": "red", "wine": "red", "burgundy": "red",
    "rose": "pink", "peach": "pink",
    "skyblue": "blue", "denim": "blue",
    "coffee": "brown", "tan": "brown",
    "lavender": "purple",
    "mint": "green",
    "silver": "grey", "gray": "grey",
}

def normalize_color(color):
    return COLOR_FAMILY_MAP.get(color, color)

# =================================================
# RGB → LAB FALLBACK
# =================================================
def rgb_to_name(r, g, b):
    pixel = np.uint8([[[int(r), int(g), int(b)]]])
    lab = cv2.cvtColor(pixel, cv2.COLOR_RGB2LAB)[0][0]

    best = "unknown"
    min_dist = float("inf")

    for name, rgb in COLOR_PALETTE.items():
        ref = np.uint8([[rgb]])
        ref_lab = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB)[0][0]
        dist = np.linalg.norm(lab - ref_lab)
        if dist < min_dist:
            min_dist = dist
            best = name

    return best

# =================================================
# COLOR DETECTION (FIXED + STABLE)
# =================================================
def detect_color(image_path, category=None):
    image = load_image(image_path)
    if image is None:
        return "unknown"

    w, h = image.size
    crop = image.crop((int(w*0.25), int(h*0.25), int(w*0.75), int(h*0.75)))
    crop = crop.resize((120, 120))

    crop_np = np.array(crop)

    # ---------- STEP 1: BLACK / WHITE ----------
    gray = cv2.cvtColor(crop_np, cv2.COLOR_RGB2GRAY)
    dark_ratio = np.mean(gray < 50)
    bright_ratio = np.mean(gray > 220)

    if dark_ratio > 0.55:
        return "black"
    if bright_ratio > 0.55:
        return "white"

    # ---------- STEP 2: SATURATION FILTER ----------
    hsv = cv2.cvtColor(crop_np, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]

    low_sat_ratio = np.mean(sat < 35)
    if low_sat_ratio > 0.6:
        return "grey"

    # ---------- STEP 3: DOMINANT COLOR ----------
    pixels = crop_np.reshape(-1, 3)
    kmeans = KMeans(n_clusters=3, n_init=10)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_
    counts = np.bincount(labels)

    dominant = centers[counts.argmax()]
    dominance_ratio = counts.max() / counts.sum()

    if dominance_ratio < 0.45:
        return "multicolor"

    r, g, b = dominant
    hsv_pixel = cv2.cvtColor(
        np.uint8([[[int(r), int(g), int(b)]]]),
        cv2.COLOR_RGB2HSV
    )[0][0]

    hue, sat_val, val = hsv_pixel
    if dominance_ratio < 0.45:
       return "multicolor"


    # ---------- STEP 4: HUE RULES ----------
    # ---------- STEP 4: HUE RULES (FIXED) ----------

#  RED / MAROON
    if hue <= 10 or hue >= 170:
       return "maroon" if val < 120 else "red"

#  GREEN / OLIVE (PRIORITY OVER YELLOW)
    if 36 <= hue <= 85:
       return "olive" if val < 120 else "green"

#  TRUE YELLOW (STRICT)
    if 22 <= hue <= 35 and sat_val > 120:
       return "yellow"

#  BLUE / NAVY
    if 101 <= hue <= 130:
        return "navy" if val < 110 else "blue"

# 🟦 TEAL (ONLY TRUE TEAL)
    if 86 <= hue <= 100:
      return "teal"

#  BROWN
    if r > 120 and g > 90 and b < 80:
       return "brown"

#  PINK
    if r > 180 and g > 130 and b > 150:
        return "pink"

   

    # ---------- STEP 5: FINAL FALLBACK ----------
    return rgb_to_name(r, g, b)


# =================================================
# STRICT CLOTHING FILTER (CLIP)
# =================================================
def is_clothing_item(image_path):
    image = load_image(image_path)
    if image is None:
        return False

    labels = [
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
        text=labels,
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        probs = model(**inputs).logits_per_image.softmax(dim=1)

    idx = probs.argmax().item()
    label = labels[idx]
    confidence = probs.max().item()

    print(" CLIP:", label, round(confidence, 3))

    return (
        label in ["a garment", "fashion apparel", "a person wearing clothes"]
        and confidence > 0.30
    )
