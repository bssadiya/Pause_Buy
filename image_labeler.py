# # from PIL import Image
# # import torch
# # from transformers import CLIPProcessor, CLIPModel
# # import numpy as np
# # from sklearn.cluster import KMeans

# # # ===============================
# # # LOAD CLIP MODEL
# # # ===============================
# # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# # # ===============================
# # # STRICT CLOTHING CATEGORIES
# # # ===============================
# # CATEGORIES = [
# #     # Top wear
# #     "crop top",
# #     "t-shirt",
# #     "shirt",
# #     "top",
# #     "hoodie",
# #     "sweater",
# #     "jacket",
# #     "kurti",

# #     # Bottom wear
# #     "shorts",
# #     "skirt",
# #     "pants",
# #     "jeans",
# #     "trousers",
# #     "palazzo",

# #     # One-piece / ethnic
# #     "dress",
# #     "saree",
# #     "lehenga",
# #     "jumpsuit"
# # ]

# # # ===============================
# # # CATEGORY DETECTION (DIRECT)
# # # ===============================
# # # def detect_category(image_path):
# # #     image = Image.open(image_path).convert("RGB")

# # #     inputs = processor(
# # #         text=CATEGORIES,
# # #         images=image,
# # #         return_tensors="pt",
# # #         padding=True
# # #     )

# # #     with torch.no_grad():
# # #         outputs = model(**inputs)
# # #         probs = outputs.logits_per_image.softmax(dim=1)

# # #     category = CATEGORIES[probs.argmax().item()]

# # #     # Add length if applicable
# # #     if category in ["skirt", "dress", "kurti"]:
# # #         length = detect_length(image_path)
# # #         return f"{length} {category}"

# # #     return category
# # def is_flared(image_path):
# #     image = Image.open(image_path)
# #     w, h = image.size

# #     lower = image.crop((0, int(h * 0.6), w, h))
# #     upper = image.crop((0, int(h * 0.2), w, int(h * 0.4)))

# #     return lower.size[0] > upper.size[0] * 1.1

# # def detect_category(image_path):
# #     image = Image.open(image_path).convert("RGB")

# #     # ---------- STEP 1: Broad type ----------
# #     broad_labels = ["top wear", "bottom wear", "one piece"]

# #     inputs = processor(
# #         text=broad_labels,
# #         images=image,
# #         return_tensors="pt",
# #         padding=True
# #     )

# #     with torch.no_grad():
# #         outputs = model(**inputs)
# #         probs = outputs.logits_per_image.softmax(dim=1)

# #     broad_type = broad_labels[probs.argmax().item()]

# #     # ---------- STEP 2: Narrow category ----------
# #     if broad_type == "bottom wear":
# #         labels = ["skirt", "lehenga", "shorts", "pants", "jeans", "trousers"]

# #     elif broad_type == "one piece":
# #         labels = ["dress", "saree", "jumpsuit"]

# #     else:
# #         labels = ["crop top", "shirt", "t-shirt", "kurti", "hoodie"]

# #     inputs = processor(
# #         text=labels,
# #         images=image,
# #         return_tensors="pt",
# #         padding=True
# #     )

# #     with torch.no_grad():
# #         outputs = model(**inputs)
# #         probs = outputs.logits_per_image.softmax(dim=1)

# #     category = labels[probs.argmax().item()]

# #     # ---------- STEP 3: lehenga rule ----------
# #     if category == "skirt":
# #         length = detect_length(image_path)
# #         if length == "long":
# #             return "lehenga"

# #     if category in ["skirt", "dress", "kurti"]:
# #         length = detect_length(image_path)
# #         return f"{length} {category}"
# #     # Strong lehenga override
# #     if category in ["skirt", "kurti"]:
# #         length = detect_length(image_path)

# #     if length == "long" and is_flared(image_path):
# #         return "lehenga"


# #     return category
# #     category = labels[probs.argmax().item()]

# # # ---- ALWAYS compute length ONCE ----
# # length = detect_length(image_path)

# # # ---- Lehenga override ----
# # if category == "skirt" and length == "long" and is_flared(image_path):
# #     return "lehenga"

# # # ---- Add length info ----
# # if category in ["skirt", "dress", "kurti"]:
# #     return f"{length} {category}"

# # return category



# # # ===============================
# # # LENGTH DETECTION (PRODUCT IMAGE)
# # # ===============================
# # def detect_length(image_path):
# #     image = Image.open(image_path)
# #     w, h = image.size
# #     ratio = h / w

# #     if ratio < 1.2:
# #         return "mini"
# #     elif ratio < 1.7:
# #         return "knee-length"
# #     else:
# #         return "long"

# # # ===============================
# # # COLOR DETECTION
# # # ===============================
# # def detect_color(image_path):
# #     image = Image.open(image_path).convert("RGB")

# #     w, h = image.size
# #     crop = image.crop((
# #         int(w * 0.25),
# #         int(h * 0.25),
# #         int(w * 0.75),
# #         int(h * 0.75)
# #     ))

# #     crop = crop.resize((120, 120))
# #     pixels = np.array(crop).reshape(-1, 3)

# #     kmeans = KMeans(n_clusters=3, n_init=10)
# #     labels = kmeans.fit_predict(pixels)
# #     centers = kmeans.cluster_centers_

# #     dominant = centers[np.bincount(labels).argmax()]
# #     return rgb_to_name(*dominant)

# # # ===============================
# # # RGB → COLOR NAME
# # # ===============================
# # def rgb_to_name(r, g, b):
# #     r, g, b = int(r), int(g), int(b)
# #     brightness = (r + g + b) / 3

# #     # ---------- TRUE BLACK ----------
# #     if brightness < 45 and abs(r - g) < 10 and abs(g - b) < 10:
# #         return "black"

# #     # ---------- WHITE ----------
# #     if brightness > 230:
# #         return "white"

# #     # ---------- YELLOW ----------
# #     if r > 180 and g > 180 and b < 120:
# #         return "yellow"

# #     # ---------- ORANGE ----------
# #     if r > 180 and g > 120 and b < 100:
# #         return "orange"

# #     # ---------- RED / MAROON ----------
# #     if r > g and r > b:
# #         if r < 130:
# #             return "maroon"
# #         return "red"

# #     # ---------- GREEN / OLIVE ----------
# #     if g > r and g > b:
# #         if g < 130:
# #             return "olive"
# #         return "green"

# #     # ---------- BLUE / NAVY ----------
# #     if b > r and b > g:
# #         if b < 130:
# #             return "navy"
# #         return "blue"

# #     # ---------- PURPLE / LAVENDER ----------
# #     if r > 140 and b > 140:
# #         if g > 140:
# #             return "lavender"
# #         return "purple"

# #     # ---------- PINK ----------
# #     if r > 180 and g > 130 and b > 150:
# #         return "pink"

# #     # ---------- GREY / SILVER ----------
# #     if abs(r - g) < 15 and abs(g - b) < 15:
# #         if brightness < 180:
# #             return "grey"
# #         return "silver"

# #     # ---------- BEIGE / BROWN ----------
# #     if r > 170 and g > 150 and b > 120:
# #         return "beige"

# #     if r > 120 and g < 100 and b < 80:
# #         return "brown"

# #     # ---------- TEAL ----------
# #     if g > 120 and b > 120 and r < 100:
# #         return "teal"

# #     return "multicolour"

# from PIL import Image
# import torch
# from transformers import CLIPProcessor, CLIPModel
# import numpy as np
# from sklearn.cluster import KMeans

# # =================================================
# # LOAD CLIP MODEL (ONCE)
# # =================================================
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# # =================================================
# # ALL CATEGORIES (NO GROUPING)
# # =================================================
# CATEGORIES = [
#     # Top wear
#     "crop top",
#     "t-shirt",
#     "shirt",
#     "top",
#     "hoodie",
#     "sweater",
#     "jacket",
#     "kurti",

#     # Bottom wear
#     "shorts",
#     "skirt",
#     "pants",
#     "jeans",
#     "trousers",
#     "palazzo",

#     # One-piece / ethnic
#     "dress",
#     "saree",
#     "lehenga",
#     "jumpsuit"
# ]

# def detect_length(image_path):
#     image = Image.open(image_path)
#     w, h = image.size
#     ratio = h / w

#     if ratio < 1.2:
#         return "mini"
#     elif ratio < 1.7:
#         return "knee-length"
#     else:
#         return "long"

# def is_flared(image_path):
#     image = Image.open(image_path)
#     w, h = image.size

#     upper = image.crop((0, int(h * 0.25), w, int(h * 0.45)))
#     lower = image.crop((0, int(h * 0.55), w, h))

#     return np.array(lower).std() > np.array(upper).std() * 1.25

# def is_crop_top(image_path):
#     image = Image.open(image_path)
#     w, h = image.size
#     return h / w < 0.9

# def is_skirt_like(image_path):
#     image = Image.open(image_path)
#     w, h = image.size
#     ratio = h / w

#     # Skirts are not extremely tall
#     if ratio > 2.0:
#         return False

#     # Bottom-heavy content
#     upper = image.crop((0, 0, w, int(h * 0.4)))
#     lower = image.crop((0, int(h * 0.4), w, h))

#     return np.array(lower).std() > np.array(upper).std() * 1.2


# def is_saree(image_path, predicted):
#     image = Image.open(image_path)
#     w, h = image.size
#     ratio = h / w

#     if predicted not in ["saree", "dress"]:
#         return False

#     if ratio < 2.5:
#         return False

#     lower = image.crop((0, int(h * 0.6), w, h))
#     return np.array(lower).std() > 60

# # =================================================
# # MAIN CATEGORY DETECTOR (STRICT RULES)
# # =================================================
# def detect_category(image_path):
#     image = Image.open(image_path).convert("RGB")

#     inputs = processor(
#         text=CATEGORIES,
#         images=image,
#         return_tensors="pt",
#         padding=True
#     )

#     with torch.no_grad():
#         outputs = model(**inputs)
#         probs = outputs.logits_per_image.softmax(dim=1)

#     predicted = CATEGORIES[probs.argmax().item()]
#     length = detect_length(image_path)





# # 1️⃣ Saree
#     if is_saree(image_path, predicted):
#       return "saree"

# # 2️⃣ Lehenga
#     if predicted == "skirt" and length == "long" and is_flared(image_path):
#       return "lehenga"

# # 3️⃣ SKIRT (block top wear)
#     if predicted is_skirt_like(image_path):
#       return f"{length} skirt"

# # 4️⃣ Crop top
#     if predicted in ["top", "shirt", "t-shirt"] and is_crop_top(image_path):
#        return "crop top"

# # 5️⃣ Dress
#     if predicted == "dress":
#        return f"{length} dress"

# # 6️⃣ Kurti
#     if predicted == "kurti":
#        return f"{length} kurti"

#     return predicted



# # =================================================
# # COLOR DETECTION
# # =================================================
# def detect_color(image_path):
#     image = Image.open(image_path).convert("RGB")

#     w, h = image.size
#     crop = image.crop((
#         int(w * 0.25),
#         int(h * 0.25),
#         int(w * 0.75),
#         int(h * 0.75)
#     ))

#     crop = crop.resize((120, 120))
#     pixels = np.array(crop).reshape(-1, 3)

#     kmeans = KMeans(n_clusters=3, n_init=10)
#     labels = kmeans.fit_predict(pixels)
#     centers = kmeans.cluster_centers_

#     counts = np.bincount(labels)
#     total = counts.sum()

#     idx = np.argsort(counts)[::-1]
#     primary = centers[idx[0]]
#     primary_ratio = counts[idx[0]] / total

#     primary_name = rgb_to_name(*primary)

#     # Strong single color
#     if primary_ratio > 0.65:
#         return primary_name

#     # Compare second dominant
#     second_name = rgb_to_name(*centers[idx[1]])
#     if primary_name == second_name:
#         return primary_name

#     return "multicolour"

# # =================================================
# # RGB → COLOR NAME (ALL UNIVERSAL COLORS)
# # =================================================
# def detect_color(image_path):
#     image = Image.open(image_path).convert("RGB")

#     w, h = image.size
#     crop = image.crop((
#         int(w * 0.25),
#         int(h * 0.25),
#         int(w * 0.75),
#         int(h * 0.75)
#     ))

#     crop = crop.resize((120, 120))
#     pixels = np.array(crop).reshape(-1, 3)

#     kmeans = KMeans(n_clusters=3, n_init=10)
#     labels = kmeans.fit_predict(pixels)
#     centers = kmeans.cluster_centers_

#     dominant = centers[np.bincount(labels).argmax()]
#     return rgb_to_name(*dominant)

# # ===============================
# # RGB → COLOR NAME
# # ===============================
# def rgb_to_name(r, g, b):
#     r, g, b = int(r), int(g), int(b)
#     brightness = (r + g + b) / 3

#     # ---------- TRUE BLACK ----------
#     if brightness < 45 and abs(r - g) < 10 and abs(g - b) < 10:
#         return "black"

#     # ---------- WHITE ----------
#     if brightness > 230:
#         return "white"

#     # ---------- YELLOW ----------
#     if r > 180 and g > 180 and b < 120:
#         return "yellow"

#     # ---------- ORANGE ----------
#     if r > 180 and g > 120 and b < 100:
#         return "orange"

#     # ---------- RED / MAROON ----------
#     if r > g and r > b:
#         if r < 130:
#             return "maroon"
#         return "red"

#     # ---------- GREEN / OLIVE ----------
#     if g > r and g > b:
#         if g < 130:
#             return "olive"
#         return "green"

#     # ---------- BLUE / NAVY ----------
#     if b > r and b > g:
#         if b < 130:
#             return "navy"
#         return "blue"

#     # ---------- PURPLE / LAVENDER ----------
#     if r > 140 and b > 140:
#         if g > 140:
#             return "lavender"
#         return "purple"

#     # ---------- PINK ----------
#     if r > 180 and g > 130 and b > 150:
#         return "pink"

#     # ---------- GREY / SILVER ----------
#     if abs(r - g) < 15 and abs(g - b) < 15:
#         if brightness < 180:
#             return "grey"
#         return "silver"

#     # ---------- BEIGE / BROWN ----------
#     if r > 170 and g > 150 and b > 120:
#         return "beige"

#     if r > 120 and g < 100 and b < 80:
#         return "brown"

#     # ---------- TEAL ----------
#     if g > 120 and b > 120 and r < 100:
#         return "teal"

#     return "multicolour"


from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from sklearn.cluster import KMeans

# =================================================
# LOAD MODEL
# =================================================
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# =================================================
# ALL ALLOWED CATEGORIES (NO GROUPING)
# =================================================
CATEGORIES = [
    # Top wear
    "crop top",
    "t-shirt",
    "shirt",
    "top",
    "hoodie",
    "sweater",
    "jacket",
    "kurti",

    # Bottom wear
    "shorts",
    "skirt",
    "pants",
    "jeans",
    "trousers",
    "palazzo",

    # One-piece / ethnic
    "dress",
    "saree",
    "lehenga",
    "jumpsuit"
]

# =================================================
# LENGTH DETECTION (PRODUCT IMAGES)
# =================================================
# def detect_length(image_path):
#     image = Image.open(image_path)
#     w, h = image.size
#     ratio = h / w

#     if ratio < 1.25:
#         return "mini"
#     elif ratio < 1.85:
#         return "knee-length"
#     else:
#         return "long"

def detect_length(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    arr = np.array(image)

    # Convert to grayscale
    gray = np.mean(arr, axis=2)

    # Detect fabric pixels (ignore white/near-white background)
    fabric_rows = np.where(gray < 235)[0]

    if len(fabric_rows) == 0:
        return "unknown"

    top = fabric_rows[0]
    bottom = fabric_rows[-1]

    fabric_ratio = (bottom - top) / h

    # ✅ TUNED THRESHOLDS
    if fabric_ratio < 0.45:
        return "mini"
    elif fabric_ratio < 0.85:
        return "knee-length"
    else:
        return "long"


# =================================================
# SHAPE / STRUCTURE CHECKS
# =================================================
def is_crop_top(image_path):
    image = Image.open(image_path)
    w, h = image.size
    return (h / w) < 0.9

def is_flared(image_path):
    image = Image.open(image_path)
    w, h = image.size

    upper = image.crop((0, int(h * 0.25), w, int(h * 0.45)))
    lower = image.crop((0, int(h * 0.55), w, h))

    return np.array(lower).std() > np.array(upper).std() * 1.25

def is_saree(image_path, predicted):
    image = Image.open(image_path)
    w, h = image.size
    ratio = h / w

    if predicted not in ["saree", "dress"]:
        return False

    if ratio < 2.4:
        return False

    lower = image.crop((0, int(h * 0.55), w, h))
    return np.array(lower).std() > 55

# =================================================
# CATEGORY DETECTOR (STRICT, ORDERED RULES)
# =================================================
def detect_category(image_path):
    image = Image.open(image_path).convert("RGB")

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

    # ---------- HARD RULES (ORDER MATTERS) ----------

    # 1️⃣ Saree
    if is_saree(image_path, predicted):
        return "saree"

    # 2️⃣ Lehenga (long + flared skirt only)
    if predicted == "skirt" and length == "long" and is_flared(image_path):
        return "lehenga"

    # 3️⃣ Crop top
    if predicted in ["top", "shirt", "t-shirt"] and is_crop_top(image_path):
        return "crop top"

    # 4️⃣ Skirt
    if predicted == "skirt":
        return f"{length} skirt"

    # 5️⃣ Dress
    if predicted == "dress":
        return f"{length} dress"

    # 6️⃣ Kurti
    if predicted == "kurti":
        return f"{length} kurti"

    # 7️⃣ Everything else (hoodie, jeans, pants, etc.)
    return predicted

# # =================================================
# # COLOR DETECTION (SINGLE FUNCTION ONLY)
# # =================================================
# def detect_color(image_path):
#     image = Image.open(image_path).convert("RGB")
#     w, h = image.size

#     crop = image.crop((
#         int(w * 0.25), int(h * 0.25),
#         int(w * 0.75), int(h * 0.75)
#     ))

#     crop = crop.resize((120, 120))
#     pixels = np.array(crop).reshape(-1, 3)

#     kmeans = KMeans(n_clusters=3, n_init=10)
#     labels = kmeans.fit_predict(pixels)
#     centers = kmeans.cluster_centers_

#     dominant = centers[np.bincount(labels).argmax()]
#     return rgb_to_name(*dominant)

# # =================================================
# # RGB → UNIVERSAL COLOR NAMES
# # =================================================
# def rgb_to_name(r, g, b):
#     r, g, b = int(r), int(g), int(b)
#     brightness = (r + g + b) / 3

#     if brightness < 45:
#         return "black"
#     if brightness > 230:
#         return "white"

#     if abs(r - g) < 15 and abs(g - b) < 15:
#         return "grey"

#     if r > 180 and g > 180 and b < 120:
#         return "yellow"
#     if r > 180 and g > 120 and b < 100:
#         return "orange"
#     if r > g and r > b:
#         return "red"
#     if g > r and g > b:
#         return "green"
#     if b > r and b > g:
#         return "blue"

#     if r > 140 and b > 140:
#         return "purple"

#     if r > 170 and g > 150 and b > 120:
#         return "beige"
#     if r > 120 and g < 100 and b < 80:
#         return "brown"

#     if g > 120 and b > 120 and r < 100:
#         return "teal"

#     return "multicolour"


def detect_color(image_path):
    image = Image.open(image_path).convert("RGB")

    w, h = image.size
    crop = image.crop((
        int(w * 0.25),
        int(h * 0.25),
        int(w * 0.75),
        int(h * 0.75)
    ))

    crop = crop.resize((120, 120))
    pixels = np.array(crop).reshape(-1, 3)

    kmeans = KMeans(n_clusters=3, n_init=10)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_

    counts = np.bincount(labels)
    total = counts.sum()

    idx = np.argsort(counts)[::-1]
    primary = centers[idx[0]]
    primary_ratio = counts[idx[0]] / total

    primary_name = rgb_to_name(*primary)

    # Strong single color
    if primary_ratio > 0.65:
        return primary_name

    # Compare second dominant
    second_name = rgb_to_name(*centers[idx[1]])
    if primary_name == second_name:
        return primary_name

    return "multicolour"

# =================================================
# RGB → COLOR NAME (ALL UNIVERSAL COLORS)
# =================================================
# def detect_color(image_path):
#     image = Image.open(image_path).convert("RGB")

#     w, h = image.size
#     crop = image.crop((
#         int(w * 0.25),
#         int(h * 0.25),
#         int(w * 0.75),
#         int(h * 0.75)
#     ))

#     crop = crop.resize((120, 120))
#     pixels = np.array(crop).reshape(-1, 3)

#     kmeans = KMeans(n_clusters=3, n_init=10)
#     labels = kmeans.fit_predict(pixels)
#     centers = kmeans.cluster_centers_

#     dominant = centers[np.bincount(labels).argmax()]
#     return rgb_to_name(*dominant)

def detect_color(image_path, category=None):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    # 🔒 Saree / Lehenga: sample mid body only
    if category in ["saree", "lehenga"]:
        crop = image.crop((
            int(w * 0.2),
            int(h * 0.35),
            int(w * 0.8),
            int(h * 0.65)
        ))
    else:
        crop = image.crop((
            int(w * 0.25),
            int(h * 0.25),
            int(w * 0.75),
            int(h * 0.75)
        ))

    crop = crop.resize((120, 120))
    pixels = np.array(crop).reshape(-1, 3)

    kmeans = KMeans(n_clusters=3, n_init=10)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_

    dominant = centers[np.bincount(labels).argmax()]
    return rgb_to_name(*dominant)


# ===============================
# RGB → COLOR NAME
# ===============================
def rgb_to_name(r, g, b):
    r, g, b = int(r), int(g), int(b)
    brightness = (r + g + b) / 3

    # ---------- BLACK ----------
    if brightness < 45 and abs(r - g) < 10 and abs(g - b) < 10:
        return "black"

    # ---------- WHITE ----------
    if brightness > 230:
        return "white"

    # ---------- GREY / SILVER ----------
    if abs(r - g) < 15 and abs(g - b) < 15:
        return "grey" if brightness < 180 else "silver"

    # ---------- YELLOW / MUSTARD ----------
    if r > 170 and g > 150 and b < 120:
        return "yellow"

    # ---------- ORANGE ----------
    if r > 180 and g > 110 and b < 90:
        return "orange"

    # ---------- BROWN ----------
    if r > 120 and g < 100 and b < 80:
        return "brown"

    # ---------- RED / MAROON ----------
    if r > g and r > b:
        return "maroon" if r < 150 else "red"

    # ---------- GREEN / OLIVE ----------
    if g > r and g > b:
        return "olive" if g < 140 else "green"

    # ---------- BLUE / NAVY ----------
    if b > r and b > g:
        return "navy" if b < 100 else "blue"

    # ---------- PURPLE / LAVENDER ----------
    if r > 140 and b > 140:
        return "lavender" if g > 140 else "purple"

    # ---------- PINK ----------
    if r > 180 and g > 130 and b > 150:
        return "pink"

    # ---------- TEAL ----------
    if g > 120 and b > 120 and r < 100:
        return "teal"

    return "multicolour"

def is_clothing_item(image_path):
       # if you are using lazy loading

    image = Image.open(image_path).convert("RGB")

    check_labels = [
        
        "a garment",
    
        "an accessory",
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

    # ✅ Accept only clothing-related labels
    if top_label in ["a piece of clothing", "a garment", "fashion apparel"] and confidence > 0.45:
        return True

    return False
