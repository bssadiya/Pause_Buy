import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from image_labeler import detect_category

# ===== MODELS =====
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def crop_for_similarity(image, category):
    w, h = image.size
    cat = category.lower()

    if any(k in cat for k in ["top", "t-shirt", "shirt", "crop", "hoodie", "sweater", "jacket"]):
        return image.crop((int(w*0.2), int(h*0.08), int(w*0.8), int(h*0.5)))

    if any(k in cat for k in ["dress", "kurti", "jumpsuit"]):
        return image.crop((int(w*0.2), int(h*0.1), int(w*0.8), int(h*0.7)))

    if any(k in cat for k in ["pants", "jeans", "trousers", "palazzo", "shorts", "skirt"]):
        return image.crop((int(w*0.2), int(h*0.45), int(w*0.8), int(h*0.9)))

    if any(k in cat for k in ["saree", "lehenga"]):
        return image.crop((int(w*0.15), int(h*0.3), int(w*0.85), int(h*0.9)))

    return image


# =================================================
# IMAGE SIMILARITY (FIXED)
# =================================================
def check_image_similarity(product_path, owned_image_paths, threshold=0.80):
    product_cat = detect_category(product_path)
    product_img = Image.open(product_path).convert("RGB")
    product_img = crop_for_similarity(product_img, product_cat)

    prod_inputs = clip_processor(images=product_img, return_tensors="pt")
    with torch.no_grad():
        prod_emb = clip_model.get_image_features(**prod_inputs)
    prod_emb = prod_emb / prod_emb.norm(dim=-1, keepdim=True)

    matches = []

    for path in owned_image_paths:
        cat = detect_category(path)
        if cat != product_cat:
            continue

        img = Image.open(path).convert("RGB")
        img = crop_for_similarity(img, cat)

        inputs = clip_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            emb = clip_model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)

        score = torch.cosine_similarity(prod_emb, emb).item()
        if score >= threshold:
            matches.append((path, score))

    return matches   #  return list, not boolean
