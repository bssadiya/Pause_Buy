from sentence_transformers import SentenceTransformer, util

# Load pretrained model
model = SentenceTransformer("all-MiniLM-L6-v2")

def check_similarity(item_to_buy, owned_items, threshold=0.75):
    if not owned_items:
        return False, []

    buy_embedding = model.encode(item_to_buy, convert_to_tensor=True)
    owned_embeddings = model.encode(owned_items, convert_to_tensor=True)

    scores = util.cos_sim(buy_embedding, owned_embeddings)[0]

    similar_items = []
    for i, score in enumerate(scores):
        if score >= threshold:
            similar_items.append((owned_items[i], float(score)))

    return len(similar_items) > 0, similar_items
