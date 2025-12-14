from similarity import check_similarity

owned_items = [
    "black hoodie",
    "grey sweatshirt",
    "blue jeans",
    "palazzo pants"
]

item_to_buy = "palazzo pants"

found, matches = check_similarity(item_to_buy, owned_items)

print("Item to buy:", item_to_buy)
print("Similar found:", found)
print("Matches:")
for item, score in matches:
    print("-", item, "->", round(score, 2))
