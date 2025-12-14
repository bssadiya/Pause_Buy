import csv
from image_labeler import detect_category
from similarity import check_similarity

CSV_FILE = "ground_truth.csv"

total = 0
exact_correct = 0
similar_correct = 0

with open(CSV_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        image_path = row["image_path"]
        true_category = row["true_category"].lower()

        raw_pred = detect_category(image_path).lower()
        predicted_category = raw_pred.split()[-1]

        total += 1

        # Exact match
        if predicted_category == true_category:
            exact_correct += 1
            similar_correct += 1
            status = "✅ EXACT"
        else:
            # Similarity check
            found, _ = check_similarity(
                predicted_category,
                [true_category]
            )
            if found:
                similar_correct += 1
                status = "🟡 SIMILAR"
            else:
                status = "❌ WRONG"

        print(f"{image_path}")
        print(f"  True: {true_category}")
        print(f"  Pred: {predicted_category}")
        print(f"  Result: {status}")
        print("-" * 40)

print("\n====================")
print(f"Total images: {total}")
print(f"Exact Accuracy: {(exact_correct / total) * 100:.2f}%")
print(f"Similarity Accuracy: {(similar_correct / total) * 100:.2f}%")
