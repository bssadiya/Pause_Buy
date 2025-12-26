
# PauseBuy — Smart Closet Tool

**Pause before you buy. Avoid duplicate clothing purchases.**

PauseBuy checks whether you already own similar clothes **before you buy**, using **category-based matching and image similarity**.


## ✨ Features
**Digital Closet** — Upload clothes you own
**Can I Buy This?** — Upload an image and check
**Flipkart Alerts** — Warns on Buy Now / Add to Cart
**Add from Orders**
**Duplicate Prevention**

##  How It Works

1. Detects if the image is clothing
2. Matches **only same clothing categories** (Top / Bottom / Dress)
3. Uses **image similarity** for final validation
4. Shows matching closet images if found

## 🛠 Technologies Used

**Backend**

* Python, Flask, Flask-CORS

**AI / ML**

* OpenAI CLIP (ViT-B/32)
* Cosine Similarity
* Perceptual Hashing (pHash)

**Image Processing**

* OpenCV, Pillow, NumPy

**Frontend**

* HTML, CSS, Jinja2

**Browser Extension**

* JavaScript, Chrome Extension APIs

**Storage**

* JSON (closet data), Local image storage

* **Exact accuracy:** 73.68%
* **Similarity accuracy:** 94.74%

Similarity accuracy is more meaningful for real buying decisions.

## Tech Stack
* Python
* Flask
* Image processing
* Semantic similarity

## Output


https://github.com/user-attachments/assets/c9572cb0-eb86-4e21-a8e3-9542cb3db2ec



## Summary

PauseBuy is a practical project focused on reducing unnecessary purchases by helping users pause and reflect before buying.
