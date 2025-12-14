# PauseBuy

## Problem

We often forget what clothes we already own and end up buying very similar items again.
There is no simple way to check *“Do I already have something like this?”* before buying.


## What PauseBuy Does

PauseBuy helps users decide whether they should buy a clothing item by comparing it with their existing clothes.
Users upload images of clothes they own to create a personal closet.
Before buying something new, they can enter a description or upload an image, and the system checks for similar items in their closet.
## Key Features
* Upload clothing images to build a personal closet
* Automatically detects:
  * Clothing category
  * Length (mini / knee-length / long)
  * Color
* Supports text-based and image based checks
* Detects similarity, not just exact matches
* Rejects non-clothing images

## Categories Supported
crop top, t-shirt, shirt, top, hoodie, sweater, jacket, kurti,
shorts, skirt, pants, jeans, trousers, palazzo,
dress, saree, lehenga, jumpsuit

## Evaluation

Tested on a custom clothing image dataset.

* **Exact accuracy:** 73.68%
* **Similarity accuracy:** 94.74%

Similarity accuracy is more meaningful for real buying decisions.

## Tech Stack
* Python
* Flask
* Image processing
* Semantic similarity

## Output

https://github.com/user-attachments/assets/a3135242-390c-40af-a0d8-c619b69ad6d1

## Summary

PauseBuy is a practical project focused on reducing unnecessary purchases by helping users pause and reflect before buying.
