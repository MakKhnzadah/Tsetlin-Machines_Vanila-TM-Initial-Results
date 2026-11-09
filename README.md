# Vanilla Tsetlin Machine – Initial Results

## Project Description

This project uses a Vanilla Tsetlin Machine (TM) to classify pairs of text documents from the **20 Newsgroups dataset**.

The goal is to predict whether two documents belong to the same category or different categories using **Reasoning by Elimination (RbE)**.

## Dataset

- **Dataset:** 20 Newsgroups (20 categories)
- **Input:** Two documents represented as binary word features
- **Target:**
  - `Y = 1`: Both documents belong to the same category.
  - `Y = 0`: Documents belong to different categories.

## Method

1. Load the 20 Newsgroups dataset.
2. Randomly select pairs of documents.
3. Convert documents into binary feature vectors.
4. Train a Vanilla Tsetlin Machine with `s = 1.0`.
5. Evaluate the model's classification accuracy.

## Objective

Investigate how a Vanilla Tsetlin Machine uses **Reasoning by Elimination** to classify document pairs.

## Technologies

- Python
- Scikit-learn
- Tsetlin Machine


