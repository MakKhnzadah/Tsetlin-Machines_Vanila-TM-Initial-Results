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
- pyTsetlinMachine
- Matplotlib

## Running the experiment

Create a project virtual environment, install the dependencies inside it, and
run the program from the project directory. This avoids the externally managed
system Python restriction on Ubuntu:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py
```

The program reads only the existing `Dataset/20_newsgroups` directory. It first
splits documents into training and testing documents, then fits the binary
`CountVectorizer` on training documents only. Each generated pair is the
concatenation of the two document word vectors, with balanced positive and
negative examples.

The initial settings are 100 clauses, threshold `T=15`, `s=1.0`, 10 epochs,
2,000 positive and 2,000 negative pairs per split, and at most 1,000 words.
They can be changed with `--help`. Results are written to `results/` as
`epoch_metrics.csv`, `final_metrics.json`, and `accuracy_by_epoch.png`.