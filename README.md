# Privacy_Guard
PrivacyGuard is an NLP-based privacy protection system that detects, classifies, and automatically redacts Personal Identifiable Information (PII) from user-generated text. It compares rule based methods, classical ML models, BiLSTM sequence taggers, and Transformer based approaches for robust privacy detection in clean and noisy multilingual text.

## Implemented stages

- Stage 1: PII-safe preprocessing and synthetic BIO-tagged train/validation/test data.
- Stage 2: custom window BoW/TF-IDF and character n-gram features, binary
  Logistic Regression and multiclass Naive Bayes implemented from scratch,
  BIO reconstruction, and shared token/span evaluation.
- Stage 3: NumPy Skip-gram embeddings, mean and TF-IDF-weighted window
  embeddings, optional pretrained Gensim comparison, and embedding-based
  Logistic Regression evaluation through the unchanged Stage 2 pipeline.

## Run Stage 2

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\train_classical.py
```

The complete machine-readable evaluation is written to
`results/stage2_classical_results.json`.

## Run Stage 3

```powershell
python scripts/train_embeddings.py
```

The complete machine-readable evaluation is written to
`results/stage3_embedding_results.json`. Pretrained Gensim vectors are
optional; a failed download is reported and the scratch-embedding arms still
finish.
