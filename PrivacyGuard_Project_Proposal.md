# PrivacyGuard: Context-Aware Detection and Redaction of Personally Identifiable Information in User-Generated Text

### Lab-Aligned Revision

## What will be done

Modern users frequently share sensitive information such as phone numbers,
email addresses, names, locations, account identifiers, and financial
details while interacting with AI assistants, chatbots, customer-support
systems, emails, online forms, and other digital platforms. However, users
often do not realize that this information can create privacy risks when
processed by automated systems.

We will build **PrivacyGuard**, an NLP-based privacy protection system that
detects, classifies, and automatically redacts Personally Identifiable
Information (PII) from user-generated text before it is shared with
external applications.

This revision restructures the project so that **every stage of the
implementation directly exercises a technique taught in the course labs**,
rather than treating PII detection and lab coursework as separate tracks.
The result is a model ladder built almost entirely from first principles:

    Regex + Edit-Distance Preprocessing
            ↓
    Classical ML (TF-IDF/BoW + Naive Bayes/Logistic Regression, from scratch)
            ↓
    Word Embeddings (Word2Vec from scratch + pretrained embeddings)
            ↓
    Sequence Tagging (BiRNN / BiLSTM, adapted from POS tagging)
            ↓
    Transformer Sequence Tagging (encoder-only, self-attention)

Each rung is evaluated on the same held-out test data using the same
metrics, so the project produces a genuine empirical comparison — not just
five separate demos.

------------------------------------------------------------------------

## Alignment with Course Lab Modules

| Lab | Techniques Taught | Where It Is Used in PrivacyGuard |
|---|---|---|
| **Lab 1** | Regex text cleaning, tokenization, stop word removal, Porter stemming, WordNet lemmatization, Edit Distance (Levenshtein DP) spelling correction | PII-safe preprocessing pipeline; edit distance corrects informal PII-adjacent noise (e.g. "nmbr" → "number") ahead of detection |
| **Lab 2** | Bag of Words, TF-IDF, N-gram language modeling | Feature extraction for the classical ML detector; character n-grams for structured PII patterns (phone/email/account) |
| **Lab 3** | Word2Vec (Skip-gram) from scratch, Naive Bayes from scratch, Logistic Regression from scratch, TF-IDF-weighted embeddings | The classical-ML rung of the model ladder, implemented from first principles rather than only via `sklearn` |
| **Lab 4** | Pretrained embeddings in PyTorch, Vanilla RNN vs. Stacked BiLSTM, **Bidirectional RNN sequence labeling (POS tagging)**, Seq2Seq | Directly repurposed: the BiRNN POS-tagging architecture becomes the **token-level BIO PII tagger** — the same input→embedding→BiRNN/BiLSTM→per-token classifier pattern, retrained on PII tags instead of POS tags |
| **Lab 5** | Encoder-only Transformer (`nn.TransformerEncoder`) for sequence classification | Adapted from sentence-level (mean-pooled) classification to **per-token classification** by removing the pooling step, giving a self-attention-based BIO tagger |

------------------------------------------------------------------------

## Implementation Plan (6 Stages)

**Stage 1 — Preprocessing & Synthetic BIO Dataset** *(complete)*
PII-safe regex/unicode cleaning, tokenization, stopword/stemming/
lemmatization utilities (Lab 1), and an edit-distance spelling corrector
restricted to non-numeric tokens. Since an external annotated PII corpus is
not accessible in this environment, a synthetic BIO-tagged dataset is
generated procedurally (in the spirit of Lab 5's synthetic sentiment
corpus): entity generators for eight PII categories, sentence templates
across three noise conditions (clean / Banglish code-mixed / noisy
informal), character-span tracking through template filling, and BIO tag
alignment via a span-aware tokenizer. Output: deduplicated, leakage-free
train/val/test JSONL splits.

**Stage 2 — Classical ML Detector**
BoW and TF-IDF vectorization (Lab 2), including character n-grams for
structured PII patterns. Naive Bayes and Logistic Regression implemented
from scratch with NumPy (Lab 3), applied at the token/window level to
classify spans as PII vs. non-PII and assign a category.

**Stage 3 — Word Embeddings & Embedding-Based Classification**
Skip-gram Word2Vec trained from scratch on the project corpus (Lab 3),
compared against pretrained embeddings loaded into PyTorch (Lab 4). Mean
and TF-IDF-weighted document embeddings are built and re-evaluated through
the same Logistic Regression classifier to isolate the effect of
representation choice, holding the classifier constant.

**Stage 4 — Sequence Tagging with BiRNN/BiLSTM**
The Lab 4 POS-tagging architecture (embedding → bidirectional RNN/LSTM →
per-token linear layer) is retrained on BIO PII tags. A Vanilla RNN and a
Stacked BiLSTM are both trained and compared, matching Lab 4's own
RNN-vs-BiLSTM contrast.

**Stage 5 — Transformer-Based Sequence Tagging**
The Lab 5 encoder-only Transformer is modified to output per-token logits
instead of a pooled sentence-level class, turning it into a self-attention
BIO tagger — the top rung of the model ladder.

**Stage 6 — Evaluation, Robustness Testing & Final Report**
All models (rule-based baseline, classical ML, BiRNN/BiLSTM, Transformer)
are run through one shared evaluation harness and compared on identical
test data, broken out by clean/Banglish/noisy subsets.

------------------------------------------------------------------------

# How it will be done

## Corpus and Dataset

A synthetic, procedurally generated corpus with token-level BIO annotations
is used for training and evaluation. The corpus covers:

-   PERSON
-   PHONE_NUMBER
-   EMAIL
-   ADDRESS
-   ACCOUNT_NUMBER
-   FINANCIAL_INFORMATION
-   ORGANIZATION
-   LOCATION

Generation approach:

1.  Entity-specific generator functions produce realistic fake values per
    category (e.g. Bangladeshi-style mobile numbers, plausible email
    addresses, English and South Asian names/organizations/locations).
2.  Sentence templates in three groups — clean English, Bengali-English
    code-mixed ("Banglish"), and noisy/informal — contain `{LABEL}`
    placeholders.
3.  While filling a template, the exact character span of each inserted
    entity value is tracked.
4.  The final sentence is tokenized with a span-aware tokenizer, and BIO
    tags are assigned per token by checking overlap against tracked entity
    spans (first overlapping token → `B-LABEL`, subsequent → `I-LABEL`,
    otherwise `O`).

Exact-duplicate sentences are removed before splitting into train,
validation, and test sets, so identical examples cannot appear across
splits.

If a real annotated PII corpus (e.g. a public Hugging Face dataset) is
available in the environment where this project is ultimately trained, it
can be added as a **supplementary evaluation set** to test whether models
trained on synthetic data generalize to real-world text — this is itself a
valuable robustness experiment for the report.

------------------------------------------------------------------------

## Data Preparation and Robustness Testing

Noise conditions generated directly into the dataset (rather than applied
as a post-hoc corruption step, so BIO alignment stays exact):

-   Bengali-English code mixing ("amar bkash nmbr {PHONE_NUMBER}")
-   Missing punctuation and informal abbreviations ("plz", "asap", "frm")
-   Multiple PII value formats (e.g. phone numbers written plain, spaced,
    or dashed)

Example:

    Clean:    "Please contact John Smith at john@example.com or call 01712345678."
    Banglish: "vaia amar bkash nmbr 018 3194 8757, taka pathai den"
    Noisy:    "hey its Maria Hasan here reach me 01625-276018 or james.akter41@outlook.com thanks"

Performance is compared across all three groups to evaluate whether
PrivacyGuard degrades gracefully on realistic, messy input rather than only
performing well on clean benchmark text.

------------------------------------------------------------------------

## Preprocessing

Implemented directly from Lab 1's toolkit, with a PII-safety adjustment:

-   Unicode normalization (NFKC)
-   HTML tag stripping
-   Whitespace normalization
-   Emoji handling (remove or tag)
-   URL handling (keep or tag)
-   Tokenization (NLTK `word_tokenize`)
-   Optional stop word removal, stemming (Porter), lemmatization (WordNet)
    — off by default, since PII detection depends on case and function
    words that aggressive preprocessing would destroy
-   Edit-distance spelling correction restricted to non-numeric tokens
    against a small PII-context vocabulary (e.g. "nmbr" → "number")

Different preprocessing configurations (aggressive vs. minimal) are
compared later to measure their effect on detection performance, directly
testing the assumption that lab-style aggressive cleaning would hurt PII
recall.

------------------------------------------------------------------------

## Models

### Baseline: Rule-Based + NER Detector

A regex-based detector (structured patterns: phone, email, account,
financial, URL) combined with a pretrained NER model (PERSON, ORGANIZATION,
LOCATION) provides a fast, interpretable floor for comparison. This layer
is also the practical component wrapped in the CLI for interactive/demo
use.

### Classical Machine Learning Models (Lab 2 + Lab 3)

TF-IDF and BoW features (unigram, bigram, character n-grams) feed into
Naive Bayes and Logistic Regression classifiers implemented from scratch
with NumPy, following the exact training loop structure from Lab 3
(log-prior/log-likelihood computation with Laplace smoothing for Naive
Bayes; gradient descent on the sigmoid/log-loss for Logistic Regression).
Character n-grams specifically target structured PII patterns.

### Word Embeddings (Lab 3 + Lab 4)

A Skip-gram Word2Vec model is trained from scratch (forward pass, softmax,
cross-entropy loss, backpropagation via NumPy) on the project corpus.
Pretrained embeddings are also loaded into PyTorch for comparison. Both
mean-pooled and TF-IDF-weighted document/token embeddings are evaluated
through the same Logistic Regression classifier to isolate whether richer
representations improve detection independent of the classifier.

### RNN and LSTM Models (Lab 4)

Following Lab 4's Vanilla RNN vs. Stacked BiLSTM comparison, a Bidirectional
RNN sequence labeler — architecturally identical to the lab's POS-tagging
model — is retrained to predict BIO PII tags per token instead of
part-of-speech tags. This is the first model in the ladder to perform true
token-level sequence tagging with learned context.

### Transformer Models (Lab 5)

The Lab 5 encoder-only Transformer (embedding + positional encoding +
`nn.TransformerEncoder`) is adapted from sentence classification to
sequence tagging by outputting per-token logits directly from the encoder
output instead of mean-pooling into a single vector. Padding masks prevent
attention from leaking through `<PAD>` tokens, exactly as in the lab.

------------------------------------------------------------------------

# Evaluation

The primary evaluation metric is **Macro-F1**, since PII categories have
different frequencies and rare categories should not be ignored.

The system is evaluated using:

-   Precision
-   Recall
-   Macro F1-score
-   Weighted F1-score
-   Token-level BIO F1
-   Exact span matching score
-   Per-category performance
-   Confusion matrix
-   Training time
-   Inference time
-   Model size

Every model in the ladder is scored against the same test split, broken out
by the three noise groups (clean / Banglish / noisy), so degradation under
realistic conditions is directly comparable across the rule-based, classical
ML, BiRNN/BiLSTM, and Transformer stages.

------------------------------------------------------------------------

# Validation and Error Analysis

Before accepting model performance, the system is tested for shortcuts and
generalization failures. Early exploration of the pipeline has already
surfaced concrete failure cases worth carrying into this section:

-   A pretrained English NER model tagging the Bengali word "amar" ("my")
    as `PERSON`, illustrating the need for the planned Bengali/Banglish
    robustness testing rather than relying on English-only pretrained
    models.
-   A generic phone-number regex misclassifying a 10-digit account number
    as `PHONE_NUMBER`, showing that structured-pattern regexes need
    tighter category-specific constraints, not just length matching.
-   The Treebank tokenizer splitting an email address into three tokens
    (`james.akter41`, `@`, `outlook.com`), which affects how cleanly BIO
    spans can be recovered — a concrete data point for the preprocessing
    comparison.
-   The edit-distance corrector leaving "plz" uncorrected at a distance
    threshold of 2, versus correctly fixing "nmbr" → "number", showing the
    threshold itself is a tunable parameter worth reporting.

Planned comparisons for the final analysis:

-   Regex-based detection versus machine-learning approaches.
-   Character n-grams versus dense embeddings versus Transformer
    attention.
-   Classical models versus BiRNN/BiLSTM versus Transformer on identical
    noisy test data.
-   Effect of aggressive vs. minimal preprocessing on Macro-F1.

------------------------------------------------------------------------

# Expected Outcome

The final system detects sensitive information, identifies its PII
category, highlights the detected span, and optionally replaces it with a
safe placeholder — accessible through a command-line interface backed by
the full model ladder.

PrivacyGuard provides a systematic, lab-grounded comparison between
classical NLP methods, sequential neural models, and Transformer-based
approaches, while investigating privacy protection in multilingual and
noisy user-generated text — with every modeling technique traceable back to
a specific taught lab exercise.

The final prototype can serve as a practical privacy layer for AI
applications, chat systems, and online communication platforms, and as a
complete demonstration of the course's NLP curriculum applied to a single,
coherent real-world problem.
