# DRUG REVIEW CLASSIFICATION AND ANALYSIS

# Model Selection for Drug Review Sentiment Classification

## Overview

We select 4 models, from different domains, for two reasons:

1. We want to be thorough in our base model selection
2. It is interesing to see the effect of the domian on the accuracy (is it better to use medical based models?, or sentimen analysis based models?)

---

## Model 1 — RoBERTa (`roberta-base`)

**Role: General-purpose baseline**

RoBERTa (Robustly Optimized BERT Pretraining Approach) is a refined version of BERT developed by Facebook AI in 2019. It improves on BERT by training longer, on more data, with larger batches, and removing the Next Sentence Prediction (NSP) objective — which was shown to hurt rather than help downstream tasks.

**Why we chose it:**
- Serves as the **general NLP baseline** with no domain or sentiment bias
- Strong tokenizer and robust pretraining make it a reliable reference point
- Any gains from the other three models can be measured relative to it
- Widely used in NLP benchmarks, making results easy to contextualize

**HuggingFace ID:** `roberta-base`

---

## Model 2 — Bio-ClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`)

**Role: Biomedical domain specialist**

Bio-ClinicalBERT was developed at MIT and trained on clinical notes from the MIMIC-III dataset — a large corpus of de-identified electronic health records from the Beth Israel Deaconess Medical Center. It builds on BioBERT, which was itself pretrained on PubMed abstracts and full-text biomedical articles.

**Why we chose it:**
- Drug reviews frequently contain **clinical vocabulary**: medication names, dosages, side effects, symptoms, and medical conditions
- Standard BERT-based models are not exposed to this language during pretraining and may misinterpret or underweight domain-specific terms
- Allows us to test whether **clinical domain knowledge transfers** to patient-written reviews
- Notable caveat: MIMIC-III contains *clinician-written* notes, not patient language — this potential mismatch is itself an interesting experimental variable

**HuggingFace ID:** `emilyalsentzer/Bio_ClinicalBERT`

---

## Model 3 — DeBERTa (`microsoft/deberta-v3-base`)

**Role: Architectural advancement**

DeBERTa (Decoding-Enhanced BERT with Disentangled Attention) was introduced by Microsoft Research in 2021. It introduces two key innovations over BERT and RoBERTa:

1. **Disentangled attention mechanism** — content and position embeddings are kept separate and their interactions are computed independently, giving the model a richer representation of how words relate to each other based on both meaning and position
2. **Enhanced mask decoder** — improves masked language model pretraining by incorporating absolute position information during decoding

DeBERTa-v3 further improves on this with ELECTRA-style replaced token detection pretraining, making it significantly more sample-efficient.

**Why we chose it:**
- Consistently outperforms RoBERTa on classification benchmarks (GLUE, SuperGLUE)
- Particularly strong on **short text classification**, which matches our use case (<200 words per review)
- Allows us to isolate whether **architectural improvements** yield gains independent of domain adaptation

**HuggingFace ID:** `microsoft/deberta-v3-base`

---

## Model 4 — Sentiment RoBERTa Large (`siebert/sentiment-roberta-large-english`)

**Role: Sentiment-pretraining specialist**

This model was developed by Siebert & Moreno (2022) and is a RoBERTa-large model fine-tuned specifically for English sentiment analysis. Crucially, it was trained on a **diverse, multi-domain sentiment corpus** covering 15 different datasets including product reviews, movie reviews, tweets, and news — making it one of the most broadly trained sentiment models available.

**Why we chose it:**
- Drug reviews are fundamentally **opinion-bearing texts**: patients express satisfaction, frustration, relief, or disappointment — all of which are sentiment signals
- Pretraining on multi-domain sentiment data may give it an advantage in recognizing the affective language patterns common in review text
- Using `large` over `base` gives it higher representational capacity, providing an upper-bound estimate on what sentiment pretraining can achieve
- The comparison with `roberta-base` (Model 1) directly isolates the effect of **sentiment-specific fine-tuning** on the same architecture family

**HuggingFace ID:** `siebert/sentiment-roberta-large-english`

---

## Summary Table

| # | Model | HuggingFace ID | What it tests |
|---|---|---|---|
| 1 | RoBERTa | `roberta-base` | General NLP baseline |
| 2 | Bio-ClinicalBERT | `emilyalsentzer/Bio_ClinicalBERT` | Biomedical domain adaptation |
| 3 | DeBERTa v3 | `microsoft/deberta-v3-base` | Architectural improvements |
| 4 | Sentiment RoBERTa Large | `siebert/sentiment-roberta-large-english` | Sentiment-specific pretraining |

---


All models will be evaluated using **macro F1-score** to account for potential class imbalance across bad / neutral / good labels, with per-class F1 also reported.