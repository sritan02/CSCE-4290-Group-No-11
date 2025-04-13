#Tweet Sentiment Classification and Financial Summarization using RoBERTa and BART

Course: Natural Language Processing

#Project Overview

This project presents a two-stage NLP pipeline that automates the classification and summarization of financial tweets. The objective is to assist financial analysts by delivering concise, sentiment-aware summaries for stocks and companies based on public discourse on social media platforms like Twitter.

Stage 1: Fine-tuned RoBERTa model classifies tweets into sentiment classes (Positive, Negative, Neutral).
Stage 2: A fine-tuned BART summarization model generates investment recommendations based on sentiment-filtered tweets.

#Problem Statement

Sentiment Detection Challenges
Financial tweets often contain domain-specific jargon, sarcasm, or informal syntax.
Imbalanced datasets skew predictions toward majority sentiment classes.
RoBERTa was fine-tuned with:
Emoji, ticker, and hashtag removal
Lemmatization and normalization
Class weighting and label smoothing
5-fold cross-validation
Information Overload
Analysts face challenges in sifting through massive tweet volumes.
BART was used to summarize sentiment-filtered tweets into stock-specific insights.

#Dataset

Sentiment Classification (RoBERTa)
Combined sent_train.csv with Financial PhraseBank.
Rebalanced to ~6000 samples per class.
Labels: 0 = Negative, 1 = Neutral, 2 = Positive
Advanced cleaning and label unification applied.
Summarization Dataset (BART)
No open-source dataset existed for financial tweet-to-summary mapping.
Curated manually and semi-automatically using outputs from RoBERTa.
Final dataset contains clean formatted_input and summary_text.

#Labels

Negative: Bearish tone or loss-driven language
Neutral: Informational or factual updates
Positive: Bullish sentiment or gain-related announcements
Label quality and distribution validated through word clouds and hashtag frequency analysis.

#Methodology

RoBERTa Fine-Tuning
Model: roberta-base
Optimizer: AdamW
Loss Function: Weighted Cross Entropy with label smoothing
F1 Score: ~0.89 | AUC Macro: 0.94

Named Entity Recognition (NER)
Extracted stock/company names from tweets using dslim/bert-base-NER
New column stock_name added to enhance summarization context

BART Summarization
Model: facebook/bart-base
Input format:
News: <clean_text>  
Q: What should an investor do?
Output: Abstract summaries like “Tesla is under pressure. Consider selling.”
Trained over 10 epochs with strong ROUGE and validation metrics

#Pipeline Integration

RoBERTa classifies tweets into sentiment classes.
Tweets with Positive or relevant Neutral sentiment are selected.
NER extracts stock names from each tweet.
BART takes cleaned tweet and stock name to generate a recommendation.

#Final Output Columns

raw_text: Original tweet or financial news
clean_text: Cleaned and normalized input
sentiment and action: Classification output
stock_name: Extracted via NER
formatted_input: Prompt used for summarization
generated_summary: BART output (e.g., “Sell Tesla.”)
is_legit_ticker: Indicates validity of the extracted entity

#Models Compared

Model	Strengths	Weaknesses
Naive Bayes	Fast, simple	Struggles with context, sarcasm
Linear SVM	Good for high-dimension text	Lacks deep semantic understanding
Logistic Regression	Strong baseline	Bag-of-words limits contextual depth
Random Forest	Non-linear, ensemble method	Computationally expensive for text
DistilBERT	Lightweight, fast inference	Lower F1, underperforms on sarcasm
FinBERT	Financial-domain specific	Less flexible and subtle than RoBERTa
RoBERTa	Best accuracy & generalization	Requires more computation than NB or SVM

#Libraries and Tools

Hugging Face Transformers (RoBERTa, BART, NER)
Scikit-learn, NLTK, spaCy
Pandas, NumPy
PyTorch
Google Colab and Google Drive
Results

#RoBERTa

Accuracy: 91.2%
F1 Score: 0.89
AUC Macro: 0.94
BART

ROUGE scores show high overlap with ground truth
Final summaries are stock-specific and actionable

#Conclusion

By integrating RoBERTa and BART with a robust preprocessing and NER pipeline, this system turns noisy financial tweets into actionable investment advice. It is accurate, scalable, and adaptable to real-world financial NLP applications.
