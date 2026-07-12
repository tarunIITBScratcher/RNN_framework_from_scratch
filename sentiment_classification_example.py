"""
Example: sentiment analysis (many-to-one) with rnn_framework.

Builds a tiny synthetic "reviews" dataset in-file - each review is a
handful of neutral filler words with exactly one positive or negative
sentiment word planted at a random position - then trains
Embedding -> LSTM -> Dense -> Sigmoid on it, and finally checks the
trained model against a few brand-new sentences it never saw during
training.

Swap Layer_LSTM(16) for Layer_SimpleRNN(16) or Layer_GRU(16) below to
compare cells on the exact same task - nothing else in this script needs
to change.

Run from the repo root:
    pip install -r requirements.txt
    python examples/sentiment_classification.py
"""

import numpy as np

from rnn_framework import (
    Layer_Embedding,
    Layer_LSTM,
    Layer_Dense,
    Activation_Sigmoid,
    Loss_BinaryCrossentropy,
    Optimizer_ADAM,
    Model,
)

np.random.seed(0)

# ---------------------------------------------------------------------
# 1. Build a tiny synthetic dataset
# ---------------------------------------------------------------------
# Neutral filler words that carry no sentiment on their own - every
# review is padded out with a handful of these.
FILLER_WORDS = [
    "the", "movie", "was", "a", "film", "this", "acting", "story",
    "plot", "scene", "director", "really", "very", "quite", "just",
    "overall", "honestly", "definitely", "watched", "saw",
]

POSITIVE_WORDS = ["great", "amazing", "wonderful", "fantastic", "loved", "brilliant"]
NEGATIVE_WORDS = ["terrible", "awful", "boring", "horrible", "hated", "disappointing"]

SEQ_LEN = 8


def make_review(sentiment_word):
    """One review: SEQ_LEN filler words with sentiment_word swapped in
    at a random position."""
    words = list(np.random.choice(FILLER_WORDS, size=SEQ_LEN, replace=True))
    position = np.random.randint(0, SEQ_LEN)
    words[position] = sentiment_word
    return words


def build_dataset(n_samples):
    reviews = []
    labels = []
    for _ in range(n_samples):
        if np.random.rand() < 0.5:
            reviews.append(make_review(np.random.choice(POSITIVE_WORDS)))
            labels.append(1)
        else:
            reviews.append(make_review(np.random.choice(NEGATIVE_WORDS)))
            labels.append(0)
    return reviews, np.array(labels)


N_SAMPLES = 1000
reviews, y = build_dataset(N_SAMPLES)

# ---------------------------------------------------------------------
# 2. Build the word -> index vocabulary (done outside the framework -
#    Layer_Embedding only ever sees integer token ids, never raw text)
# ---------------------------------------------------------------------
vocab = sorted(set(FILLER_WORDS + POSITIVE_WORDS + NEGATIVE_WORDS))
word_to_index = {word: idx for idx, word in enumerate(vocab)}
vocab_size = len(vocab)


def reviews_to_ids(reviews):
    return np.array([[word_to_index[w] for w in review] for review in reviews])


X = reviews_to_ids(reviews)

# ---------------------------------------------------------------------
# 3. Build and train the model: Embedding -> LSTM -> Dense -> Sigmoid
# ---------------------------------------------------------------------
model = Model(
    [
        Layer_Embedding(vocab_size, embedding_dim=16),
        Layer_LSTM(16),
        Layer_Dense(1),
        Activation_Sigmoid(),
    ]
)

model.compile(
    loss=Loss_BinaryCrossentropy(),
    optimizer=Optimizer_ADAM(learning_rate=0.01),
)

model.fit(X, y, epochs=40, batch_size=32, validation_split=0.2)


# ---------------------------------------------------------------------
# 4. Try the trained model on brand-new sentences
# ---------------------------------------------------------------------
def predict_sentiment(sentence_words):
    """
    Run a single new review (a list of words) through the trained model.

    model.forward(X) after compile() returns raw logits, not
    probabilities - compile() popped the standalone Activation_Sigmoid
    layer off self.layers, since the fused loss performs that activation
    internally during training. So to get an actual probability back
    out, we run the logits through a fresh Activation_Sigmoid() here.
    """
    ids = np.array([[word_to_index[w] for w in sentence_words]])
    logits = model.forward(ids)
    probability = Activation_Sigmoid().forward(logits)
    return float(probability[0, 0])


test_sentences = [
    ["the", "movie", "was", "really", "amazing", "this", "story"],
    ["this", "film", "was", "terrible", "quite", "boring", "honestly"],
    ["honestly", "the", "acting", "was", "brilliant", "this", "film"],
    ["the", "plot", "was", "awful", "disappointing", "overall", "story"],
]
# every word above comes from FILLER_WORDS/POSITIVE_WORDS/NEGATIVE_WORDS,
# so it's guaranteed to already be in word_to_index

print("Predictions on brand-new sentences (probability of positive sentiment):")
for sentence in test_sentences:
    p = predict_sentiment(sentence)
    label = "positive" if p > 0.5 else "negative"
    print(f"  {' '.join(sentence):55s} -> {p:.3f} ({label})")
