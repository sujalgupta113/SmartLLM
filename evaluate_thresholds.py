
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Each pair contains:
# (question A, question B, should_be_similar)
#
# True means their intended meanings match.
# False means they should not share a cached answer.

test_pairs = [
    # Exact repeats
    ("What is Python?", "What is Python?", True),

    # Paraphrases
    ("What is Python?", "Explain Python to me", True),
    (
        "How can I learn Python?",
        "What is the best way to learn Python?",
        True
    ),
    (
        "What is machine learning?",
        "Explain machine learning",
        True
    ),

    # Clearly unrelated questions
    (
        "What is Python?",
        "How do I bake a chocolate cake?",
        False
    ),
    (
        "What is machine learning?",
        "What is the capital of Japan?",
        False
    ),

    # Similar wording, different meaning
    (
        "How do I reset my password?",
        "How do I change my password?",
        False
    ),
    (
        "How do I delete my account?",
        "How do I deactivate my account?",
        False
    ),
    (
        "How can I increase my bank balance?",
        "How can I check my bank balance?",
        False
    ),
]

# Encode every question once.
all_questions = []
for q1, q2, _ in test_pairs:
    all_questions.extend([q1, q2])

embeddings = model.encode(
    all_questions,
    normalize_embeddings=True
)

# Calculate the similarity for each pair.
scores = []

for i, (q1, q2, should_match) in enumerate(test_pairs):
    score = float(
        cosine_similarity(
            embeddings[2 * i].reshape(1, -1),
            embeddings[2 * i + 1].reshape(1, -1)
        )[0][0]
    )

    scores.append((q1, q2, should_match, score))

print("\nQUESTION PAIR RESULTS")
print("=" * 75)

for q1, q2, should_match, score in scores:
    expected = "SIMILAR" if should_match else "DIFFERENT"

    print(f"\nQuestion A: {q1}")
    print(f"Question B: {q2}")
    print(f"Expected:   {expected}")
    print(f"Similarity: {score:.3f}")


print("\nTHRESHOLD EVALUATION")
print("=" * 75)

thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

for threshold in thresholds:
    tp = fp = fn = tn = 0

    for _, _, should_match, score in scores:
        predicted_match = score >= threshold

        if predicted_match and should_match:
            tp += 1
        elif predicted_match and not should_match:
            fp += 1
        elif not predicted_match and should_match:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0
    )

    print(
        f"Threshold={threshold:.2f} | "
        f"TP={tp} FP={fp} FN={fn} TN={tn} | "
        f"Precision={precision:.2f} | "
        f"Recall={recall:.2f} | F1={f1:.2f}"
    )
