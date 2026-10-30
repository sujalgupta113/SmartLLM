
from sentence_transformers import SentenceTransformer
import numpy as np
from cache import SemanticCache


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# (question A, question B, safe_to_reuse_answer)
#
# True: reusing the same answer is appropriate.
# False: a fresh answer is required.
#
# These are illustrative labels for our small test set.

test_pairs = [
    # Exact repeats and paraphrases
    ("What is Python?", "What is Python?", True),
    ("What is Python?", "Explain Python to me", True),
    (
        "How can I learn Python?",
        "What is the best way to learn Python?",
        True,
    ),
    (
        "What is machine learning?",
        "Explain machine learning",
        True,
    ),
    (
        "What is a semantic cache?",
        "Explain semantic caching",
        True,
    ),
    (
        "What is photosynthesis?",
        "Explain how photosynthesis works",
        True,
    ),

    # Clearly unrelated questions
    ("What is Python?", "How do I bake a chocolate cake?", False),
    (
        "What is machine learning?",
        "What is the capital of Japan?",
        False,
    ),
    (
        "Explain photosynthesis",
        "How do I repair a bicycle tire?",
        False,
    ),

    # Similar wording, different intentions
    (
        "How do I reset my password?",
        "How do I change my password?",
        False,
    ),
    (
        "How do I delete my account?",
        "How do I deactivate my account?",
        False,
    ),
    (
        "How can I increase my bank balance?",
        "How can I check my bank balance?",
        False,
    ),
    (
        "How do I transfer money?",
        "How do I check my bank balance?",
        False,
    ),
    (
        "How do I delete my account?",
        "How do I download my account data?",
        False,
    ),
    (
        "What dosage should I take?",
        "What is the usual prescription dosage?",
        False,
    ),
    (
        "How do I change my password?",
        "How do I update my email address?",
        False,
    ),
    (
        "How do I deactivate my account?",
        "How do I close my account?",
        False,
    ),
    (
        "How can I buy shares?",
        "How can I sell shares?",
        False,
    ),

    # Other cases
    (
        "How do I install Python?",
        "How do I install Python on Windows?",
        True,
    ),
    (
        "What is a database?",
        "Explain what a database is",
        True,
    ),
]


def safe_to_cache(question):
    return SemanticCache.is_cacheable(question)


# Encode all questions once for the experiment.
all_questions = []
for q1, q2, _ in test_pairs:
    all_questions.extend([q1, q2])

embeddings = model.encode(
    all_questions,
    normalize_embeddings=True
)


def calculate_metrics(predictions, labels):
    tp = fp = fn = tn = 0

    for predicted, actual in zip(predictions, labels):
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


baseline_predictions = []
safe_predictions = []
labels = []
rows = []

threshold = 0.75

for i, (q1, q2, expected_safe) in enumerate(test_pairs):
    score = float(
        np.dot(embeddings[2 * i], embeddings[2 * i + 1])
    )

    baseline_match = score >= threshold

    # Require both questions to be eligible for caching.
    # A risky question must bypass semantic reuse.
    safety_match = (
        baseline_match
        and safe_to_cache(q1)
        and safe_to_cache(q2)
    )

    baseline_predictions.append(baseline_match)
    safe_predictions.append(safety_match)
    labels.append(expected_safe)

    rows.append({
        "Question A": q1,
        "Question B": q2,
        "Similarity": round(score, 3),
        "Expected safe": expected_safe,
        "Baseline match": baseline_match,
        "Safety-aware match": safety_match,
    })


baseline = calculate_metrics(baseline_predictions, labels)
improved = calculate_metrics(safe_predictions, labels)

print("\nPAIR-BY-PAIR RESULTS")
print("=" * 90)

for row in rows:
    print(
        f"\nA: {row['Question A']}\n"
        f"B: {row['Question B']}\n"
        f"Similarity: {row['Similarity']:.3f}\n"
        f"Expected safe: {row['Expected safe']}\n"
        f"Baseline match: {row['Baseline match']}\n"
        f"Safety-aware match: {row['Safety-aware match']}"
    )


print("\nBEFORE vs AFTER")
print("=" * 90)

for name, metrics in [
    ("Baseline", baseline),
    ("Safety-aware", improved),
]:
    print(f"\n{name}")
    for key, value in metrics.items():
        if key in ("Precision", "Recall", "F1"):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")

print("\nSAFETY RULES")
print("=" * 90)

for question in [
    "How do I reset my password?",
    "How do I transfer money?",
    "What is machine learning?",
]:
    print(
        f"{question} -> "
        f"{SemanticCache.safety_reason(question) or 'cacheable'}"
    )
