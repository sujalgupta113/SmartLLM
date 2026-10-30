
from sentence_transformers import SentenceTransformer
import numpy as np
import re

from database import (
    initialize_db,
    save_cache_entry,
    load_cache_entries,
    clear_cache_database,
    get_cache_count,
    DB_PATH
)


class SemanticCache:

    RISKY_PATTERNS = {
        "password_or_security": [
            r"\bpasswords?\b",
            r"\bpasscodes?\b",
            r"\botp\b",
            r"\btwo.factor authentication\b",
            r"\b2fa\b",
            r"\bsecurity codes?\b",
        ],

        "account_changes": [
            r"\bdelete\b.{0,30}\baccount\b",
            r"\bdeactivate\b.{0,30}\baccount\b",
            r"\bclose\b.{0,20}\baccount\b",
            r"\breset\b.{0,30}\b(account|password|credentials?)\b",
            r"\bchange\b.{0,30}\b(password|credentials?)\b",
            r"\bunlock\b.{0,20}\baccount\b",
        ],

        "financial_actions": [
            r"\btransfer\b",
            r"\bwithdraw\b",
            r"\bdeposit\b",
            r"\bsend money\b",
            r"\bpay\b",
            r"\bpurchase\b",
            r"\bbuy\b",
            r"\bsell\b",
            r"\binvest\b",
            r"\btrade\b",
            r"\bloan\b",
            r"\bcredit card\b",
            r"\bbank balance\b",
            r"\baccount balance\b",
        ],

        "other_sensitive_actions": [
            r"\bmedical advice\b",
            r"\bdosage\b",
            r"\bprescription\b",
            r"\blegal advice\b",
            r"\bdelete my data\b",
            r"\bpersonal information\b",
        ],
    }

    def __init__(self, threshold=0.80):

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.threshold = threshold

        self.questions = []
        self.answers = []
        self.embeddings = []

        # Initialize SQLite database
        initialize_db()

        # Load saved entries from SQLite
        self.load_from_database()

    # -------------------- NORMALIZE QUESTION --------------------

    @staticmethod
    def normalize(question):

        return " ".join(question.casefold().split())

    # -------------------- SAFETY CHECK --------------------

    @classmethod
    def safety_reason(cls, question):

        text = question.casefold()

        for category, patterns in cls.RISKY_PATTERNS.items():

            for pattern in patterns:

                if re.search(pattern, text):
                    return category

        return None

    @classmethod
    def is_cacheable(cls, question):

        return cls.safety_reason(question) is None

    # -------------------- CREATE EMBEDDING --------------------

    def _embed(self, text):

        return self.model.encode(
            text,
            normalize_embeddings=True
        )

    # -------------------- LOAD FROM SQLITE --------------------

    def load_from_database(self):

        entries = load_cache_entries()

        self.questions = []
        self.answers = []
        self.embeddings = []

        for entry in entries:

            question = entry["question"]
            answer = entry["answer"]

            embedding = np.array(
                entry["embedding"],
                dtype=np.float32
            )

            # Skip unsafe saved questions
            if not self.is_cacheable(question):
                continue

            self.questions.append(question)
            self.answers.append(answer)
            self.embeddings.append(embedding)

        print(
            f"SQLite: Loaded {len(self.questions)} cache entries."
        )

        print(f"Database path: {DB_PATH}")
        print(f"Total database entries: {get_cache_count()}")

    # -------------------- FIND CACHE --------------------

    def find(self, question):

        if not self.is_cacheable(question):
            return None, None

        if not self.questions:
            return None, None

        normalized = self.normalize(question)

        # Exact match
        for i, saved_question in enumerate(self.questions):

            if normalized == self.normalize(saved_question):

                return self.answers[i], 1.0

        # Semantic similarity
        query_embedding = self._embed(question)

        scores = [
            float(np.dot(query_embedding, embedding))
            for embedding in self.embeddings
        ]

        best_index = int(np.argmax(scores))
        best_score = scores[best_index]

        if not self.is_cacheable(
            self.questions[best_index]
        ):
            return None, best_score

        if best_score >= self.threshold:

            return self.answers[best_index], best_score

        return None, best_score

    # -------------------- SAVE CACHE --------------------

    def save(self, question, answer):

        if not self.is_cacheable(question):
            print("SQLite: Unsafe question. Not cached.")
            return False

        # Avoid duplicate exact questions
        normalized = self.normalize(question)

        for saved_question in self.questions:

            if normalized == self.normalize(saved_question):
                print("SQLite: Duplicate question. Not saved.")
                return False

        # Generate embedding
        embedding = self._embed(question)

        # Save to SQLite and receive confirmed row ID
        saved_id = save_cache_entry(
            question,
            answer,
            embedding
        )

        if not saved_id:
            raise RuntimeError(
                "SQLite did not confirm the saved entry."
            )

        # Verify saved entry exists
        database_count = get_cache_count()

        print(f"SQLite saved entry ID: {saved_id}")
        print(f"SQLite database path: {DB_PATH}")
        print(f"SQLite total entries: {database_count}")

        # Update in-memory cache only after successful database save
        self.questions.append(question)
        self.answers.append(answer)
        self.embeddings.append(
            np.array(embedding, dtype=np.float32)
        )

        return True

    # -------------------- CLEAR CACHE --------------------

    def clear(self):

        clear_cache_database()

        self.questions = []
        self.answers = []
        self.embeddings = []

        print("SQLite cache cleared successfully.")
        print(f"Database path: {DB_PATH}")
        print(f"Remaining entries: {get_cache_count()}")
