from __future__ import annotations

import csv
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal


INTENT_LABELS: tuple[str, ...] = ("browsing", "researching", "ready_to_buy")
_TOKEN_RE = re.compile(r"[a-z0-9']+")
_TEMPLATES: dict[str, list[str]] = {
    "browsing": [
        "show me {ptype}s in {city}",
        "just looking at {ptype}s with {amenity}",
        "any nice homes around {city}",
        "what listings are out there in {city}",
        "browse options with {bed} in {city}",
        "curious about places near {target}",
        "I am casually checking homes with {amenity}",
        "what is available this week in {city}",
    ],
    "researching": [
        "what neighborhoods in {city} are best for families",
        "compare average prices for {ptype}s in {city}",
        "is {city} a good market for first-time buyers",
        "how much does a {bed} usually cost in {city}",
        "what should I know before buying in {city}",
        "can you explain HOA costs for {ptype}s",
        "research schools and commute near {target}",
        "which areas have lower taxes around {city}",
    ],
    "ready_to_buy": [
        "I need a {bed} in {city} under {budget} with {amenity}",
        "schedule showings for homes in {city} under {budget}",
        "ready to make an offer on a {ptype} in {city}",
        "find me a {bed} with {amenity} and close by Friday",
        "I am pre-approved and need homes below {budget} in {city}",
        "shortlist properties with {amenity} and {bed} in {city}",
        "I want to tour 3 homes this weekend in {city}",
        "send only move-in ready listings under {budget}",
    ],
}
_INTENT_KEYWORDS: dict[str, set[str]] = {
    "browsing": {
        "show",
        "looking",
        "browse",
        "casually",
        "curious",
        "available",
        "options",
        "listings",
    },
    "researching": {
        "compare",
        "research",
        "market",
        "neighborhoods",
        "how",
        "explain",
        "know",
        "taxes",
        "schools",
    },
    "ready_to_buy": {
        "need",
        "schedule",
        "showings",
        "offer",
        "pre",
        "approved",
        "tour",
        "shortlist",
        "move",
        "ready",
    },
}
_INTENT_PATTERNS: dict[str, tuple[str, ...]] = {
    "browsing": (
        r"\bjust looking\b",
        r"\bcasually\b",
        r"\bcurious\b",
        r"\bwhat is available\b",
        r"\bshow me\b",
        r"\bbrowse\b",
    ),
    "researching": (
        r"\bcompare\b",
        r"\bresearch\b",
        r"\bhow much\b",
        r"\bwhat should i know\b",
        r"\bmarket\b",
        r"\bneighborhood",
        r"\bschools?\b",
        r"\btaxes?\b",
    ),
    "ready_to_buy": (
        r"\bi need\b",
        r"\bpre-?approved\b",
        r"\bready to make an offer\b",
        r"\bschedule showings?\b",
        r"\bi want to tour\b",
        r"\bshortlist\b",
        r"\bthis weekend\b",
        r"\bmove-?in ready\b",
    ),
}


def build_default_labeled_dataset(seed: int = 42) -> list[tuple[str, str]]:
    """
    Build a deterministic, balanced dataset with 200+ labeled queries.
    """
    rng = random.Random(seed)
    cities = ["Irvine", "San Diego", "Los Angeles", "Austin", "Seattle", "Phoenix"]
    property_types = ["condo", "townhome", "single-family home", "duplex", "loft"]
    amenities = ["pool", "garage", "backyard", "office", "fireplace", "updated kitchen"]
    budgets = ["500k", "650k", "750k", "900k", "1m", "1.2m"]
    bed_options = ["2 bed", "3 bed", "4 bed", "3-4 bed"]
    commute_targets = ["downtown", "tech campuses", "good schools", "public transit"]

    dataset: list[tuple[str, str]] = []
    per_intent = 84  # 252 total
    for intent in INTENT_LABELS:
        intent_templates = _TEMPLATES[intent]
        for i in range(per_intent):
            template = intent_templates[i % len(intent_templates)]
            query = template.format(
                city=rng.choice(cities),
                ptype=rng.choice(property_types),
                amenity=rng.choice(amenities),
                budget=rng.choice(budgets),
                bed=rng.choice(bed_options),
                target=rng.choice(commute_targets),
            )
            dataset.append((query, intent))

    rng.shuffle(dataset)
    return dataset


def build_grouped_labeled_dataset(seed: int = 42) -> list[tuple[str, str, str]]:
    """
    Build dataset with template-group metadata for stronger holdout evaluation.
    Returns (query, intent, group_id), where group_id is template-based.
    """
    rng = random.Random(seed)
    cities = ["Irvine", "San Diego", "Los Angeles", "Austin", "Seattle", "Phoenix"]
    property_types = ["condo", "townhome", "single-family home", "duplex", "loft"]
    amenities = ["pool", "garage", "backyard", "office", "fireplace", "updated kitchen"]
    budgets = ["500k", "650k", "750k", "900k", "1m", "1.2m"]
    bed_options = ["2 bed", "3 bed", "4 bed", "3-4 bed"]
    commute_targets = ["downtown", "tech campuses", "good schools", "public transit"]

    rows: list[tuple[str, str, str]] = []
    per_intent = 84
    for intent in INTENT_LABELS:
        intent_templates = _TEMPLATES[intent]
        for i in range(per_intent):
            template_idx = i % len(intent_templates)
            template = intent_templates[template_idx]
            query = template.format(
                city=rng.choice(cities),
                ptype=rng.choice(property_types),
                amenity=rng.choice(amenities),
                budget=rng.choice(budgets),
                bed=rng.choice(bed_options),
                target=rng.choice(commute_targets),
            )
            rows.append((query, intent, f"{intent}:template_{template_idx}"))
    rng.shuffle(rows)
    return rows


def save_dataset_csv(dataset: Iterable[tuple[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["query", "intent"])
        writer.writerows(dataset)


@dataclass(frozen=True)
class IntentPrediction:
    intent: str
    confidence: float
    probabilities: dict[str, float]
    is_uncertain: bool


class IntentClassifier:
    def __init__(self, uncertainty_threshold: float = 0.6, random_state: int = 42):
        self.labels = list(INTENT_LABELS)
        self.uncertainty_threshold = uncertainty_threshold
        self.random_state = random_state
        self._is_trained = False
        self._class_doc_counts: dict[str, int] = {label: 0 for label in self.labels}
        self._term_counts: dict[str, dict[str, float]] = {label: {} for label in self.labels}
        self._class_total_terms: dict[str, float] = {label: 0.0 for label in self.labels}
        self._vocab: set[str] = set()
        self._idf: dict[str, float] = {}
        self._class_priors: dict[str, float] = {}

    def _tokenize(self, text: str) -> list[str]:
        tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
        if len(tokens) < 2:
            return tokens
        # Add simple bigram features for intent phrases.
        bigrams = [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]
        return tokens + bigrams

    def train(self, queries: list[str], labels: list[str]) -> None:
        if len(queries) != len(labels):
            raise ValueError("queries and labels must have same length")
        if not queries:
            raise ValueError("training data cannot be empty")

        self._class_doc_counts = {label: 0 for label in self.labels}
        self._term_counts = {label: {} for label in self.labels}
        self._class_total_terms = {label: 0.0 for label in self.labels}
        self._vocab = set()
        self._idf = {}
        self._class_priors = {}

        doc_freq: dict[str, int] = {}
        n_docs = len(queries)
        for query, label in zip(queries, labels):
            if label not in self._class_doc_counts:
                raise ValueError(f"Unknown label: {label}")
            self._class_doc_counts[label] += 1
            tokens = self._tokenize(query)
            seen: set[str] = set()
            for tok in tokens:
                self._vocab.add(tok)
                self._term_counts[label][tok] = self._term_counts[label].get(tok, 0.0) + 1.0
                self._class_total_terms[label] += 1.0
                if tok not in seen:
                    doc_freq[tok] = doc_freq.get(tok, 0) + 1
                    seen.add(tok)

        self._idf = {tok: math.log((1 + n_docs) / (1 + df)) + 1.0 for tok, df in doc_freq.items()}
        self._class_priors = {
            label: self._class_doc_counts[label] / n_docs for label in self.labels if self._class_doc_counts[label] > 0
        }
        self._is_trained = True

    def predict(self, query: str) -> IntentPrediction:
        if not self._is_trained:
            raise RuntimeError("IntentClassifier must be trained before calling predict().")

        tokens = self._tokenize(query)
        lower_query = query.lower()
        cue_hits = {
            label: sum(1 for pattern in _INTENT_PATTERNS[label] if re.search(pattern, lower_query))
            for label in self.labels
        }
        max_cue = max(cue_hits.values()) if cue_hits else 0
        winners = [label for label, hits in cue_hits.items() if hits == max_cue and hits > 0]
        if len(winners) == 1:
            intent = winners[0]
            confidence = min(0.98, 0.86 + 0.06 * max_cue)
            remaining = (1.0 - confidence) / (len(self.labels) - 1)
            probabilities = {
                label: (confidence if label == intent else remaining)
                for label in self.labels
            }
            return IntentPrediction(
                intent=intent,
                confidence=float(confidence),
                probabilities={label: float(probabilities.get(label, 0.0)) for label in self.labels},
                is_uncertain=confidence < self.uncertainty_threshold,
            )

        token_counts: dict[str, int] = {}
        for tok in tokens:
            token_counts[tok] = token_counts.get(tok, 0) + 1

        vocab_size = max(len(self._vocab), 1)
        log_scores: dict[str, float] = {}
        for label in self.labels:
            if label not in self._class_priors:
                continue
            score = math.log(self._class_priors[label])
            denom = self._class_total_terms[label] + vocab_size
            label_terms = self._term_counts[label]
            for tok, count in token_counts.items():
                tfidf_weight = count * self._idf.get(tok, 1.0)
                prob = (label_terms.get(tok, 0.0) + 1.0) / denom
                score += tfidf_weight * math.log(prob)

            # Small keyword prior improves generalization on unseen phrasings.
            keyword_hits = sum(1 for tok in _TOKEN_RE.findall(lower_query) if tok in _INTENT_KEYWORDS[label])
            score += 0.35 * keyword_hits

            # Pattern-level intent cues capture action phrases better than bag-of-words alone.
            pattern_hits = cue_hits[label]
            score += 1.4 * pattern_hits
            log_scores[label] = score

        max_log = max(log_scores.values())
        exp_scores = {label: math.exp(score - max_log) for label, score in log_scores.items()}
        total = sum(exp_scores.values()) or 1.0
        probabilities = {label: val / total for label, val in exp_scores.items()}
        intent, confidence = max(probabilities.items(), key=lambda kv: kv[1])
        return IntentPrediction(
            intent=intent,
            confidence=float(confidence),
            probabilities={label: float(probabilities.get(label, 0.0)) for label in self.labels},
            is_uncertain=confidence < self.uncertainty_threshold,
        )

    def evaluate(self, queries: list[str], labels: list[str]) -> dict[str, float]:
        if not self._is_trained:
            raise RuntimeError("IntentClassifier must be trained before calling evaluate().")
        preds = [self.predict(query).intent for query in queries]
        correct = sum(1 for pred, gold in zip(preds, labels) if pred == gold)
        accuracy = correct / len(labels) if labels else 0.0
        return {"accuracy": float(accuracy)}

    def train_test_evaluate(
        self,
        dataset: list[tuple[str, str]] | None = None,
        test_size: float = 0.2,
        split_strategy: Literal["stratified", "template_holdout"] = "stratified",
    ) -> dict[str, float | str]:
        samples = dataset or build_default_labeled_dataset()
        if split_strategy == "stratified":
            train, test = self._stratified_split(samples, test_size=test_size, seed=self.random_state)
        elif split_strategy == "template_holdout":
            grouped_samples = build_grouped_labeled_dataset(seed=self.random_state)
            train, test = self._template_holdout_split(grouped_samples, test_templates_per_label=2, seed=self.random_state)
        else:
            raise ValueError("split_strategy must be 'stratified' or 'template_holdout'")
        X_train = [q for q, _ in train]
        y_train = [y for _, y in train]
        X_test = [q for q, _ in test]
        y_test = [y for _, y in test]

        self.train(X_train, y_train)
        metrics: dict[str, float | str] = dict(self.evaluate(X_test, y_test))
        metrics["train_size"] = float(len(train))
        metrics["test_size"] = float(len(test))
        metrics["split_strategy"] = split_strategy
        return metrics

    def _stratified_split(
        self,
        samples: list[tuple[str, str]],
        test_size: float,
        seed: int,
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        by_label: dict[str, list[tuple[str, str]]] = {label: [] for label in self.labels}
        for query, label in samples:
            by_label[label].append((query, label))

        rng = random.Random(seed)
        train: list[tuple[str, str]] = []
        test: list[tuple[str, str]] = []
        for label, rows in by_label.items():
            if not rows:
                continue
            rng.shuffle(rows)
            cutoff = max(1, int(len(rows) * test_size))
            test.extend(rows[:cutoff])
            train.extend(rows[cutoff:])
        rng.shuffle(train)
        rng.shuffle(test)
        return train, test

    def _template_holdout_split(
        self,
        grouped_samples: list[tuple[str, str, str]],
        test_templates_per_label: int,
        seed: int,
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        rng = random.Random(seed)
        grouped_by_label: dict[str, dict[str, list[tuple[str, str]]]] = {label: {} for label in self.labels}

        for query, label, group in grouped_samples:
            grouped_by_label[label].setdefault(group, []).append((query, label))

        train: list[tuple[str, str]] = []
        test: list[tuple[str, str]] = []
        for label, groups in grouped_by_label.items():
            group_ids = list(groups.keys())
            rng.shuffle(group_ids)
            test_group_ids = set(group_ids[:test_templates_per_label])
            for group_id, rows in groups.items():
                if group_id in test_group_ids:
                    test.extend(rows)
                else:
                    train.extend(rows)

        rng.shuffle(train)
        rng.shuffle(test)
        return train, test