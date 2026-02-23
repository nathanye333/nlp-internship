import json
import os
from collections import Counter
from typing import Dict, List

import nltk
import pandas as pd
from nltk.util import ngrams


def load_remarks(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path)
    return df["remarks"].dropna().astype(str).str.lower().tolist()


def tokenize(texts: List[str]) -> List[str]:
    # Ensure tokenizer data is available
    nltk.download("punkt", quiet=True)
    tokens: List[str] = []
    for t in texts:
        tokens.extend(nltk.word_tokenize(t))
    return tokens


def build_ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter(ngrams(tokens, n))


def categorize_term(term: str) -> str:
    term_l = term.lower()
    if any(k in term_l for k in ["bed", "bath", "sq ft", "square foot", "acres", "lot"]):
        return "property_features"
    if any(k in term_l for k in ["pool", "spa", "gym", "fitness", "fireplace", "garage", "parking", "laundry"]):
        return "amenities"
    if any(k in term_l for k in ["downtown", "beach", "ocean", "lake", "waterfront", "park", "trail"]):
        return "location"
    if any(k in term_l for k in ["school", "district", "shopping", "dining", "restaurant"]):
        return "community"
    if any(k in term_l for k in ["luxury", "remodeled", "updated", "turnkey", "move-in"]):
        return "condition_and_style"
    if any(k in term_l for k in ["condo", "townhome", "single family", "apartment", "loft"]):
        return "property_type"
    if any(k in term_l for k in ["view", "sunset", "panoramic", "city lights", "mountain"]):
        return "views"
    return "other"


def make_taxonomy(remarks_csv: str, max_terms: int = 260) -> Dict:
    texts = load_remarks(remarks_csv)
    tokens = tokenize(texts)

    unigram_counts = Counter(tokens)
    bigram_counts = build_ngram_counts(tokens, 2)

    # Build candidate phrases: top unigrams and bigrams combined
    top_unigrams = [" ".join((w,)) for w, _ in unigram_counts.most_common(160)]
    top_bigrams = [" ".join(bg) for bg, _ in bigram_counts.most_common(200)]

    seen = set()
    phrases: List[str] = []
    for phrase in top_bigrams + top_unigrams:
        norm = phrase.strip()
        if len(norm) < 3:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        phrases.append(norm)
        if len(phrases) >= max_terms:
            break

    categories = [
        "property_features",
        "amenities",
        "location",
        "community",
        "condition_and_style",
        "property_type",
        "views",
        "other",
    ]

    terms = []
    for idx, phrase in enumerate(phrases, start=1):
        terms.append(
            {
                "id": idx,
                "term": phrase,
                "category": categorize_term(phrase),
            }
        )

    taxonomy = {
        "categories": categories,
        "terms": terms,
        "meta": {
            "source": "listing_sample.csv",
            "description": "Real estate listing remark taxonomy built from n-gram frequencies.",
        },
    }
    return taxonomy


def save_taxonomy(taxonomy: Dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(taxonomy, f, indent=2)


def main() -> None:
    input_csv = "data/processed/listing_sample.csv"
    output_json = "data/processed/taxonomy.json"
    taxonomy = make_taxonomy(input_csv)

    if len(taxonomy["terms"]) < 200:
        raise RuntimeError("Taxonomy must contain at least 200 terms.")

    save_taxonomy(taxonomy, output_json)
    print(f"Saved taxonomy with {len(taxonomy['terms'])} terms to {output_json}")


if __name__ == "__main__":
    main()