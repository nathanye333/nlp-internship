import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Allow running as script: ensure project root is on path so "scripts" can be imported
if __name__ == "__main__":
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from scripts.entity_extractor import EntityExtractor


DATA_DIR = Path("data/processed")
DEFAULT_LABELS_PATH = DATA_DIR / "remarks_labeled.jsonl"


EntityDict = Dict[str, object]


def load_labeled_data(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def entities_by_type(record: Dict) -> Dict[str, List[Dict]]:
    by_type: Dict[str, List[Dict]] = defaultdict(list)
    for ent in record.get("entities", []):
        label = ent.get("label")
        if not label:
            continue
        by_type[label.upper()].append(ent)
    return by_type


def prediction_from_extractor(text: str, extractor: EntityExtractor) -> Dict[str, List[Dict]]:
    """
    Convert EntityExtractor outputs into the same abstract format as labels
    for evaluation purposes, using only the normalized value (not spans).
    """
    result = extractor.extract_all(text)
    preds: Dict[str, List[Dict]] = defaultdict(list)

    if result.get("bedrooms") is not None:
        preds["BEDROOMS"].append({"value": result["bedrooms"]})

    if result.get("bathrooms") is not None:
        preds["BATHROOMS"].append({"value": result["bathrooms"]})

    if result.get("price") is not None:
        preds["PRICE"].append({"value": result["price"]})

    sqft_vals = result.get("sqft")
    if isinstance(sqft_vals, list):
        for v in sqft_vals:
            if v is not None:
                preds["SQFT"].append({"value": v})
    elif sqft_vals is not None:
        preds["SQFT"].append({"value": sqft_vals})

    amenities = result.get("amenities")
    if isinstance(amenities, list):
        for amenity in amenities:
            preds["AMENITY"].append({"value": amenity})

    return preds


def normalize_value(v):
    """Normalize entity values for comparison. Especially for AMENITY: map gold span
    variants (e.g. 'a spacious', 'the spacious', 'pool and') to canonical terms."""
    if isinstance(v, str):
        v = v.strip().lower()
        # Strip leading article so "a spacious" / "the spacious" -> "spacious"
        for prefix in ("a ", "the "):
            if v.startswith(prefix):
                v = v[len(prefix) :].strip()
        # Map phrase variants to canonical amenity terms (must match extractor output)
        amenity_aliases = {
            "pool and": "pool",
            "pool and spa": "pool",
            "space for": "space",
        }
        v = amenity_aliases.get(v, v)
        # "pool and" with trailing space or similar
        if v.endswith(" and"):
            v = v[:-4].strip()
        return v
    return v


def compute_prf(
    gold: List[Dict], preds: List[Dict]
) -> Tuple[int, int, int, float, float, float]:
    """
    Compute TP, FP, FN and P/R/F1 based on value equality only.
    """
    gold_vals = [normalize_value(e.get("value")) for e in gold]
    pred_vals = [normalize_value(e.get("value")) for e in preds]

    gold_counter = Counter(gold_vals)
    pred_counter = Counter(pred_vals)

    tp = sum((gold_counter & pred_counter).values())
    fp = sum((pred_counter - gold_counter).values())
    fn = sum((gold_counter - pred_counter).values())

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return tp, fp, fn, precision, recall, f1


def evaluate(path: Path) -> None:
    extractor = EntityExtractor()
    records = load_labeled_data(path)

    labels = ["BEDROOMS", "BATHROOMS", "PRICE", "SQFT", "AMENITY"]
    agg_gold: Dict[str, List[Dict]] = {l: [] for l in labels}
    agg_pred: Dict[str, List[Dict]] = {l: [] for l in labels}

    # For error analysis
    errors: Dict[str, Dict[str, List[Dict]]] = {
        l: {"fp": [], "fn": [], "tp": []} for l in labels
    }

    for rec in records:
        text = rec.get("text", "")
        gold_by_type = entities_by_type(rec)
        pred_by_type = prediction_from_extractor(text, extractor)

        for label in labels:
            g = gold_by_type.get(label, [])
            p = pred_by_type.get(label, [])

            agg_gold[label].extend(g)
            agg_pred[label].extend(p)

            # Per-record error analysis
            gold_vals = Counter(normalize_value(e.get("value")) for e in g)
            pred_vals = Counter(normalize_value(e.get("value")) for e in p)

            tp_vals = gold_vals & pred_vals
            fp_vals = pred_vals - gold_vals
            fn_vals = gold_vals - pred_vals

            if any(tp_vals.values()) or any(fp_vals.values()) or any(fn_vals.values()):
                errors[label]["tp"].append(
                    {
                        "text": text,
                        "gold": g,
                        "pred": p,
                        "tp_vals": dict(tp_vals),
                    }
                )
                if fp_vals:
                    errors[label]["fp"].append(
                        {
                            "text": text,
                            "gold": g,
                            "pred": p,
                            "fp_vals": dict(fp_vals),
                        }
                    )
                if fn_vals:
                    errors[label]["fn"].append(
                        {
                            "text": text,
                            "gold": g,
                            "pred": p,
                            "fn_vals": dict(fn_vals),
                        }
                    )

    print(f"Evaluating on {len(records)} labeled remarks from {path}")
    print()

    macro_p = macro_r = macro_f1 = 0.0
    for label in labels:
        tp, fp, fn, p, r, f1 = compute_prf(agg_gold[label], agg_pred[label])
        macro_p += p
        macro_r += r
        macro_f1 += f1
        print(
            f"{label:10s}  TP={tp:4d}  FP={fp:4d}  FN={fn:4d}  "
            f"P={p:5.3f}  R={r:5.3f}  F1={f1:5.3f}"
        )

    n = len(labels)
    print()
    print(
        f"Macro-average: P={macro_p/n:5.3f}  R={macro_r/n:5.3f}  F1={macro_f1/n:5.3f}"
    )

    # Basic error analysis: show a few representative FNs/FPs per label
    print("\nError analysis (showing up to 5 examples per label/type):")
    for label in labels:
        for kind in ("fn", "fp"):
            bucket = errors[label][kind][:5]
            if not bucket:
                continue
            print(f"\n[{label}] {kind.upper()} examples:")
            for ex in bucket:
                print("-" * 60)
                print("Text:", ex["text"])
                print("Gold:", ex["gold"])
                print("Pred:", ex["pred"])
                print(f"{kind.upper()} values:", ex[f"{kind}_vals"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate EntityExtractor against labeled JSONL dataset."
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=DEFAULT_LABELS_PATH,
        help="Path to labeled remarks JSONL (default: data/processed/remarks_labeled.jsonl)",
    )
    args = parser.parse_args()

    evaluate(args.labels_path)


if __name__ == "__main__":
    main()

