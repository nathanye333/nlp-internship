import sys
import csv
from pathlib import Path


# Ensure project root is on sys.path so `scripts` can be imported.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.intent_classifier import IntentClassifier, build_default_labeled_dataset, save_dataset_csv
from scripts.query_parser import QueryParser


def test_default_dataset_has_200_plus_labeled_queries():
    dataset = build_default_labeled_dataset()
    assert len(dataset) >= 200
    assert set(label for _, label in dataset) == {"browsing", "researching", "ready_to_buy"}


def test_classifier_reaches_target_accuracy():
    classifier = IntentClassifier()
    metrics = classifier.train_test_evaluate()
    assert float(metrics["accuracy"]) >= 0.80


def test_prediction_has_confidence_and_probability_breakdown():
    classifier = IntentClassifier()
    dataset = build_default_labeled_dataset()
    classifier.train([q for q, _ in dataset], [y for _, y in dataset])

    prediction = classifier.predict("I need a 3 bed in Irvine under 900k with garage")
    assert prediction.intent in {"browsing", "researching", "ready_to_buy"}
    assert 0.0 <= prediction.confidence <= 1.0
    assert set(prediction.probabilities.keys()) == {"browsing", "researching", "ready_to_buy"}
    assert abs(sum(prediction.probabilities.values()) - 1.0) < 1e-6
    assert isinstance(prediction.is_uncertain, bool)


def test_query_parser_integration_produces_richer_output():
    classifier = IntentClassifier()
    dataset = build_default_labeled_dataset()
    classifier.train([q for q, _ in dataset], [y for _, y in dataset])
    parser = QueryParser()

    enriched = parser.parse_with_intent(
        "3 bed under 700k in Irvine with pool, schedule showings this week",
        intent_classifier=classifier,
    )

    assert "filters" in enriched
    assert "where_sql" in enriched
    assert "params" in enriched
    assert "intent" in enriched
    assert "intent_confidence" in enriched
    assert "intent_uncertain" in enriched
    assert "intent_probabilities" in enriched
    assert enriched["intent"] in {"browsing", "researching", "ready_to_buy"}


def test_can_export_labeled_dataset_csv(tmp_path):
    dataset = build_default_labeled_dataset()
    output_path = tmp_path / "intent_dataset.csv"
    save_dataset_csv(dataset, output_path)
    assert output_path.exists()


def test_human_like_holdout_generalization():
    holdout_path = PROJECT_ROOT / "data" / "processed" / "intent_human_like_eval.csv"
    rows: list[tuple[str, str]] = []
    with holdout_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append((row["query"], row["intent"]))

    classifier = IntentClassifier()
    train_dataset = build_default_labeled_dataset()
    classifier.train([q for q, _ in train_dataset], [y for _, y in train_dataset])

    correct = 0
    for query, gold in rows:
        if classifier.predict(query).intent == gold:
            correct += 1

    accuracy = correct / len(rows)
    assert accuracy >= 0.80
