import re
import unicodedata
from collections import Counter

import pandas as pd


class TextCleaner:
    def __init__(self):
        self.abbrev_map = {
            # Bedrooms / bathrooms
            "br": "bedroom",
            "bdrm": "bedroom",
            "bd": "bedroom",
            "ba": "bathroom",
            "bth": "bathroom",
            "mbr": "master bedroom",
            "mbdrm": "master bedroom",
            "lr": "living room",
            "dr": "dining room",
            "fr": "family room",
            # Measurements / area
            "sqft": "square feet",
            "sf": "square feet",
            "sq ft": "square feet",
            "s.f.": "square feet",
            # Amenities / features
            "w/": "with",
            "w/o": "without",
            "w/d": "washer dryer",
            "hw": "hardwood",
            "hwd": "hardwood",
            "fp": "fireplace",
            "frplc": "fireplace",
            "ss": "stainless steel",
            "ss appl": "stainless steel appliances",
            "ss apps": "stainless steel appliances",
            "gran": "granite",
            "grnt": "granite",
            "kit": "kitchen",
            "eik": "eat in kitchen",
            "rec rm": "recreation room",
            "bsmt": "basement",
            "gar": "garage",
            "det": "detached",
            "att": "attached",
            # Misc listing shorthand
            "hoa": "homeowners association",
            "ac": "air conditioning",
            "a/c": "air conditioning",
            "hvac": "heating ventilation air conditioning",
        }

    def clean_text(self, text):
        text = self.normalize_unicode(text)
        text = self.normalize_punctuation(text)
        text = self.normalize_prices(text)
        text = self.normalize_measurements(text)
        text = self.normalize_symbols(text)
        text = self.expand_abbreviations(text)
        return text.strip() if isinstance(text, str) else text

    def normalize_prices(self, text):
        if not isinstance(text, str):
            return text
        # 450k → 450000
        text = re.sub(
            r"(\d+)k",
            lambda m: str(int(m.group(1)) * 1000),
            text,
            flags=re.I,
        )
        # 1.2m → 1200000
        text = re.sub(
            r"(\d+\.?\d*)m",
            lambda m: str(int(float(m.group(1)) * 1000000)),
            text,
            flags=re.I,
        )
        return text

    def normalize_unicode(self, text):
        """
        Normalize unicode punctuation and compatibility characters.
        """
        if not isinstance(text, str):
            return text

        # First apply Unicode compatibility normalization
        text = unicodedata.normalize("NFKC", text)

        # Normalize common punctuation variants via regex
        replacements = {
            r"[\u2018\u2019\u201B]": "'",  # single quotes / apostrophes
            r"[\u201C\u201D\u201F]": '"',  # double quotes
            r"[\u2010\u2011\u2012\u2013\u2014\u2212]": "-",  # dashes / minus
            r"\u00A0": " ",  # non‑breaking space → normal space
        }
        for pattern, repl in replacements.items():
            text = re.sub(pattern, repl, text)

        # Collapse multiple whitespace characters
        text = re.sub(r"\s+", " ", text)
        return text

    def normalize_punctuation(self, text):
        """
        Normalize basic punctuation spacing and repetitions using regex.
        """
        if not isinstance(text, str):
            return text

        # Remove spaces before punctuation like , . ! ? :
        text = re.sub(r"\s+([,\.!?;:])", r"\1", text)

        # Ensure a single space after punctuation when followed by a word character
        text = re.sub(r"([,\.!?;:])(?=\w)", r"\1 ", text)

        # Collapse repeated punctuation: "!!!" -> "!", ".." -> "."
        text = re.sub(r"([!?\.])\1+", r"\1", text)

        return text

    def normalize_measurements(self, text):
        """
        Normalize common real‑estate style measurements using regex.
        """
        if not isinstance(text, str):
            return text

        # Normalize square‑footage expressions to "<number> sqft"
        # e.g. "1,200 sq ft", "1200 sf", "1200sqft" → "1200 sqft"
        def _sqft_repl(m):
            num = m.group(1)
            num = num.replace(",", "")
            return f"{num} sqft"

        text = re.sub(
            r"(\d[\d,]*)\s*(?:sq\.?\s*ft|sq\s*ft|square\s*feet|sf|sqft)\b",
            _sqft_repl,
            text,
            flags=re.I,
        )

        # Normalize "x by y" room dimensions: "10 x 12 ft" → "10x12 ft"
        text = re.sub(
            r"(\d+)\s*[xX×]\s*(\d+)",
            r"\1x\2",
            text,
        )

        return text

    def normalize_symbols(self, text):
        """
        Normalize a few common symbolic tokens using regex.
        """
        if not isinstance(text, str):
            return text

        symbol_replacements = {
            r"\s*&\s*": " and ",
            r"\s*@\s*": " at ",
            r"\s*\+\s*": " plus ",
        }
        for pattern, repl in symbol_replacements.items():
            text = re.sub(pattern, repl, text)

        # Collapse any extra whitespace introduced
        text = re.sub(r"\s+", " ", text)
        return text

    def expand_abbreviations(self, text):
        """
        Expand known abbreviations using a single regex over the abbrev_map.
        """
        if not isinstance(text, str):
            return text

        # Build a regex that matches any abbreviation, respecting boundaries.
        # Use negative look‑behind / look‑ahead for word chars so entries like
        # "w/" still work correctly.
        keys = sorted(self.abbrev_map.keys(), key=len, reverse=True)
        pattern = r"(?<!\w)(" + "|".join(re.escape(k) for k in keys) + r")(?!\w)"

        def _repl(m):
            key = m.group(1)
            value = self.abbrev_map.get(key.lower())
            return value if value is not None else key

        return re.sub(pattern, _repl, text, flags=re.I)

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())

    def _extract_top_ngrams(self, series, n: int = 20, ngram: int = 2) -> list[tuple[str, int]]:
        counts: Counter[tuple[str, ...]] = Counter()
        for raw in series.fillna("").astype(str):
            cleaned = self.clean_text(raw)
            if not isinstance(cleaned, str) or not cleaned:
                continue
            toks = self._tokenize(cleaned)
            if len(toks) < ngram:
                continue
            for i in range(len(toks) - ngram + 1):
                counts[tuple(toks[i : i + ngram])] += 1
        return [(" ".join(k), v) for k, v in counts.most_common(n)]

    def _detect_abbreviations(self, series, n: int = 20) -> list[tuple[str, int]]:
        keys = set(k.lower() for k in self.abbrev_map.keys())
        counts: Counter[str] = Counter()
        for raw in series.fillna("").astype(str):
            text = self.normalize_unicode(raw)
            if not isinstance(text, str) or not text:
                continue
            lower = text.lower()
            for k in keys:
                if k in lower:
                    counts[k] += 1
        return counts.most_common(n)
    def profile_column(self, df, column_name):
        """Analyze what's actually in a free-text column (e.g. remarks)."""
        col = df[column_name]
        col_str = col.fillna("").astype(str)
        return {
            "null_rate": col.isnull().mean(),
            "avg_length": col_str.str.len().mean(),
            "common_terms": self._extract_top_ngrams(col_str),
            "price_mentions": col_str.str.contains(r"\$\d", na=False).sum(),
            "has_html": col_str.str.contains("<", na=False).sum(),
            "common_abbreviations": self._detect_abbreviations(col_str),
        }


def test_price_normalization():
    cleaner = TextCleaner()
    assert "450000" in cleaner.normalize_prices("priced at 450k")
    assert "1200000" in cleaner.normalize_prices("$1.2m home")
    print("test_price_normalization passed")


def test_profiling():
    cleaner = TextCleaner()
    df = pd.read_csv("data/processed/listing_sample.csv")
    profile = cleaner.profile_column(df, "remarks")
    assert "null_rate" in profile
    assert "avg_length" in profile
    print("test_profiling passed")


if __name__ == "__main__":
    test_price_normalization()
    test_profiling()

    cleaner = TextCleaner()
    df = pd.read_csv("data/processed/listing_sample.csv")
    profile = cleaner.profile_column(df, "remarks")
    print(f"HTML tags found in {profile['has_html']} listings")
    print(f"Common abbreviations: {profile['common_abbreviations']}")