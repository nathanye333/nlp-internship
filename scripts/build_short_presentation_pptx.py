"""Generate a concise 2-3 minute executive deck as a .pptx file.

Output:
    docs/final_presentation_short.pptx

This version is intentionally short and presentation-friendly:
problem -> architecture -> key features -> one challenge solved -> outcome.
It is fully editable in Google Slides after import.
"""

from __future__ import annotations

from pathlib import Path
import sys

from pptx import Presentation
from pptx.enum.text import PP_ALIGN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_presentation_pptx import (
    ACCENT,
    ACCENT_AMBER,
    ACCENT_VIOLET,
    BG,
    FONT_MONO,
    GOOD,
    INK,
    LINE,
    MUTED,
    MX,
    PANEL,
    PANEL_STRONG,
    SLIDE_H,
    SLIDE_W,
    USABLE_W,
    WARN,
    add_code_block,
    add_metric_card,
    add_paragraphs,
    add_rect,
    add_table,
    add_text,
    add_topbar,
    add_title,
    slide_blank,
)
from pptx.util import Inches


def slide_01_title(prs: Presentation) -> None:
    s = slide_blank(prs)
    add_text(
        s,
        Inches(1.0), Inches(1.15), Inches(11.3), Inches(0.45),
        "REAL ESTATE NLP SYSTEM   ·   2–3 MINUTE TECHNICAL OVERVIEW",
        size=18, color=ACCENT, bold=True, align=PP_ALIGN.CENTER,
    )
    add_text(
        s,
        Inches(0.9), Inches(1.95), Inches(11.5), Inches(1.8),
        "How we turned free-text MLS\nremarks into a production-ready product",
        size=38, color=INK, bold=True, align=PP_ALIGN.CENTER, line_spacing=1.08,
    )
    add_text(
        s,
        Inches(1.5), Inches(4.15), Inches(10.3), Inches(1.0),
        "A single system for natural-language search, structured extraction, "
        "listing summarization, buyer-intent signals, and Fair Housing compliance.",
        size=20, color=MUTED, align=PP_ALIGN.CENTER, line_spacing=1.35,
    )

    stats = [
        ("40,890", "listings processed"),
        ("12", "API endpoints"),
        ("18", "weeks? no — 12-week build"),
        ("live", "Docker + Render deploy"),
    ]
    card_w = Inches(2.55)
    gap = Inches(0.22)
    total_w = card_w * 4 + gap * 3
    start_x = (SLIDE_W - total_w) / 2
    for i, (big, small) in enumerate(stats):
        x = start_x + (card_w + gap) * i
        y = Inches(5.7)
        add_metric_card(s, x, y, card_w, Inches(1.35), "", big, small, color=ACCENT)


def slide_02_problem_approach(prs: Presentation) -> None:
    s = slide_blank(prs)
    add_topbar(s, "01 · Problem & Approach", "Why we built it")
    add_title(s, "The core problem: MLS value is trapped in unstructured remarks.")

    col_w = (USABLE_W - Inches(0.4)) / 2
    lx, rx = MX, MX + col_w + Inches(0.4)
    y = Inches(1.95)
    h = Inches(4.85)

    add_rect(s, lx, y, col_w, h, fill=PANEL, line=LINE, radius=0.08)
    add_text(s, lx + Inches(0.3), y + Inches(0.25), col_w - Inches(0.6), Inches(0.45),
             "Why keyword search falls short", size=22, color=ACCENT, bold=True)
    add_paragraphs(
        s, lx + Inches(0.35), y + Inches(0.95), col_w - Inches(0.7), Inches(3.5),
        [
            "Users ask: “3 bed under 700k in Irvine with pool.” Traditional filters only understand part of that request.",
            "The most useful listing information lives in agent remarks: amenities, condition, financing language, neighborhood context.",
            "Manual compliance review does not scale and creates legal risk for Fair Housing violations.",
        ],
        size=18, line_spacing=1.35, space_after=10,
    )

    add_rect(s, rx, y, col_w, h, fill=PANEL_STRONG, line=LINE, radius=0.08)
    add_text(s, rx + Inches(0.3), y + Inches(0.25), col_w - Inches(0.6), Inches(0.45),
             "Our technical approach", size=22, color=ACCENT_AMBER, bold=True)
    add_paragraphs(
        s, rx + Inches(0.35), y + Inches(0.95), col_w - Inches(0.7), Inches(3.5),
        [
            "Normalize remarks and queries first, so abbreviations and price formats stop breaking downstream logic.",
            "Combine rule-based extraction with semantic retrieval, rather than forcing one model to do everything.",
            "Expose the entire pipeline through a FastAPI service so the same logic powers demos and production.",
        ],
        size=18, line_spacing=1.35, space_after=10,
    )

    add_text(
        s, MX, Inches(6.95), USABLE_W, Inches(0.35),
        "Design principle: use the simplest reliable method per task — regex for structured signals, embeddings for retrieval, rules for compliance-critical blocking.",
        size=15, color=MUTED, italic=True, line_spacing=1.3,
    )


def slide_03_architecture(prs: Presentation) -> None:
    s = slide_blank(prs)
    add_topbar(s, "02 · Overall Architecture", "Data → NLP → API → UI")
    add_title(s, "Architecture: one pipeline, multiple business capabilities.")

    # Four horizontal layers
    layer_x = MX
    layer_w = USABLE_W
    block_gap = Inches(0.18)

    def layer(y, label, items, color):
        add_text(s, layer_x, y, Inches(1.4), Inches(0.3), label, size=13, color=MUTED, bold=True)
        box_y = y + Inches(0.35)
        box_h = Inches(0.95)
        bw = (layer_w - block_gap * (len(items) - 1)) / len(items)
        for i, (head, sub) in enumerate(items):
            x = layer_x + (bw + block_gap) * i
            add_rect(s, x, box_y, bw, box_h, fill=PANEL if label != "SERVING" else PANEL_STRONG, line=color, radius=0.08)
            add_text(s, x + Inches(0.12), box_y + Inches(0.12), bw - Inches(0.24), Inches(0.28),
                     head, size=15, color=INK, bold=True, align=PP_ALIGN.CENTER)
            add_text(s, x + Inches(0.12), box_y + Inches(0.45), bw - Inches(0.24), Inches(0.32),
                     sub, size=12, color=MUTED, align=PP_ALIGN.CENTER, line_spacing=1.2)

    layer(Inches(1.8), "DATA", [
        ("MLS tables", "rets_property · openhouse · sold"),
        ("taxonomy.json", "200+ terms · 8 categories"),
        ("FAISS index", "semantic corpus for search"),
        ("signals JSONL", "40,890 listing enrichments"),
    ], ACCENT)

    layer(Inches(3.15), "NLP CORE", [
        ("Text cleaning", "normalize prices, sqft, abbreviations"),
        ("Extraction", "beds · baths · amenities · signals"),
        ("Search", "semantic + BM25 + hybrid ranker"),
        ("Safety", "answerability + compliance checks"),
    ], ACCENT_VIOLET)

    layer(Inches(4.5), "SERVING", [
        ("FastAPI", "12 endpoints behind one service"),
        ("Cross-cutting", "cache · rate limit · logs · health"),
        ("Streamlit UI", "search · compare · metrics demo"),
    ], ACCENT_AMBER)

    add_text(
        s, MX, Inches(6.45), USABLE_W, Inches(0.6),
        "The architecture is modular by design: every NLP component is a pure Python module, and the API layer only orchestrates them. That made iterative delivery possible across the 12-week project.",
        size=17, color=MUTED, line_spacing=1.35,
    )


def slide_04_key_features(prs: Presentation) -> None:
    s = slide_blank(prs)
    add_topbar(s, "03 · Key Features Implemented", "What the system actually does")
    add_title(s, "Three capabilities mattered most.")

    col_w = (USABLE_W - Inches(0.5)) / 3
    y = Inches(1.95)
    h = Inches(4.9)
    cards = [
        (
            "1. Hybrid Search",
            ACCENT,
            [
                "Parses natural-language queries into structured filters.",
                "Combines semantic embeddings with BM25 keyword retrieval.",
                "Returns stronger results for both exact tokens and paraphrases.",
            ],
        ),
        (
            "2. Listing Intelligence",
            ACCENT_VIOLET,
            [
                "Extracts beds, baths, price, sqft, amenities, condition, and location signals.",
                "Generates short listing summaries for result cards.",
                "Adds buyer-intent classification for lead routing.",
            ],
        ),
        (
            "3. Safety & Production",
            ACCENT_AMBER,
            [
                "Fair Housing compliance checker flags risky phrasing before publish.",
                "FastAPI service adds validation, caching, rate limiting, and health checks.",
                "Deployed with Docker and Render as API + UI services.",
            ],
        ),
    ]
    for i, (title, color, bullets) in enumerate(cards):
        x = MX + (col_w + Inches(0.25)) * i
        add_rect(s, x, y, col_w, h, fill=PANEL if i != 2 else PANEL_STRONG, line=LINE, radius=0.08)
        add_text(s, x + Inches(0.25), y + Inches(0.25), col_w - Inches(0.5), Inches(0.55),
                 title, size=22, color=color, bold=True)
        add_paragraphs(
            s, x + Inches(0.3), y + Inches(1.0), col_w - Inches(0.6), Inches(3.2),
            bullets, size=17, line_spacing=1.35, space_after=10,
        )

    add_text(
        s, MX, Inches(6.95), USABLE_W, Inches(0.35),
        "Together, these features turn raw listing text into search relevance, product UX improvements, and compliance risk reduction.",
        size=15, color=MUTED, italic=True, line_spacing=1.3,
    )


def slide_05_challenge(prs: Presentation) -> None:
    s = slide_blank(prs)
    add_topbar(s, "04 · One Technical Challenge Solved", "Hybrid retrieval under structured filters")
    add_title(s, "Challenge: semantic search and strict filters do not naturally fit together.")

    col_w = (USABLE_W - Inches(0.4)) / 2
    lx, rx = MX, MX + col_w + Inches(0.4)

    add_text(s, lx, Inches(1.95), col_w, Inches(0.45),
             "What broke", size=20, color=ACCENT, bold=True)
    add_paragraphs(
        s, lx, Inches(2.45), col_w, Inches(2.0),
        [
            "FAISS can rank by semantic similarity, but it does not know about structured constraints like city, price, beds, and baths.",
            "That means a query like “3 bed under 700k in Irvine with pool” can retrieve semantically similar homes in the wrong city.",
            "If filters are applied only after retrieval, top-k can collapse to zero useful results.",
        ],
        size=18, line_spacing=1.35, space_after=10,
    )

    add_text(s, lx, Inches(4.95), col_w, Inches(0.45),
             "Our solution", size=20, color=ACCENT_AMBER, bold=True)
    add_paragraphs(
        s, lx, Inches(5.45), col_w, Inches(1.4),
        [
            "When structured filters are present, the API intentionally over-retrieves a larger candidate pool.",
            "Then it applies metadata filters before truncating back to top-k.",
            "This preserves recall while keeping latency low enough for real-time search.",
        ],
        size=17, line_spacing=1.35, space_after=10,
    )

    code = (
        "candidate_top_k = retrieval_depth(top_k, filters, metadata_store)\n"
        "raw_hits = run_search(query, mode=mode, top_k=candidate_top_k)\n"
        "metadata_filtered = apply_metadata_filters(\n"
        "    raw_hits, filters, metadata_store)\n"
        "final_hits = apply_filters(metadata_filtered, filters)[:top_k]"
    )
    add_code_block(s, rx, Inches(2.0), col_w, Inches(2.8), code, size=14)
    add_metric_card(
        s, rx, Inches(5.2), col_w, Inches(1.55),
        "RESULT", "< 30 ms", "filtered hybrid search stayed interactive while avoiding empty top-k failures",
        color=GOOD,
    )


def slide_06_close(prs: Presentation) -> None:
    s = slide_blank(prs)
    add_topbar(s, "05 · Takeaway", "Why the approach matters")
    add_title(s, "The takeaway.")

    add_rect(s, Inches(1.0), Inches(1.9), Inches(11.3), Inches(3.0), fill=PANEL_STRONG, line=LINE, radius=0.1)
    add_text(
        s, Inches(1.45), Inches(2.35), Inches(10.4), Inches(1.8),
        "We built a modular NLP system that converts unstructured MLS remarks into\n"
        "search relevance, structured product features, and compliance protection —\n"
        "and we shipped it as a real deployable service, not just a notebook demo.",
        size=26, color=INK, bold=True, align=PP_ALIGN.CENTER, line_spacing=1.2,
    )

    card_w = Inches(3.3)
    gap = Inches(0.35)
    start_x = Inches(1.0)
    stats = [
        ("VALUE", "better search", "semantic + structured retrieval"),
        ("RISK", "safer listings", "Fair Housing blocking workflow"),
        ("DELIVERY", "production-ready", "FastAPI + UI + Docker deploy"),
    ]
    for i, (lbl, big, small) in enumerate(stats):
        add_metric_card(
            s,
            start_x + (card_w + gap) * i,
            Inches(5.35),
            card_w,
            Inches(1.35),
            lbl,
            big,
            small,
            color=ACCENT if i == 0 else ACCENT_AMBER if i == 1 else GOOD,
        )

    add_text(
        s, MX, Inches(7.0), USABLE_W, Inches(0.3),
        "Questions?", size=18, color=MUTED, italic=True, align=PP_ALIGN.CENTER
    )


def build(out_path: Path) -> None:
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    builders = [
        slide_01_title,
        slide_02_problem_approach,
        slide_03_architecture,
        slide_04_key_features,
        slide_05_challenge,
        slide_06_close,
    ]
    for fn in builders:
        fn(prs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out_path)
    print(f"Wrote {out_path}  ({len(builders)} slides)")


if __name__ == "__main__":
    here = Path(__file__).resolve().parents[1]
    build(here / "docs" / "final_presentation_short.pptx")
