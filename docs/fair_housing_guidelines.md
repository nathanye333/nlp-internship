# Fair Housing Compliance: Team Guidelines

Who this is for: anyone on the listings team who drafts, edits, approves, or
publishes real-estate advertisements. Read this before writing listing copy
and before shipping changes to `ComplianceChecker`.

---

## 1. Why this matters

The federal **Fair Housing Act** (42 U.S.C. 3601 et seq.) makes it unlawful to
"make, print, or publish" any advertisement for the sale or rental of a
dwelling that indicates a preference, limitation, or discrimination based on
a **protected class**. The prohibition applies to the *advertisement itself*,
regardless of the advertiser's intent. There is no "I didn't mean it that
way" defense.

Federally protected classes:

1. Race
2. Color
3. National origin
4. Religion
5. Sex (including gender identity and sexual orientation, per 24 CFR 100.5)
6. Familial status (presence of children under 18, pregnancy, custody of a
   minor)
7. Disability (physical or mental)

Many states and cities add more classes — commonly **source of income**
(Section 8 vouchers), **age**, **marital status**, **military/veteran
status**, and **lawful occupation**. The checker ships with a `source_of_income`
and `age` category; add state-specific rules via `extra_patterns`.

Penalties for a HUD-enforced violation start at roughly **$25,000 per first
offense** for a company and climb from there. The reputational cost is
typically larger.

## 2. The golden rule

> Describe the **property**, not the **ideal occupant**.

If a sentence could be rewritten to describe what is being sold (square
footage, appliances, location, amenities) instead of who should buy or rent
it, rewrite it.

## 3. Severity model

`ComplianceChecker` returns one of three severities for every finding:

| Severity  | Meaning                                                                                                                            | Workflow action                        |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| `error`   | The phrase is almost always a violation. Example: `"adults only"`, `"no Section 8"`, `"English-speaking tenants"`.                 | **Block publication** until rewritten. |
| `warning` | Context-dependent. Example: `"perfect for young professionals"`, `"walking distance to the church"`, `"mature tenants"`.           | **Human reviewer must confirm** before publishing. |
| `info`    | Legal today but historically tied to steering. Example: `"great schools"`, `"safe neighborhood"`, `"exclusive community"`.         | **Log and keep editor aware**. No block. |

The goal of three levels is to get real-world adoption. If we block every
borderline phrase the team starts disabling the check; if we silently allow
errors we ship violations. Errors block, warnings escalate, info teaches.

## 4. Category-by-category reference

### 4.1 Race & color
- Never mention race, color, or ethnicity of current or desired occupants.
- Avoid demographic descriptors of neighborhoods ("traditionally white",
  "ethnic enclave", "diverse area"). Even positive-sounding words like
  "diverse" have been found to constitute steering.
- Allowed: objective descriptors (walkability, zoning, transit access).

### 4.2 National origin
- No references to country of origin ("Americans only", "no foreigners").
- **Language requirements** are proxies: "English-speaking required" is
  almost always a violation. You may list languages *staff* speak
  ("Spanish-speaking agent available") but not require them of tenants.

### 4.3 Religion
- Never describe a "Christian community", "Jewish neighborhood", etc.
- **Landmarks are allowed but flagged for review**: "two blocks from St.
  Mary's Cathedral" is factual and legal, but the checker will emit a
  `warning` so a reviewer can confirm the landmark is used for directions,
  not as a religious preference signal. If you use a religious landmark,
  balance it with a secular one (transit stop, park).

### 4.4 Sex, gender identity, sexual orientation
- No gender restrictions on tenants ("women only"). Narrow shared-living
  exceptions (e.g., sharing a bathroom in an owner-occupied unit) exist but
  require legal review before using.
- No orientation-based preferences or exclusions.
- Avoid "bachelor pad", "man cave only", etc.

### 4.5 Familial status
- **"Adults only", "no children", "mature tenants only"** are all errors.
- The only exemption is **housing for older persons** (HOPA): communities
  that are 100% 62+, or that qualify under the 80%-55+ rule with documented
  HUD compliance. That documentation lives with the listing metadata, not in
  the description. State the community type ("55+ active-adult community")
  rather than excluding children.
- **Occupancy limits** must apply to *all* occupants, not to children
  specifically. HUD's Keating Memo guideline is roughly two persons per
  bedroom, subject to state/local building code. "No more than 2 children"
  is an error; "maximum 4 occupants per local code" is not.
- Describing the property's suitability ("home sits on a steep cliff lot")
  is fine. Stating that it is "not suitable for children" is a violation.

### 4.6 Disability
- Never exclude occupants on the basis of disability or mobility
  ("no wheelchairs", "must be able-bodied").
- **Service and assistance animals are not pets**. A "no pets" policy may
  still require you to accommodate service/emotional-support animals as a
  reasonable accommodation. Never advertise "no service animals".
- You **may** describe physical access honestly: "third-floor walk-up, no
  elevator" is a factual statement about the property and is legal, though
  the checker emits a `warning` so a human verifies it is descriptive rather
  than exclusionary.

### 4.7 Source of income (many jurisdictions)
- "No Section 8" and "no vouchers" are illegal in an expanding list of
  states and cities (CA, NY, NJ, MA, WA, DC, Chicago, Seattle, etc.).
- Treat them as errors by default in the shared checker. Override only with
  explicit legal sign-off for a specific market.

### 4.8 Age
- Age limits ("must be over 55") are only permissible in HOPA-qualified
  communities. Document that status separately.

### 4.9 Steering language (`info`)
Words like "safe", "exclusive", "desirable", and "great schools" have a long
history of being used to steer buyers by race or income. They are not
automatically illegal, but they are vague and high-risk. The checker emits
`info` for these so editors learn to replace them with objective facts
("gated entry, on-site security"; "Blue Ribbon School District (buyer to
verify)").

## 5. Using `ComplianceChecker`

```python
from scripts.compliance_checker import ComplianceChecker

checker = ComplianceChecker()
report = checker.check_listing(listing.description)

if not report.compliant:
    raise PublishingBlocked(report.blocking())

for w in report.violations:
    logger.log(w.severity, "%s: %s", w.category, w.message)
```

Add state-specific rules without forking the library:

```python
checker = ComplianceChecker(extra_patterns={
    "state_ca_source_of_income": [
        (r"(?<![A-Za-z0-9])no\s+vouchers(?![A-Za-z0-9])",
         "error",
         "CA Civil Code 12955 bars source-of-income discrimination",
         "Remove. State financial qualifications instead."),
    ],
})
```

See `scripts/listing_submission_example.py` for the full workflow.

## 6. Rewrite cheat sheet

| Instead of                         | Write                                                                 |
| ---------------------------------- | --------------------------------------------------------------------- |
| "Perfect for a Christian family"   | "Three-bedroom home near downtown"                                    |
| "Adults only community"            | "55+ active-adult community (HOPA documentation on file)"             |
| "No children"                      | *(delete; state factual property details)*                            |
| "No Section 8"                     | "Monthly income of 3x rent required"                                  |
| "English-speaking only"            | *(delete)*                                                            |
| "Safe neighborhood, great schools" | "Gated entry with 24-hr security; Lincoln Unified School District (buyer to verify)" |
| "Walk to church"                   | "Walk to Main Street shops and the bus line"                          |
| "Bachelor pad"                     | "Open studio with Murphy bed"                                         |
| "Must be able-bodied"              | *(delete; describe property access factually)*                        |

## 7. Quality targets

The checker is validated against an internal labelled corpus
(`tests/test_compliance_checker.py`). We hold ourselves to:

- **Recall = 1.0** on known violations — no false negatives.
- **Precision > 0.8** on the combined corpus — few false positives.

When you add a new pattern, also add at least one positive and one clean
example to the test corpus. Re-run `pytest tests/test_compliance_checker.py`
before merging.

## 8. When in doubt

- Ask legal. HUD's advertising guidance is at
  <https://www.hud.gov/program_offices/fair_housing_equal_opp>.
- Prefer silence over a risky phrase. If a sentence does not help a buyer
  evaluate the property, delete it.
- A `warning` is not a green light. It is a request for a human to look.
