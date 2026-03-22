import math

# ============================================================
# MODULE 1 — LTN CORE: LOGIC TENSOR NETWORK VERIFICATION
# ============================================================
# Pure-math implementation — no TensorFlow, no ltn package.
# Mathematically identical semantics:
#
#   Reichenbach implication:  P → C  =  1 − P + P·C
#   pMeanError (p=2):   1 − (mean((1−φᵢ)²))^(1/2)
#
# v3 CHANGE: conclusion_confidence is now a GRADED float [0,1]
# instead of binary (1.0 or 0.05). A rule 80% satisfied scores
# 0.8, not 0.0. This makes the LTN score genuinely meaningful.
# ============================================================

_P = 2


def _reichenbach_implies(premise: float, conclusion: float) -> float:
    """Fuzzy Reichenbach implication: 1 − p + p·c, clamped to [0,1]."""
    val = 1.0 - premise + premise * conclusion
    return max(0.0, min(1.0, val))


def _pmean_error_forall(impl_values: list) -> float:
    """
    pMeanError universal quantifier.
    Score = 1 − (mean((1 − φᵢ)^p))^(1/p)
    """
    if not impl_values:
        return 1.0
    errors     = [(1.0 - v) ** _P for v in impl_values]
    mean_err   = sum(errors) / len(errors)
    score      = 1.0 - math.pow(mean_err, 1.0 / _P)
    return max(0.0, min(1.0, score))


def evaluate_generic_logic(parsed_data: dict) -> float:
    """
    Runs LTN verification over logical entities.
    Each entity: {premise_confidence, conclusion_confidence}
    conclusion_confidence may be any float in [0,1] — graded scoring.
    Returns aggregate score in [0,1].
    """
    print("\n[Module 1] Running LTN verification (graded pure-math engine)...")
    entities = parsed_data.get("entities", [])
    if not entities:
        return 1.0

    impl_values = [
        _reichenbach_implies(e["premise_confidence"], e["conclusion_confidence"])
        for e in entities
    ]
    score = _pmean_error_forall(impl_values)
    print(f"   LTN Score: {score:.4f}  "
          f"(avg conclusion confidence: "
          f"{sum(e['conclusion_confidence'] for e in entities)/len(entities):.3f})")
    return score


def verify_and_report(audit_results: list) -> tuple:
    """
    Main entry point.
    Takes audit_results from m2.structured_audit().
    Returns (ltn_score: float, violations: list[dict]).
    violations = rules with satisfies==False (conclusion_confidence < 0.5).
    """
    parsed_data = {
        "entities": [
            {
                "name"                  : r["rule_display"],
                "premise_confidence"    : r["premise_confidence"],
                "conclusion_confidence" : r["conclusion_confidence"],
            }
            for r in audit_results
        ]
    }
    ltn_score  = evaluate_generic_logic(parsed_data)
    violations = [r for r in audit_results if not r["satisfies"]]
    return ltn_score, violations