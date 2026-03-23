import anthropic
import json
import re
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# MODULE 2 — LLM PARSER & STRUCTURED CONSTRAINT AUDITOR
# ============================================================
# v3 CHANGES:
#  1. OpenAI → Anthropic Claude (claude-sonnet-4-5)
#  2. GRADED SCORING — audit returns compliance_score [0,1]
#     instead of binary pass/fail. A rule 80% satisfied
#     returns conclusion_confidence=0.8, not 0.05.
#  3. Graded symbolic scoring for numerical rules:
#     how far is the value from the threshold?
#  4. Graded semantic scoring for boolean/qualitative rules:
#     Claude returns a 0-1 quality score, not just true/false.
#  All other optimisations preserved:
#  - Shared client cache, parallel rule parsing,
#    batched audit, result cache, scope-aware extraction,
#    unit normalisation, domain validity check.
# ============================================================

_MODEL       = "claude-sonnet-4-5"
_MAX_WORKERS = 8
_AUDIT_BATCH = 12

_CLIENT_CACHE: dict = {}
_PARSE_CACHE:  dict = {}


def _get_client(api_key: str) -> anthropic.Anthropic:
    key = (api_key or "").strip()
    if not key:
        raise ValueError(
            "Anthropic API key is empty. Set ANTHROPIC_API_KEY or paste it in the UI."
        )
    h = hashlib.md5(key.encode()).hexdigest()
    if h not in _CLIENT_CACHE:
        _CLIENT_CACHE[h] = anthropic.Anthropic(api_key=key)
    return _CLIENT_CACHE[h]


def _claude(api_key: str, prompt: str, max_tokens: int = 4096) -> str:
    """Single-turn Claude call using cached client."""
    client   = _get_client(api_key)
    response = client.messages.create(
        model    = _MODEL,
        max_tokens = max_tokens,
        messages = [{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ── Rule parsing ──────────────────────────────────────────────────────────────

_PARSE_SCHEMA = """{
    "original"        : "<the original rule text>",
    "rule_nature"     : "<constraint | observation>",
    "variable"        : "<snake_case identifier>",
    "constraint_type" : "<numerical_upper_bound | numerical_lower_bound | numerical_range | boolean | categorical>",
    "operator"        : "< | <= | > | >= | == | in_range | contains | excludes",
    "threshold"       : <primary numeric threshold as float, or null>,
    "threshold_low"   : <lower bound for in_range, or null>,
    "threshold_high"  : <upper bound for in_range, or null>,
    "unit"            : "<unit string or empty string>",
    "display"         : "<concise human-readable form>",
    "scope"           : "<always | initial | final | maximum | minimum | conditional | context_only>",
    "scope_hint"      : "<which occurrence to check>",
    "extraction_hint" : "<how to find this value in generated text>"
}"""


def parse_rule_to_constraint(rule_text: str, api_key: str, llm_config: dict = None) -> dict:
    """Parse a single NL rule into a structured constraint dict. Cached."""
    cache_key = hashlib.md5(rule_text.strip().encode()).hexdigest()
    if cache_key in _PARSE_CACHE:
        return _PARSE_CACHE[cache_key]

    prompt = f"""You are a formal logic compiler. Convert this rule into a JSON constraint.

RULE: "{rule_text}"

Return ONLY raw JSON (no markdown, no backticks) with this exact schema:
{_PARSE_SCHEMA}

RULE_NATURE: 'constraint' = must be enforced | 'observation' = input context fact
SCOPE: always | initial | final | maximum | minimum | conditional | context_only

Key: if the rule contains words like 'currently', 'is reading', 'now shows', 'at present'
it is an 'observation' with scope 'context_only', NOT a constraint enforced everywhere."""

    try:
        raw    = _call_llm(prompt, (llm_config or {"provider":"anthropic","model":"claude-sonnet-4-5","api_key":api_key}), max_tokens=512)
        clean  = raw.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)
        _PARSE_CACHE[cache_key] = result
        return result
    except Exception:
        slug = re.sub(r"[^a-z0-9]+", "_", rule_text.lower().strip())[:40].strip("_") or "rule"
        fallback = {
            "original": rule_text, "variable": slug,
            "constraint_type": "boolean", "operator": "==",
            "threshold": None, "threshold_low": None, "threshold_high": None,
            "unit": "", "display": rule_text, "scope": "always",
            "rule_nature": "constraint",
            "scope_hint": f"Verify: {rule_text}",
            "extraction_hint": f"Check whether text satisfies: {rule_text}",
        }
        _PARSE_CACHE[cache_key] = fallback
        return fallback


def parse_rules_parallel(rule_texts: list, api_key: str, llm_config: dict = None) -> list:
    """Parse ALL rules concurrently. N rules in ~time-of-1 rule."""
    if not rule_texts:
        return []
    results  = [None] * len(rule_texts)
    uncached = [(i, r) for i, r in enumerate(rule_texts)
                if hashlib.md5(r.strip().encode()).hexdigest() not in _PARSE_CACHE]
    for i, r in enumerate(rule_texts):
        h = hashlib.md5(r.strip().encode()).hexdigest()
        if h in _PARSE_CACHE:
            results[i] = _PARSE_CACHE[h]
    if not uncached:
        return results
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(uncached))) as pool:
        futures = {pool.submit(parse_rule_to_constraint, rule, api_key, llm_config): idx
                   for idx, rule in uncached}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    print(f"   [M2] Parsed {len(uncached)} rule(s) in {time.perf_counter()-t0:.2f}s")
    return results


# ── Scope instruction builder ─────────────────────────────────────────────────

def _build_scope_instruction(rule: dict) -> str:
    scope  = rule.get("scope", "always")
    hint   = rule.get("scope_hint", "")
    nature = rule.get("rule_nature", "constraint")
    if scope == "context_only" or nature == "observation":
        return (f"SCOPE=OBSERVATION: Given fact, not hard constraint. "
                f"Check output USES this value as starting context. {hint}")
    scope_map = {
        "initial"     : "SCOPE=INITIAL: Check ONLY first/starting value.",
        "final"       : "SCOPE=FINAL: Check ONLY last/final value.",
        "maximum"     : "SCOPE=MAXIMUM: Find and check the single highest value.",
        "minimum"     : "SCOPE=MINIMUM: Find and check the single lowest value.",
        "conditional" : f"SCOPE=CONDITIONAL: Only applies when condition met. {hint}",
        "always"      : "SCOPE=ALL: Check every occurrence; any violation = fail.",
    }
    return f"{scope_map.get(scope, 'SCOPE=ALL: Check every occurrence.')} {hint}".strip()


# ── Batched audit (v3: graded compliance_score) ───────────────────────────────

def structured_audit(draft_text: str, structured_rules: list, api_key: str, llm_config: dict = None) -> list:
    """
    Audit ALL rules in batched Claude calls.
    v3: returns compliance_score [0,1] per rule, not binary pass/fail.
    """
    if not structured_rules:
        return []
    all_results = []
    for start in range(0, len(structured_rules), _AUDIT_BATCH):
        chunk = structured_rules[start:start + _AUDIT_BATCH]
        all_results.extend(_batch_audit_chunk(draft_text, chunk, start, api_key, llm_config))
    return all_results


def _batch_audit_chunk(draft_text: str, rules: list, offset: int, api_key: str, llm_config: dict = None) -> list:
    """One batched audit prompt. Returns graded results."""
    rules_block = ""
    for i, rule in enumerate(rules):
        idx  = offset + i
        unit = rule.get("unit", "").strip() or "same unit as threshold"
        rules_block += f"""
RULE_{idx}:
  original:    "{rule.get('original', '')}"
  display:     {rule.get('display', rule.get('original', ''))}
  unit:        {unit}
  {_build_scope_instruction(rule)}
  hint:        {rule.get('extraction_hint', '')}
"""
    prompt = f"""You are a strict constraint auditor for a neurosymbolic AI system.

DRAFT TEXT:
\"\"\"{draft_text[:40000]}\"\"\"

RULES TO AUDIT:
{rules_block}

For EACH rule, extract the relevant value and assess compliance.

GRADED SCORING — compliance_score is a float [0.0, 1.0]:
  1.0 = perfectly satisfies the rule
  0.8 = mostly satisfies (minor issue, e.g. 8/10 facts have sources)
  0.5 = partially satisfies (e.g. some values meet constraint, some don't)
  0.2 = barely satisfies (minor partial compliance)
  0.0 = completely fails or variable not found

For NUMERICAL rules: if the value clearly satisfies the constraint → 1.0
  If it violates by a small margin → 0.3–0.5. Large violation → 0.0–0.1.
For BOOLEAN/QUALITATIVE rules: rate the actual quality of compliance 0–1.
  "Every fact must have a source" with 7/10 facts cited → 0.7
  "Report must be up to date" with mostly 2023 sources → 0.8

UNIT NORMALIZATION: convert to constraint unit before comparing.
  Example: "45 minutes" with unit "hours" → extracted_value_num = 0.75

Return ONLY a raw JSON array (no markdown) with one object per rule:
[
  {{
    "rule_index"           : <int>,
    "extracted_value_raw"  : "<exact phrase>",
    "extracted_value_num"  : <float or null>,
    "compliance_score"     : <float 0.0-1.0>,
    "unit_conversion_note" : "<note or 'none'>",
    "scope_note"           : "<which occurrence you checked>",
    "satisfies"            : <true if compliance_score >= 0.5>,
    "explanation"          : "<one sentence — be specific about what was or wasn't found>"
  }},
  ...
]

Rules:
- Respect SCOPE instructions — checking wrong occurrence = verification error
- Normalise units BEFORE comparing
- If variable not found → compliance_score=0.0, satisfies=false
- For strict rules: < 10 and value==10 → compliance_score=0.0"""

    try:
        t0  = time.perf_counter()
        raw = _call_llm(prompt, (llm_config or {"provider":"anthropic","model":"claude-sonnet-4-5","api_key":api_key}), max_tokens=4096)
        print(f"   [M2] Batch audit {len(rules)} rule(s) in {time.perf_counter()-t0:.2f}s")
        clean    = raw.strip().replace("```json", "").replace("```", "").strip()
        llm_list = json.loads(clean)
        by_index = {item["rule_index"]: item for item in llm_list}
        return [_build_result(offset + i, rule, by_index.get(offset + i))
                for i, rule in enumerate(rules)]
    except Exception as e:
        print(f"   [M2] Batch audit failed ({e}), falling back to parallel calls.")
        return _parallel_audit_fallback(draft_text, rules, offset, api_key, llm_config)


def _parallel_audit_fallback(draft_text: str, rules: list,
                              offset: int, api_key: str, llm_config: dict = None) -> list:
    results = [None] * len(rules)
    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(rules))) as pool:
        futures = {pool.submit(_single_audit, draft_text, rule, offset + i, api_key, llm_config): i
                   for i, rule in enumerate(rules)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results


def _single_audit(draft_text: str, rule: dict, idx: int, api_key: str, llm_config: dict = None) -> dict:
    """Single-rule audit fallback with graded scoring."""
    unit = rule.get("unit", "").strip() or "same unit as threshold"
    prompt = f"""You are a strict constraint auditor.

RULE: "{rule.get('original', '')}"
CONSTRAINT: {rule.get('display', rule.get('original', ''))}
UNIT: {unit}
{_build_scope_instruction(rule)}
HINT: {rule.get('extraction_hint', '')}

DRAFT:
\"\"\"{draft_text[:40000]}\"\"\"

Return ONLY raw JSON with graded compliance_score [0.0-1.0]:
{{
  "extracted_value_raw": "<exact phrase>",
  "extracted_value_num": <float or null>,
  "compliance_score": <float 0.0-1.0>,
  "unit_conversion_note": "<note>",
  "scope_note": "<which occurrence>",
  "satisfies": <true if score >= 0.5>,
  "explanation": "<one sentence>"
}}"""
    try:
        raw     = _call_llm(prompt, (llm_config or {"provider":"anthropic","model":"claude-sonnet-4-5","api_key":api_key}), max_tokens=512)
        clean   = raw.strip().replace("```json", "").replace("```", "").strip()
        llm_res = json.loads(clean)
    except Exception:
        llm_res = {
            "extracted_value_raw": "EXTRACTION FAILED",
            "extracted_value_num": None,
            "compliance_score": 0.0,
            "satisfies": False,
            "explanation": "Claude extraction failed — treating as violation.",
        }
    return _build_result(idx, rule, llm_res)


def _build_result(idx: int, rule: dict, llm_res: dict | None) -> dict:
    """
    Merge LLM audit result with symbolic checks and domain validation.
    v3: uses graded compliance_score → conclusion_confidence.
    """
    if llm_res is None:
        llm_res = {"extracted_value_raw": "NO RESULT", "extracted_value_num": None,
                   "compliance_score": 0.0, "satisfies": False,
                   "explanation": "No audit result returned."}

    rule_display  = rule.get("display", rule.get("original", ""))
    scope         = rule.get("scope", "always")

    num_val = llm_res.get("extracted_value_num")
    op      = rule.get("operator")
    thresh  = rule.get("threshold")
    t_low   = rule.get("threshold_low")
    t_high  = rule.get("threshold_high")

    # ── Boolean presence check detection ────────────────────────────────────
    # Rules like "Year 3 revenue figure must be specified" use threshold=1 as
    # a presence flag (1=present, 0=absent), NOT as a value to compare against.
    # Running 52_000_000 == 1.0 gives False even when the value is clearly there.
    # Detect these and skip numerical symbolic checking — defer to LLM result.
    ctype = rule.get("constraint_type", "")
    is_boolean_presence = (
        ctype == "boolean"
        and op == "=="
        and thresh is not None
        and float(thresh) in (0.0, 1.0)
    )

    # ── Symbolic double-check + GRADED distance score for numerical rules ────
    symbolic_override  = None
    symbolic_score     = None   # graded [0,1] from symbolic check

    if num_val is not None and op is not None and not is_boolean_presence:
        try:
            num_val = float(num_val)

            def _dist_score(val, limit, direction):
                """Score how well val satisfies val {direction} limit."""
                if direction == "lt":   # val < limit
                    if val < limit:    return 1.0
                    over = (val - limit) / max(abs(limit), 1e-9)
                    return max(0.0, 1.0 - over)
                elif direction == "le": # val <= limit
                    if val <= limit:   return 1.0
                    over = (val - limit) / max(abs(limit), 1e-9)
                    return max(0.0, 1.0 - over)
                elif direction == "gt": # val > limit
                    if val > limit:    return 1.0
                    under = (limit - val) / max(abs(limit), 1e-9)
                    return max(0.0, 1.0 - under)
                elif direction == "ge": # val >= limit
                    if val >= limit:   return 1.0
                    under = (limit - val) / max(abs(limit), 1e-9)
                    return max(0.0, 1.0 - under)
                return 1.0 if val == limit else 0.0

            if   op == "<"  and thresh is not None:
                symbolic_override = num_val < float(thresh)
                symbolic_score    = _dist_score(num_val, float(thresh), "lt")
            elif op == "<=" and thresh is not None:
                symbolic_override = num_val <= float(thresh)
                symbolic_score    = _dist_score(num_val, float(thresh), "le")
            elif op == ">"  and thresh is not None:
                symbolic_override = num_val > float(thresh)
                symbolic_score    = _dist_score(num_val, float(thresh), "gt")
            elif op == ">=" and thresh is not None:
                symbolic_override = num_val >= float(thresh)
                symbolic_score    = _dist_score(num_val, float(thresh), "ge")
            elif op == "==" and thresh is not None:
                symbolic_override = num_val == float(thresh)
                symbolic_score    = 1.0 if symbolic_override else max(0.0,
                    1.0 - abs(num_val - float(thresh)) / max(abs(float(thresh)), 1e-9))
            elif op == "in_range" and t_low is not None and t_high is not None:
                symbolic_override = float(t_low) <= num_val <= float(t_high)
                if symbolic_override:
                    symbolic_score = 1.0
                else:
                    span = float(t_high) - float(t_low)
                    if num_val < float(t_low):
                        symbolic_score = max(0.0, 1.0 - (float(t_low)-num_val)/max(span,1e-9))
                    else:
                        symbolic_score = max(0.0, 1.0 - (num_val-float(t_high))/max(span,1e-9))
        except (TypeError, ValueError):
            pass

    # ── Determine final compliance score ────────────────────────────────────
    llm_compliance = float(llm_res.get("compliance_score", 0.5 if llm_res.get("satisfies") else 0.0))

    if symbolic_score is not None:
        # Symbolic wins for numerical rules.
        # compliance_score (graded) goes into the LTN for a nuanced aggregate.
        # satisfies (boolean) uses the ACTUAL symbolic comparison result —
        # not the graded threshold — so 498 <= 400 is correctly False even
        # though the graded score is 0.755.
        final_compliance = symbolic_score
        final_satisfies  = bool(symbolic_override)   # exact boolean, not graded threshold
    else:
        # Semantic LLM score for boolean/categorical rules
        final_compliance = llm_compliance
        final_satisfies  = llm_compliance >= 0.5

    # ── Domain validity check ────────────────────────────────────────────────
    domain_warning = ""
    if num_val is not None:
        unit_toks = set(re.split(r"[_\s]+", rule.get("unit", "").lower()))
        var_toks  = set(re.split(r"[_\s]+", rule.get("variable", "").lower()))
        all_toks  = unit_toks | var_toks
        if all_toks & {"proportion", "ratio", "fraction"} and num_val > 1.0:
            domain_warning  = (f"DOMAIN VIOLATION: '{rule.get('variable','')}' is a proportion "
                               f"but {num_val} > 1.0 is impossible.")
            final_compliance = 0.0
            final_satisfies  = False
        elif all_toks & {"probability", "prob"} and (num_val > 1.0 or num_val < 0.0):
            domain_warning  = (f"DOMAIN VIOLATION: '{rule.get('variable','')}' probability "
                               f"{num_val} is outside [0,1].")
            final_compliance = 0.0
            final_satisfies  = False
        elif (all_toks & {"percent", "confidence"} or "%" in all_toks):
            if num_val > 100.0 or num_val < 0.0:
                domain_warning  = (f"DOMAIN VIOLATION: '{rule.get('variable','')}' percentage "
                                   f"{num_val} is outside [0,100].")
                final_compliance = 0.0
                final_satisfies  = False

    status = "PASS" if final_satisfies else "FAIL"
    method = "(symbolic)" if symbolic_override is not None else "(semantic)"
    print(f"   R{idx+1} {status} {method} score={final_compliance:.2f} — {rule_display[:55]}")
    if domain_warning:
        print(f"        DOMAIN: {domain_warning}")

    return {
        "rule_id"              : idx,
        "rule_display"         : rule_display,
        "original_rule"        : rule.get("original", ""),
        "scope"                : scope,
        "source_name"          : rule.get("source_name", ""),
        "extracted_value_raw"  : llm_res.get("extracted_value_raw", "N/A"),
        "extracted_value_num"  : num_val,
        "compliance_score"     : round(final_compliance, 4),
        "unit_conversion_note" : llm_res.get("unit_conversion_note", ""),
        "scope_note"           : llm_res.get("scope_note", ""),
        "domain_warning"       : domain_warning,
        "satisfies"            : final_satisfies,
        "symbolic_check_used"  : symbolic_override is not None,
        "premise_confidence"   : 1.0,
        "conclusion_confidence": round(final_compliance, 4),  # graded, not binary
        "explanation"          : llm_res.get("explanation", "No explanation."),
    }


def audit_results_to_ltn_entities(audit_results: list) -> dict:
    return {
        "entities": [
            {"name": r["rule_display"],
             "premise_confidence"   : r["premise_confidence"],
             "conclusion_confidence": r["conclusion_confidence"]}
            for r in audit_results
        ]
    }
