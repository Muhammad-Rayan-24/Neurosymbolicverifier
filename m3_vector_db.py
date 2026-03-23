import time
import uuid
import json
from datetime import datetime

# ============================================================
# MODULE 3 — SYMBOLIC MEMORY (Qdrant + sentence-transformers)
# ============================================================
# v3 CHANGES vs original:
#  - Embeddings: OpenAI text-embedding-3-small → sentence-transformers
#    (free, local, no API key needed, ~80ms per batch)
#  - Model: all-MiniLM-L6-v2  (384-dim, fast, accurate enough)
#  - Everything else identical: Qdrant in-memory or cloud,
#    3 collections (rules / sources / audit), rich payloads
#    for Second Brain visualisation.
# ============================================================

# ── Qdrant Cloud credentials ──────────────────────────────────────────────────
# Edit these two lines to point at your Qdrant instance.
# Leave QDRANT_URL as "" to use fast in-memory Qdrant (no persistence).
# When QDRANT_URL is set all runs are persisted in the cloud cluster and
# every record is tagged with a run_id so queries never bleed across runs.
QDRANT_URL     = "https://d217835b-8105-4b9b-a47d-439ff47e0a44.sa-east-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.ZeQ9egz-8PwY5US-q_mAxhw1dEEwPWzEhFyg5KaDLuM"
# ─────────────────────────────────────────────────────────────────────────────

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        Filter, FieldCondition, MatchValue,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

_EMBED_DIM   = 384          # all-MiniLM-L6-v2
_COLLECTIONS = ["rules", "sources", "audit"]
_CLIENT: "QdrantClient | None" = None
_EMBED_MODEL = None         # lazy-loaded


def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
    return _EMBED_MODEL


def _embed(text: str) -> list:
    """Embed a single text string. Returns float list."""
    model = _get_embed_model()
    vec   = model.encode([text[:512]], normalize_embeddings=True)
    return vec[0].tolist()


def _batch_embed(texts: list) -> list:
    """Embed a list of texts in one pass."""
    model   = _get_embed_model()
    trimmed = [t[:512] for t in texts]
    vecs    = model.encode(trimmed, normalize_embeddings=True, batch_size=32)
    return [v.tolist() for v in vecs]


def _get_client(url: str = None, api_key: str = None) -> "QdrantClient":
    global _CLIENT
    if _CLIENT is None:
        # Fall back to module-level constants if no args provided
        _url = url or QDRANT_URL or ""
        _key = api_key or QDRANT_API_KEY or ""
        if _url:
            _CLIENT = QdrantClient(url=_url, api_key=_key or None)
            print(f"[M3] Connected to Qdrant at {_url}")
        else:
            _CLIENT = QdrantClient(":memory:")
            print("[M3] Using in-memory Qdrant")
    return _CLIENT


def reset_client():
    """
    Tear down the cached Qdrant client so the next setup_memory() call
    creates a fresh connection.  Called on every pipeline reset so that
    in-memory Qdrant is completely wiped and cloud Qdrant gets a fresh
    connection object — preventing stale data from a previous run from
    leaking into the next run's retrieve_context() call.
    """
    global _CLIENT
    _CLIENT = None
    print("[M3] Qdrant client reset — fresh connection on next call.")


def setup_memory(url: str = None, qdrant_api_key: str = None, **kwargs):
    """
    Initialise Qdrant collections. Returns the client.
    If url/qdrant_api_key are not provided, falls back to the module-level
    QDRANT_URL / QDRANT_API_KEY constants defined at the top of this file.
    kwargs absorbed for API compatibility (old code passed openai_api_key).
    """
    if not QDRANT_AVAILABLE:
        raise ImportError("qdrant-client not installed. Run: pip install qdrant-client")
    # Use module constants as defaults so cloud Qdrant is always used when
    # the UI fields are left blank
    _url = url or QDRANT_URL or ""
    _key = qdrant_api_key or QDRANT_API_KEY or ""
    client   = _get_client(url=_url or None, api_key=_key or None)
    existing = {c.name for c in client.get_collections().collections}
    for name in _COLLECTIONS:
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=_EMBED_DIM, distance=Distance.COSINE),
            )
            print(f"[M3] Created collection '{name}'")
    return client


# ── Store functions ───────────────────────────────────────────────────────────

def store_all_rules(client, structured_rules: list, run_id: str = "", **kwargs) -> list:
    """Batch-embed and store all rules. Returns list of point IDs."""
    if not structured_rules:
        return []
    texts   = [r.get("display", r.get("original", str(r))) for r in structured_rules]
    vectors = _batch_embed(texts)
    ts      = datetime.utcnow().isoformat()
    points, ids = [], []
    for rule, vec, text in zip(structured_rules, vectors, texts):
        pid = str(uuid.uuid4())
        ids.append(pid)
        points.append(PointStruct(
            id=pid, vector=vec,
            payload={
                "text"            : text,
                "original"        : rule.get("original", text),
                "variable"        : rule.get("variable", ""),
                "constraint_type" : rule.get("constraint_type", "boolean"),
                "operator"        : rule.get("operator", ""),
                "threshold"       : rule.get("threshold"),
                "threshold_low"   : rule.get("threshold_low"),
                "threshold_high"  : rule.get("threshold_high"),
                "unit"            : rule.get("unit", ""),
                "scope"           : rule.get("scope", "always"),
                "rule_nature"     : rule.get("rule_nature", "constraint"),
                "source_name"     : rule.get("source_name", ""),
                "run_id"          : run_id,
                "stored_at"       : ts,
                "record_type"     : "rule",
            }
        ))
    client.upsert(collection_name="rules", points=points)
    print(f"   [M3] Stored {len(points)} rule(s) in Qdrant (run={run_id}).")
    return ids


def store_source(client, source: dict, run_id: str = "", **kwargs) -> str:
    text = source.get("context", "")[:512]
    pid  = str(uuid.uuid4())
    vec  = _embed(text)
    client.upsert(collection_name="sources", points=[PointStruct(
        id=pid, vector=vec,
        payload={
            "text"        : text,
            "title"       : source.get("title", ""),
            "reference"   : source.get("reference", ""),
            "source_name" : source.get("source_name", ""),
            "run_id"      : run_id,
            "stored_at"   : datetime.utcnow().isoformat(),
            "record_type" : "source",
        }
    )])
    return pid


def store_audit_result(client, audit_result: dict, run_id: str = "", **kwargs) -> str:
    text = (f"{audit_result.get('rule_display','')} — "
            f"{'PASS' if audit_result.get('satisfies') else 'FAIL'}: "
            f"{audit_result.get('explanation','')}")
    pid = str(uuid.uuid4())
    vec = _embed(text)
    client.upsert(collection_name="audit", points=[PointStruct(
        id=pid, vector=vec,
        payload={
            "text"                 : text,
            "rule_display"         : audit_result.get("rule_display", ""),
            "satisfies"            : audit_result.get("satisfies", False),
            "compliance_score"     : audit_result.get("compliance_score", 0.0),
            "scope"                : audit_result.get("scope", ""),
            "extracted_value_raw"  : audit_result.get("extracted_value_raw", ""),
            "extracted_value_num"  : audit_result.get("extracted_value_num"),
            "explanation"          : audit_result.get("explanation", ""),
            "symbolic_check_used"  : audit_result.get("symbolic_check_used", False),
            "premise_confidence"   : audit_result.get("premise_confidence", 1.0),
            "conclusion_confidence": audit_result.get("conclusion_confidence", 0.0),
            "domain_warning"       : audit_result.get("domain_warning", ""),
            "run_id"               : run_id,
            "stored_at"            : datetime.utcnow().isoformat(),
            "record_type"          : "audit",
        }
    )])
    return pid


# ── Retrieve functions ────────────────────────────────────────────────────────

def retrieve_context(client, query_text: str, n_results: int = 4,
                     run_id: str = "", **kwargs) -> list:
    """
    Semantic search across rules and sources for the CURRENT run only.

    run_id is mandatory for isolation — when provided, only records tagged
    with that run_id are searched.  This prevents context from a previous
    run leaking into the current generation prompt (the cross-query
    contamination bug).  When run_id is empty the filter is skipped and
    all records are searched (legacy / test behaviour).
    """
    try:
        vec = _embed(query_text)
        texts = []

        # Build a run_id filter so we only pull context from THIS run
        _run_filter = None
        if run_id:
            try:
                _run_filter = Filter(
                    must=[FieldCondition(
                        key="run_id",
                        match=MatchValue(value=run_id),
                    )]
                )
            except Exception:
                _run_filter = None  # qdrant-client version doesn't support it — skip

        for col in ("rules", "sources"):
            count = client.get_collection(col).points_count
            if count == 0:
                continue
            try:
                # qdrant-client >= 1.10 uses query_points()
                qr = client.query_points(
                    collection_name=col,
                    query=vec,
                    query_filter=_run_filter,
                    limit=min(n_results, count),
                    with_payload=True,
                )
                hits = qr.points
            except AttributeError:
                # fallback for older qdrant-client
                hits = client.search(
                    collection_name=col,
                    query_vector=vec,
                    query_filter=_run_filter,
                    limit=min(n_results, count),
                )
            texts.extend(h.payload.get("text", "") for h in hits)
        return texts[:n_results]
    except Exception as e:
        print(f"   [M3] Retrieval error: {e}")
        return []


def get_all_records(client, collection: str, limit: int = 200,
                    run_id: str = "") -> list:
    """
    Fetch stored points for Second Brain visualisation.
    When run_id is provided, returns only records from that run so the
    Second Brain tab shows the current session's data, not every run ever.
    """
    try:
        _scroll_filter = None
        if run_id:
            try:
                _scroll_filter = Filter(
                    must=[FieldCondition(
                        key="run_id",
                        match=MatchValue(value=run_id),
                    )]
                )
            except Exception:
                _scroll_filter = None
        result = client.scroll(
            collection_name=collection,
            scroll_filter=_scroll_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [pt.payload for pt in result[0]]
    except Exception as e:
        print(f"   [M3] scroll error ({collection}): {e}")
        return []


def get_collection_stats(client) -> dict:
    stats = {}
    for name in _COLLECTIONS:
        try:
            stats[name] = client.get_collection(name).points_count
        except Exception:
            stats[name] = 0
    return stats


def store_knowledge(client, doc_id: str, text: str, **kwargs) -> str:
    """Legacy compatibility shim."""
    return store_source(client, {"context": text, "title": doc_id,
                                  "reference": "", "source_name": doc_id})
