import anthropic
import wikipedia
import urllib.request
import urllib.parse
import json
import re
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# MODULE 4 — AGENTIC ROUTER: MULTI-SOURCE RESEARCH
# ============================================================
# v3 CHANGES:
#  - OpenAI GPT → Anthropic Claude (claude-sonnet-4-5)
#  - All other logic preserved:
#    concurrent Wikipedia + DuckDuckGo fetches,
#    single topic extraction (cached),
#    word-stem relevance filter + LLM domain gate,
#    deduplication against Wikipedia result.
# ============================================================

_MODEL        = "claude-sonnet-4-5"
_TOPIC_CACHE  : dict = {}
_CLIENT_CACHE : dict = {}


def _get_client(api_key: str) -> anthropic.Anthropic:
    key = (api_key or "").strip()
    if not key:
        raise ValueError("Anthropic API key is empty.")
    h = hashlib.md5(key.encode()).hexdigest()
    if h not in _CLIENT_CACHE:
        _CLIENT_CACHE[h] = anthropic.Anthropic(api_key=key)
    return _CLIENT_CACHE[h]


def _claude(api_key: str, prompt: str, max_tokens: int = 256) -> str:
    client   = _get_client(api_key)
    response = client.messages.create(
        model      = _MODEL,
        max_tokens = max_tokens,
        messages   = [{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ── PUBLIC ENTRY POINT ────────────────────────────────────────────────────────

def research_all_sources(query_text: str, api_key: str = None) -> list:
    """
    Search Wikipedia AND DuckDuckGo concurrently.
    Returns list of {source_name, context, reference, title}.
    """
    search_topic = _extract_search_topic(query_text, api_key)
    print(f"\n[Module 4] Search topic: '{search_topic}'")

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        wiki_f = pool.submit(_wiki_fetch, search_topic, query_text, api_key)
        ddg_f  = pool.submit(_ddg_fetch, search_topic)
        wiki_result = wiki_f.result()
        ddg_result  = ddg_f.result()
    print(f"   [M4] Both sources fetched in {time.perf_counter()-t0:.2f}s")

    results = []
    if wiki_result["context"] not in _FAILED_CONTEXTS:
        wiki_result["source_name"] = "Wikipedia"
        results.append(wiki_result)
    else:
        print("   [M4] Wikipedia: no usable result.")

    if ddg_result["context"] not in _FAILED_CONTEXTS:
        if not _same_topic(ddg_result, wiki_result):
            ddg_result["source_name"] = "DuckDuckGo"
            results.append(ddg_result)
        else:
            print("   [M4] DuckDuckGo: same topic as Wikipedia — skipping.")
    else:
        print("   [M4] DuckDuckGo: no usable result.")

    return results


# ── SOURCE 1: WIKIPEDIA ──────────────────────────────────────────────────────

def _wiki_fetch(search_query: str, original_query: str, api_key: str = None) -> dict:
    print(f"\n[Module 4] Wikipedia search: '{search_query}'")
    try:
        candidates = wikipedia.search(search_query, results=5)
        if not candidates:
            return _failed("No Wikipedia results found.")

        for title in candidates:
            try:
                if not _is_relevant(title, search_query):
                    print(f"   [M4] Wikipedia: skipping '{title}' (word filter)")
                    continue
                page    = wikipedia.page(title, auto_suggest=False)
                summary = wikipedia.summary(title, sentences=5, auto_suggest=False)

                # LLM domain gate — catches wrong-domain articles word-overlap misses
                if api_key and not _llm_domain_check(page.title, summary,
                                                      search_query, api_key):
                    print(f"   [M4] Wikipedia: LLM rejected '{page.title}'")
                    continue

                print(f"   ✅ Wikipedia: {page.url}")
                return {"context": summary, "reference": page.url, "title": page.title}

            except wikipedia.exceptions.DisambiguationError as e:
                try:
                    page    = wikipedia.page(e.options[0], auto_suggest=False)
                    summary = wikipedia.summary(e.options[0], sentences=5, auto_suggest=False)
                    if api_key and not _llm_domain_check(page.title, summary,
                                                          search_query, api_key):
                        continue
                    print(f"   ✅ Wikipedia: {page.url}")
                    return {"context": summary, "reference": page.url, "title": page.title}
                except Exception:
                    continue
            except Exception:
                continue

        return _failed("No relevant Wikipedia results found.")
    except Exception as e:
        return _failed(f"Wikipedia search failed: {e}")


def wiki_search_fallback(query_text: str, api_key: str = None) -> dict:
    """Legacy public alias."""
    topic = _extract_search_topic(query_text, api_key)
    return _wiki_fetch(topic, query_text, api_key)


# ── SOURCE 2: DUCKDUCKGO ─────────────────────────────────────────────────────

def _ddg_fetch(search_query: str) -> dict:
    print(f"\n[Module 4] DuckDuckGo search: '{search_query}'")
    try:
        encoded = urllib.parse.quote_plus(search_query)
        url     = (f"https://api.duckduckgo.com/?q={encoded}"
                   f"&format=json&no_redirect=1&no_html=1&skip_disambig=1")
        req = urllib.request.Request(url, headers={"User-Agent": "NeurosymbolicAI/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        abstract = data.get("AbstractText", "").strip()
        abs_url  = data.get("AbstractURL", "").strip()
        abs_src  = data.get("AbstractSource", "DuckDuckGo")

        if abstract and len(abstract) > 80:
            print(f"   ✅ DuckDuckGo ({abs_src}): {abs_url or 'N/A'}")
            return {
                "context"  : abstract,
                "reference": abs_url or f"https://duckduckgo.com/?q={encoded}",
                "title"    : data.get("Heading", search_query),
            }

        snippets = []
        for item in data.get("RelatedTopics", []):
            if isinstance(item, dict):
                t = item.get("Text", "").strip()
                if t and len(t) > 40:
                    snippets.append(t)
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, dict):
                        t = sub.get("Text", "").strip()
                        if t and len(t) > 40:
                            snippets.append(t)

        if snippets:
            print("   ✅ DuckDuckGo: related topics fallback")
            return {
                "context"  : " ".join(snippets[:4]),
                "reference": f"https://duckduckgo.com/?q={encoded}",
                "title"    : search_query,
            }
        return _failed("DuckDuckGo returned no usable content.")
    except Exception as e:
        return _failed(f"DuckDuckGo failed: {e}")


def duckduckgo_search(query_text: str, api_key: str = None) -> dict:
    """Legacy public alias."""
    topic = _extract_search_topic(query_text, api_key)
    return _ddg_fetch(topic)


# ── HELPERS ───────────────────────────────────────────────────────────────────

_FAILED_CONTEXTS = {
    "No Wikipedia results found.",
    "No relevant Wikipedia results found.",
    "DuckDuckGo returned no usable content.",
}


def _failed(reason: str) -> dict:
    return {"context": reason, "reference": "None", "title": "None"}


def _extract_search_topic(query_text: str, api_key: str = None) -> str:
    """Extract 3-5 word search topic. Cached per query."""
    cache_key = hashlib.md5(query_text.strip().lower().encode()).hexdigest()
    if cache_key in _TOPIC_CACHE:
        return _TOPIC_CACHE[cache_key]

    if api_key:
        try:
            prompt = (
                f'Extract the core subject of this request as a short search query '
                f'(3-5 words max). Focus on TOPIC only — ignore personal details.\n'
                f'Return ONLY the search query string, no explanation, no quotes.\n\n'
                f'Request: "{query_text}"\n\nSearch query:'
            )
            result = _claude(api_key, prompt, max_tokens=32)
            result = result.replace('"', "").replace("'", "").strip()[:60]
            _TOPIC_CACHE[cache_key] = result
            return result
        except Exception:
            pass

    result = _simple_clean(query_text)
    _TOPIC_CACHE[cache_key] = result
    return result


def _simple_clean(q: str) -> str:
    filler = ["i want to", "i want you to", "i need to", "help me", "can you",
              "please", "create", "make", "build", "generate", "write", "give me"]
    q = q.lower().strip()
    for p in filler:
        q = q.replace(p, " ")
    return " ".join(w for w in q.split() if len(w) > 2)[:60]


def _llm_domain_check(title: str, summary: str, query: str, api_key: str) -> bool:
    """
    Ask Claude whether a Wikipedia article is domain-relevant to the query.
    Only rejects articles where the domain is CLEARLY wrong — e.g. a music
    article for an engineering query. Does NOT reject articles that are
    topically adjacent (e.g. CubeSat article for CubeSat power management).
    Defaults to accepting on any error or uncertainty.
    """
    try:
        prompt = (
            f'Wikipedia article: "{title}"\n'
            f'User query topic: "{query}"\n'
            f'Article summary: "{summary[:300]}"\n\n'
            f'Is this article from a CLEARLY DIFFERENT domain than the query? '
            f'Only answer "yes" if the domains are completely unrelated '
            f'(e.g. a cooking article for a physics query). '
            f'If the article is even tangentially related to the query topic, answer "no".\n'
            f'Answer ONLY "yes" (reject) or "no" (accept).'
        )
        answer = _claude(api_key, prompt, max_tokens=10)
        # "yes" means clearly different domain → reject
        # "no" means same or related domain → accept
        return not answer.strip().lower().startswith("yes")
    except Exception:
        return True  # default to accepting on error


def _is_relevant(title: str, query: str) -> bool:
    """Token-overlap relevance check (fast, no LLM)."""
    stop = {"the","a","an","of","in","on","at","to","for","and","or",
            "with","by","from","as","is","are","its","it","that","this"}

    def tok(text):
        return [t for t in re.split(r"[\s\-_/]+", text.lower())
                if t not in stop and len(t) > 2]

    qt = set(tok(query))
    tt = set(tok(title))
    if not qt:
        return True
    if qt & tt:
        return True
    for qw in qt:
        for tw in tt:
            if len(qw) >= 4 and len(tw) >= 4:
                if tw.startswith(qw) or qw.startswith(tw):
                    return True
    return False


def _same_topic(a: dict, b: dict) -> bool:
    stop = {"the","a","an","of","in","on","at","to","for","and"}
    wa = {w for w in a.get("title","").lower().split() if w not in stop and len(w) > 2}
    wb = {w for w in b.get("title","").lower().split() if w not in stop and len(w) > 2}
    return len(wa & wb) >= 2