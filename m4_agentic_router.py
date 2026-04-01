import wikipedia
import urllib.request
import urllib.parse
import urllib.error
import json
import re
import hashlib
import time
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

_TOPIC_CACHE  : dict = {}
_CLIENT_CACHE : dict = {}
_O_SERIES = {'o3','o3-pro','o3-mini','o4-mini','o1','o1-mini','o1-pro'}

# Exact failure context strings returned by _failed()
_FAILED_CONTEXTS = frozenset([
    'No Wikipedia results found.',
    'No relevant Wikipedia results found.',
    'DuckDuckGo returned no usable content.',
])

# Prefixes that mark a result as a failed/unusable fetch — checked with startswith()
# so new error messages from _failed() are caught without needing to enumerate them all.
_FAILED_PREFIXES = (
    "No Wikipedia",
    "No relevant Wikipedia",
    "DuckDuckGo returned",
    "Web search returned",
    "Web search:",
    "Wikipedia search failed",
    "DuckDuckGo failed",
    "Failed to fetch",
    "HTTP ",
    "URL error",
    "PDF at ",
    "VIDEO URL BLOCKED",
    "JS-ONLY URL BLOCKED",
    "Google Search returned",
    "Google Search API failed",
    "Google Search: could not",
)

def _is_valid_source(src: dict) -> bool:
    """
    Return True only if this source dict contains genuine retrievable content.
    Single source of truth — used both inside M4 and in app.py filtering.
    """
    ctx = (src.get("context") or "").strip()

    # Must have meaningful content (60 chars covers a short but real paragraph)
    if len(ctx) < 60:
        print(f"   [M4/filter] Rejected — too short ({len(ctx)} chars): {ctx[:60]!r}")
        return False

    # Must not be an exact known failure string
    if ctx in _FAILED_CONTEXTS:
        print(f"   [M4/filter] Rejected — known failure string: {ctx[:80]!r}")
        return False

    # Must not start with any failure prefix
    for p in _FAILED_PREFIXES:
        if ctx.startswith(p):
            print(f"   [M4/filter] Rejected — failure prefix {p!r}: {ctx[:80]!r}")
            return False

    return True

# ── URL categories that cannot be scraped for text content ────────────────────
# Video platforms: JS-rendered players, no transcript in plain HTML.
# Pasting a YouTube/Vimeo link will return garbage HTML — detect and reject
# with a clear, actionable message instead of hallucinating an answer.
_VIDEO_DOMAINS = frozenset([
    "youtube.com", "youtu.be", "vimeo.com", "dailymotion.com",
    "twitch.tv", "tiktok.com", "rumble.com", "odysee.com",
    "bilibili.com", "nicovideo.jp", "streamable.com",
])
# JS-only / login-walled domains that always return unusable HTML
_JS_ONLY_DOMAINS = frozenset([
    "twitter.com", "x.com", "instagram.com", "facebook.com",
    "linkedin.com", "threads.net", "discord.com", "slack.com",
    "notion.so", "figma.com", "canva.com", "docs.google.com",
    "sheets.google.com", "drive.google.com",
])


def _call_llm(prompt, api_key, llm_config=None, max_tokens=256):
    cfg      = llm_config or {'provider':'anthropic','model':'claude-sonnet-4-6','api_key':api_key}
    provider = cfg.get('provider','anthropic').lower()
    model    = cfg.get('model','claude-sonnet-4-6')
    key      = cfg.get('api_key', api_key or '').strip()
    cache_k  = hashlib.md5((provider+model+key).encode()).hexdigest()
    if provider == 'anthropic':
        import anthropic as _ant
        if cache_k not in _CLIENT_CACHE:
            _CLIENT_CACHE[cache_k] = _ant.Anthropic(api_key=key)
        r = _CLIENT_CACHE[cache_k].messages.create(
            model=model, max_tokens=max_tokens,
            messages=[{'role':'user','content':prompt}])
        return r.content[0].text
    elif provider == 'openai':
        import openai as _oai
        if cache_k not in _CLIENT_CACHE:
            _CLIENT_CACHE[cache_k] = _oai.OpenAI(api_key=key)
        client = _CLIENT_CACHE[cache_k]
        kw = {'model': model, 'messages': [{'role':'user','content':prompt}]}
        if model in _O_SERIES or model.startswith('gpt-5'):
            kw['max_completion_tokens'] = max_tokens
        else:
            kw['max_tokens'] = max_tokens
        return client.chat.completions.create(**kw).choices[0].message.content
    elif provider == 'google':
        import google.generativeai as _genai
        _genai.configure(api_key=key)
        return _genai.GenerativeModel(model).generate_content(prompt).text
    raise ValueError(f'Unknown provider: {provider}')


def _claude(api_key, prompt, max_tokens=256):
    return _call_llm(prompt, api_key, None, max_tokens)


def research_all_sources(query_text, api_key=None, llm_config=None, research_config=None):
    '''
    Fetch from all enabled research sources concurrently.
    research_config: {
        wikipedia: bool, duckduckgo: bool, custom_urls: [str]
    }
    '''
    if research_config is None:
        research_config = {'wikipedia': True, 'duckduckgo': True, 'custom_urls': []}
    use_wiki    = research_config.get('wikipedia', True)
    use_ddg     = research_config.get('duckduckgo', True)
    custom_urls = [u.strip() for u in research_config.get('custom_urls', []) if u.strip()]

    search_topic = None
    if use_wiki or use_ddg:
        search_topic = _extract_search_topic(query_text, api_key, llm_config)
        print(f"\n[Module 4] Search topic: '{search_topic}'")

    use_web_search    = research_config.get('web_search', False)
    use_google        = research_config.get('google', False)
    google_api_key    = research_config.get('google_api_key','')
    google_cx         = research_config.get('google_cx','')

    results = []
    tasks   = {}
    with ThreadPoolExecutor(max_workers=max(2, len(custom_urls)+3)) as pool:
        if use_wiki and search_topic:
            print(f"[Module 4] Wikipedia search: '{search_topic}'")
            tasks['wiki'] = pool.submit(_wiki_fetch, search_topic, query_text, api_key, llm_config)
        if use_ddg and search_topic:
            print(f"[Module 4] DuckDuckGo search: '{search_topic}'")
            tasks['ddg'] = pool.submit(_ddg_fetch, search_topic)
        if use_web_search and search_topic:
            print(f'[Module 4] Unrestricted web search: \'{search_topic}\'')
            tasks['web'] = pool.submit(_web_search_fetch, search_topic, api_key, llm_config)

        if use_google and google_api_key and google_cx and search_topic:
            print(f'[Module 4] Google Search: \'{search_topic}\'')
            tasks['google'] = pool.submit(
                _google_search_fetch, search_topic,
                google_api_key, google_cx, llm_config
            )

        for i, url in enumerate(custom_urls):
            print(f'[Module 4] Custom URL {i+1}: {url[:80]}')
            tasks[f'url_{i}'] = pool.submit(_url_fetch, url, query_text, api_key, llm_config)

        wiki_result = None
        if 'wiki' in tasks:
            t0 = time.perf_counter()
            wiki_result = tasks['wiki'].result()
            print(f'   [M4] Wikipedia fetched in {time.perf_counter()-t0:.2f}s')
            if _is_valid_source(wiki_result):
                wiki_result['source_name'] = 'Wikipedia'
                results.append(wiki_result)
            else:
                print(f'   [M4] Wikipedia: no usable result — {wiki_result.get("context","")[:80]}')

        if 'ddg' in tasks:
            t0 = time.perf_counter()
            ddg_result = tasks['ddg'].result()
            print(f'   [M4] DuckDuckGo fetched in {time.perf_counter()-t0:.2f}s')
            if _is_valid_source(ddg_result):
                if wiki_result is None or not _same_topic(ddg_result, wiki_result):
                    ddg_result['source_name'] = 'DuckDuckGo'
                    results.append(ddg_result)
                else:
                    print('   [M4] DuckDuckGo: same topic as Wikipedia -- skipping.')
            else:
                print(f'   [M4] DuckDuckGo: no usable result — {ddg_result.get("context","")[:80]}')

        for i in range(len(custom_urls)):
            key_name = f'url_{i}'
            if key_name in tasks:
                t0 = time.perf_counter()
                url_result = tasks[key_name].result()
                print(f'   [M4] Custom URL {i+1} fetched in {time.perf_counter()-t0:.2f}s')
                if _is_valid_source(url_result):
                    url_result['source_name'] = 'Custom URL'
                    results.append(url_result)
                else:
                    print(f"   [M4] Custom URL {i+1} failed: {url_result.get('context','')[:80]}")

        if 'web' in tasks:
            t0 = time.perf_counter()
            web_results = tasks['web'].result()
            print(f'   [M4] Web search completed in {time.perf_counter()-t0:.2f}s')
            for wr in web_results:
                if _is_valid_source(wr):
                    results.append(wr)
                else:
                    print(f"   [M4] Web result filtered: {wr.get('context','')[:80]}")

        if 'google' in tasks:
            t0 = time.perf_counter()
            google_results = tasks['google'].result()
            print(f'   [M4] Google Search completed in {time.perf_counter()-t0:.2f}s')
            for gr in google_results:
                if _is_valid_source(gr):
                    results.append(gr)
                else:
                    print(f"   [M4] Google result filtered: {gr.get('context','')[:80]}")
    return results


def _wiki_fetch(search_query, original_query, api_key=None, llm_config=None):
    try:
        candidates = wikipedia.search(search_query, results=5)
        if not candidates:
            return _failed('No Wikipedia results found.')
        for title in candidates:
            try:
                if not _is_relevant(title, search_query):
                    print(f"   [M4] Wikipedia: skipping '{title}' (word filter)")
                    continue
                page    = wikipedia.page(title, auto_suggest=False)
                summary = wikipedia.summary(title, sentences=5, auto_suggest=False)
                if api_key and not _llm_domain_check(page.title, summary, search_query, api_key, llm_config):
                    print(f"   [M4] Wikipedia: LLM rejected '{page.title}'")
                    continue
                print(f'   \u2705 Wikipedia: {page.url}')
                return {'context': summary, 'reference': page.url, 'title': page.title}
            except wikipedia.exceptions.DisambiguationError as e:
                try:
                    page    = wikipedia.page(e.options[0], auto_suggest=False)
                    summary = wikipedia.summary(e.options[0], sentences=5, auto_suggest=False)
                    if api_key and not _llm_domain_check(page.title, summary, search_query, api_key, llm_config):
                        continue
                    print(f'   \u2705 Wikipedia: {page.url}')
                    return {'context': summary, 'reference': page.url, 'title': page.title}
                except Exception:
                    continue
            except Exception:
                continue
        return _failed('No relevant Wikipedia results found.')
    except Exception as e:
        return _failed(f'Wikipedia search failed: {e}')


def _ddg_fetch(search_query):
    """
    DuckDuckGo source.  Two-tier:
    1. Instant Answer API (api.duckduckgo.com) — fast, free, works for famous topics.
    2. DDGS full web search fallback — if Instant Answer returns nothing, tries a
       real DDGS search and returns the first usable snippet.  This means the
       DuckDuckGo (instant) checkbox now reliably returns something for any query.
    """
    try:
        encoded = urllib.parse.quote_plus(search_query)
        url     = (f'https://api.duckduckgo.com/?q={encoded}'
                   f'&format=json&no_redirect=1&no_html=1&skip_disambig=1')
        req = urllib.request.Request(url, headers={'User-Agent': 'NeurosymbolicAI/1.0'})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        abstract = data.get('AbstractText','').strip()
        abs_url  = data.get('AbstractURL','').strip()
        abs_src  = data.get('AbstractSource','DuckDuckGo')
        if abstract and len(abstract) > 80:
            ref = abs_url or f'https://duckduckgo.com/?q={encoded}'
            print(f'   \u2705 DuckDuckGo instant ({abs_src}): {abs_url or "N/A"}')
            return {'context': abstract, 'reference': ref,
                    'title': data.get('Heading', search_query),
                    'source_name': 'DuckDuckGo'}
        snippets = []
        for item in data.get('RelatedTopics', []):
            if isinstance(item, dict):
                t = item.get('Text','').strip()
                if t and len(t) > 40: snippets.append(t)
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, dict):
                        t = sub.get('Text','').strip()
                        if t and len(t) > 40: snippets.append(t)
        if snippets:
            print('   \u2705 DuckDuckGo instant: related topics fallback')
            return {'context': ' '.join(snippets[:4]),
                    'reference': f'https://duckduckgo.com/?q={encoded}',
                    'title': search_query,
                    'source_name': 'DuckDuckGo'}
    except Exception as e:
        print(f'   [M4] DuckDuckGo instant API failed: {e}')

    # ── Tier 2: DDGS full web search fallback ────────────────────────────────
    # The instant API only works for famous topics. For everything else, fall
    # back to a real DDGS search and return the first good result as a snippet.
    print(f'   [M4] DuckDuckGo instant returned nothing — trying DDGS fallback...')
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            hits = list(ddgs.text(search_query, max_results=5))
        for hit in hits:
            body = hit.get('body', '').strip()
            href = hit.get('href', '')
            title = hit.get('title', search_query)
            if body and len(body) >= 80:
                print(f'   \u2705 DuckDuckGo DDGS fallback: {href[:60]}')
                return {'context': body[:4000],
                        'reference': href,
                        'title': title,
                        'source_name': 'DuckDuckGo'}
    except ImportError:
        pass  # duckduckgo-search not installed — that's fine
    except Exception as e:
        print(f'   [M4] DuckDuckGo DDGS fallback failed: {e}')

    return _failed('DuckDuckGo returned no usable content.')


def _url_fetch(url, original_query, api_key=None, llm_config=None):
    '''Fetch a custom URL (HTML page or online PDF) and extract text.'''
    # ── Block video platforms early — they're JS-rendered, no text content ──
    try:
        _parsed_host = urllib.parse.urlparse(url).netloc.lower().lstrip("www.")
    except Exception:
        _parsed_host = ""

    if any(_parsed_host == d or _parsed_host.endswith("." + d)
           for d in _VIDEO_DOMAINS):
        return _failed(
            f"VIDEO URL BLOCKED: '{url}' is a video platform. "
            f"Video pages contain no extractable text — the system cannot "
            f"watch or transcribe videos. To use this content: paste the "
            f"video transcript or description directly into the Reference "
            f"Document field instead."
        )

    if any(_parsed_host == d or _parsed_host.endswith("." + d)
           for d in _JS_ONLY_DOMAINS):
        return _failed(
            f"JS-ONLY URL BLOCKED: '{url}' requires JavaScript or login "
            f"to render content and cannot be scraped. Copy-paste the "
            f"relevant text directly into the Reference Document field."
        )

    try:
        req = urllib.request.Request(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0 (compatible; NeurosymbolicAI/1.0)',
                'Accept': 'text/html,application/xhtml+xml,application/pdf,*/*',
            }
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            content_type = resp.headers.get('Content-Type','').lower()
            raw_bytes    = resp.read()
        is_pdf = ('application/pdf' in content_type or url.lower().endswith('.pdf'))
        if is_pdf:
            text = _extract_pdf_text(raw_bytes)
            if not text.strip():
                return _failed(f'PDF at {url} -- could not extract text (may be scanned).')
            title   = _title_from_url(url)
            context = text[:6000]
            print(f'   \u2705 Custom URL (PDF): {url[:60]} -- {len(text):,} chars')
            return {'context': context, 'reference': url, 'title': title, 'is_pdf': True}
        html  = raw_bytes.decode('utf-8', errors='replace')
        text  = _strip_html(html)
        if len(text) < 100:
            return _failed(f'URL {url} -- too little text (may require JS).')
        title   = _extract_html_title(html) or _title_from_url(url)
        context = text[:6000]
        print(f'   \u2705 Custom URL (HTML): {url[:60]} -- {len(text):,} chars extracted')
        return {'context': context, 'reference': url, 'title': title}
    except urllib.error.HTTPError as e:
        return _failed(f'HTTP {e.code} fetching {url}: {e.reason}')
    except urllib.error.URLError as e:
        return _failed(f'URL error fetching {url}: {e.reason}')
    except Exception as e:
        return _failed(f'Failed to fetch {url}: {type(e).__name__}: {e}')


def _extract_pdf_text(raw_bytes):
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
        pages  = [page.extract_text() or '' for page in reader.pages]
        return '\n\n'.join(p for p in pages if p.strip())
    except ImportError:
        pass
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            return '\n\n'.join(p.extract_text() or '' for p in pdf.pages)
    except Exception:
        return ''


def _strip_html(html):
    html = re.sub(r'<(script|style)[^>]*>.*?</(script|style)>', ' ', html,
                  flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', html)
    for ent, ch in [('&amp;','&'),('&lt;','<'),('&gt;','>'),
                    ('&nbsp;',' '),('&quot;','"'),('&#39;',"'")]:
        text = text.replace(ent, ch)
    text = re.sub(r'[\t ]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _extract_html_title(html):
    m = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
    return m.group(1).strip() if m else ''


def _title_from_url(url):
    path = urllib.parse.urlparse(url).path
    name = path.rstrip('/').split('/')[-1]
    name = re.sub(r'[-_]', ' ', name)
    name = re.sub(r'\.\w+$', '', name)
    return name.title() if name else url[:50]


def _web_search_fetch(query_text, api_key=None, llm_config=None,
                       max_results=5, max_chars_per_source=4000):
    """
    Unrestricted web search using duckduckgo_search (DDGS).
    Fetches top N search results, extracts text from each page,
    filters to trusted/relevant sources, returns list of source dicts.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return [_failed("duckduckgo-search not installed. Run: pip install duckduckgo-search")]

    topic = _extract_search_topic(query_text, api_key, llm_config)
    print(f"   [M4] Web search (unrestricted): '{topic}'")

    # ── Step 1: Get search results ────────────────────────────────────────────
    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(topic, max_results=max_results * 2))
    except Exception as e:
        return [_failed(f"Web search failed: {e}")]

    if not raw_results:
        return [_failed("Web search returned no results.")]

    # ── Step 2: Score and filter results ─────────────────────────────────────
    # Prefer authoritative domains; skip low-quality sources
    LOW_QUALITY = {
        "pinterest", "instagram", "twitter", "x.com", "facebook",
        "tiktok", "reddit", "quora", "yahoo answers", "answers.com",
        "ask.com", "yelp", "tripadvisor", "amazon", "ebay", "etsy",
    }
    TRUSTED_BOOST = {
        "wikipedia", "britannica", "gov", "edu", "ieee", "nature.com",
        "pubmed", "ncbi", "nih", "who.int", "un.org", "arxiv", "springer",
        "elsevier", "bbc", "reuters", "ap.org", "nytimes", "theguardian",
    }

    def _score_result(r):
        url = r.get("href","").lower()
        if any(lq in url for lq in LOW_QUALITY):
            return -1
        score = 0
        if any(tr in url for tr in TRUSTED_BOOST):
            score += 2
        body = r.get("body","")
        if len(body) > 200: score += 1
        if len(body) > 500: score += 1
        return score

    filtered = [r for r in raw_results if _score_result(r) >= 0]
    filtered.sort(key=_score_result, reverse=True)
    top = filtered[:max_results]

    # ── Step 3: Fetch full page content for each result ───────────────────────
    sources = []
    fetch_tasks = {}
    with ThreadPoolExecutor(max_workers=min(5, len(top))) as pool:
        for i, result in enumerate(top):
            url = result.get("href","")
            if url:
                fetch_tasks[i] = pool.submit(
                    _url_fetch, url, query_text, api_key, llm_config
                )

        for i, result in enumerate(top):
            url    = result.get("href","")
            title  = result.get("title", _title_from_url(url))
            body   = result.get("body","")

            context = None
            if i in fetch_tasks:
                fetched = fetch_tasks[i].result()
                # Bug fix: use _is_valid_source() (the authoritative filter) instead
                # of the partial _FAILED_CONTEXTS set.  The set only covers 3 exact
                # strings — dynamic error messages like "HTTP 404 fetching ..." or
                # "Failed to fetch ...: ConnectionError" pass through as valid content
                # and end up injected into the generation prompt.
                if _is_valid_source(fetched):
                    context = fetched["context"][:max_chars_per_source]

            # Fall back to the search snippet if fetch failed or was filtered
            if not context and body:
                context = body[:max_chars_per_source]

            if context:
                print(f"   [M4] Web result {i+1}: {url[:70]}")
                sources.append({
                    "context"    : context,
                    "reference"  : url,
                    "title"      : title,
                    "source_name": "Web Search",
                })

    if not sources:
        return [_failed("Web search: could not extract content from any result.")]

    print(f"   [M4] Web search: {len(sources)} source(s) retrieved")
    return sources


def _google_search_fetch(query_text, api_key_google, cx,
                          llm_config=None, max_results=5, max_chars=4000):
    """
    Google Custom Search JSON API — fetches top N results and reads each page.
    api_key_google: Google API key (AIza...)
    cx: Custom Search Engine ID
    """
    import urllib.parse, urllib.request, json

    topic = _extract_search_topic(query_text, None, llm_config)
    print(f"   [M4] Google Search: '{topic}'")

    try:
        encoded = urllib.parse.quote_plus(topic)
        url = (
            f"https://www.googleapis.com/customsearch/v1"
            f"?key={api_key_google}&cx={cx}&q={encoded}&num={max_results}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "NeurosymbolicAI/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return [_failed(f"Google Search API failed: {e}")]

    items = data.get("items", [])
    if not items:
        return [_failed("Google Search returned no results.")]

    sources = []
    tasks = {}
    with ThreadPoolExecutor(max_workers=min(5, len(items))) as pool:
        for i, item in enumerate(items):
            link = item.get("link","")
            if link:
                tasks[i] = pool.submit(_url_fetch, link, query_text, None, llm_config)

        for i, item in enumerate(items):
            link    = item.get("link","")
            title   = item.get("title","")
            snippet = item.get("snippet","")

            context = None
            if i in tasks:
                fetched = tasks[i].result()
                # Bug fix: use _is_valid_source() — same reason as _web_search_fetch.
                if _is_valid_source(fetched):
                    context = fetched["context"][:max_chars]
            if not context and snippet:
                context = snippet[:max_chars]

            if context:
                print(f"   [M4] Google result {i+1}: {link[:70]}")
                sources.append({
                    "context"    : context,
                    "reference"  : link,
                    "title"      : title,
                    "source_name": "Google Search",
                })

    if not sources:
        return [_failed("Google Search: could not extract content from any result.")]
    print(f"   [M4] Google Search: {len(sources)} source(s) retrieved")
    return sources


def _failed(reason):
    return {'context': reason, 'reference': 'None', 'title': 'None'}


def _extract_search_topic(query_text, api_key=None, llm_config=None):
    cache_key = hashlib.md5(query_text.strip().lower().encode()).hexdigest()
    if cache_key in _TOPIC_CACHE:
        return _TOPIC_CACHE[cache_key]
    if api_key:
        try:
            prompt = (
                'Extract the core subject as a short search query (3-5 words max). '
                'Focus on TOPIC only.\n'
                'Return ONLY the search query, no explanation, no quotes.\n\n'
                f'Request: "{query_text}"\n\nSearch query:'
            )
            result = _call_llm(prompt, api_key, llm_config, max_tokens=32)
            result = result.replace('"','').replace("'",'').strip()[:60]
            _TOPIC_CACHE[cache_key] = result
            return result
        except Exception:
            pass
    result = _simple_clean(query_text)
    _TOPIC_CACHE[cache_key] = result
    return result


def _simple_clean(q):
    filler = ['i want to','i want you to','i need to','help me','can you',
              'please','create','make','build','generate','write','give me']
    q = q.lower().strip()
    for p in filler: q = q.replace(p, ' ')
    return ' '.join(w for w in q.split() if len(w) > 2)[:60]


def _llm_domain_check(title, summary, query, api_key, llm_config=None):
    try:
        prompt = (
            f'Wikipedia article: "{title}"\n'
            f'User query topic: "{query}"\n'
            f'Article summary: "{summary[:300]}"\n\n'
            'Is this article from a CLEARLY DIFFERENT domain than the query? '
            'Only answer "yes" if completely unrelated, "no" otherwise.\n'
            'Answer ONLY "yes" or "no".'
        )
        answer = _call_llm(prompt, api_key, llm_config, max_tokens=10)
        return not answer.strip().lower().startswith('yes')
    except Exception:
        return True


def _is_relevant(title, query):
    stop = {'the','a','an','of','in','on','at','to','for','and','or',
            'with','by','from','as','is','are','its','it','that','this'}
    def tok(text):
        return [t for t in re.split(r'[\s\-_/]+', text.lower())
                if t not in stop and len(t) > 2]
    qt = set(tok(query)); tt = set(tok(title))
    if not qt: return True
    if qt & tt: return True
    for qw in qt:
        for tw in tt:
            if len(qw)>=4 and len(tw)>=4:
                if tw.startswith(qw) or qw.startswith(tw): return True
    return False


def _same_topic(a, b):
    stop = {'the','a','an','of','in','on','at','to','for','and'}
    wa = {w for w in a.get('title','').lower().split() if w not in stop and len(w)>2}
    wb = {w for w in b.get('title','').lower().split() if w not in stop and len(w)>2}
    return len(wa & wb) >= 2


def wiki_search_fallback(query_text, api_key=None, llm_config=None):
    topic = _extract_search_topic(query_text, api_key, llm_config)
    return _wiki_fetch(topic, query_text, api_key, llm_config)

def duckduckgo_search(query_text, api_key=None):
    topic = _extract_search_topic(query_text, api_key)
    return _ddg_fetch(topic)
