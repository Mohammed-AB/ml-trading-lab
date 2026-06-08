"""Web news fetchers for the Research Agent.

Fetches headlines from Google News (RSS search), ForexLive, FXStreet, and Reuters.
Uses ``requests`` + stdlib XML parsing (no feedparser dependency).
Rate-limits to at most one HTTP request per logical source per 15 minutes
(cached responses reused within the window).
"""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from html import unescape
from typing import Optional
from urllib.parse import quote_plus, urlencode

import requests

_log = logging.getLogger("scalp_mode")

# --- Defaults from research plan ---
DEFAULT_GOOGLE_KEYWORDS = [
    "forex",
    "federal reserve",
    "tariffs",
    "oil price",
    "Iran",
    "Trump trade",
    "USD",
    "central bank",
    "interest rates",
    "geopolitical risk",
]

# Reasonable browser-like UA (many feeds block Python-urllib)
_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ScalpingBotResearch/1.0; "
        "+https://example.local; research-rss-reader)"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}

# RSS feed URLs (may change; fallbacks listed where helpful)
_FOREXLIVE_URL = "https://www.forexlive.com/feed/"
_FXSTREET_URLS = (
    "https://www.fxstreet.com/rss",
    "https://www.fxstreet.com/sitemap/rss/news",
)
# Reuters often returns 401 to automated clients; keep tries, then wire fallbacks.
_REUTERS_URLS = (
    "https://www.reuters.com/business/markets/rss",
    "https://www.reuters.com/world/rss",
    "https://feeds.reuters.com/reuters/topNews",
)
_WIRE_FALLBACKS: tuple[tuple[str, str], ...] = (
    ("bbc_business", "https://feeds.bbci.co.uk/news/business/rss.xml"),
    ("investing_forex", "https://www.investing.com/rss/news_285.rss"),
)

_CACHE_TTL_SEC = 900  # 15 minutes
_cache: dict[str, tuple[float, list[dict]]] = {}


def _cache_get(key: str) -> Optional[list[dict]]:
    entry = _cache.get(key)
    if not entry:
        return None
    ts, data = entry
    if time.monotonic() - ts > _CACHE_TTL_SEC:
        return None
    return data


def _cache_set(key: str, data: list[dict]) -> None:
    _cache[key] = (time.monotonic(), data)


def _strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _clean_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = unescape(s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_rss_items(xml_text: str, source_label: str, max_items: int = 40) -> list[dict]:
    """Parse RSS 2.0 / Atom-ish XML into headline dicts."""
    items: list[dict] = []
    try:
        root = ET.fromstring(xml_text.encode("utf-8", errors="replace"))
    except ET.ParseError as e:
        _log.warning("web_scraper: RSS parse error (%s): %s", source_label, e)
        return []

    # RSS: channel/item; Atom: entry
    for elem in root.iter():
        tag = _strip_ns(elem.tag)
        if tag not in ("item", "entry"):
            continue
        title = None
        link = None
        summary = None
        pub = None
        for child in elem:
            ct = _strip_ns(child.tag)
            text = (child.text or "").strip()
            if ct == "title" and not title:
                title = _clean_text(text)
            elif ct == "link":
                if text:
                    link = text
                elif child.get("href"):
                    link = child.get("href")
            elif ct in ("description", "summary", "content"):
                if not summary:
                    summary = _clean_text(text)
            elif ct in ("published", "updated", "pubDate"):
                pub = text or child.get("href") or ""

        if title:
            items.append(
                {
                    "title": title,
                    "summary": summary or "",
                    "link": link or "",
                    "source": source_label,
                    "published": pub or None,
                }
            )
        if len(items) >= max_items:
            break

    return items


def _http_get(url: str, timeout: float = 20.0) -> Optional[str]:
    try:
        r = requests.get(url, headers=_DEFAULT_HEADERS, timeout=timeout)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "utf-8"
        return r.text
    except requests.RequestException as e:
        _log.warning("web_scraper: request failed %s: %s", url, e)
        return None


def _normalize_title_key(title: str) -> str:
    t = title.lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t[:120]


def deduplicate_headlines(items: list[dict]) -> list[dict]:
    """Drop near-duplicate headlines (normalized title). Keeps first occurrence."""
    seen: set[str] = set()
    out: list[dict] = []
    for it in items:
        key = _normalize_title_key(it.get("title") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def fetch_google_news(
    keywords: Optional[list[str]] = None,
    *,
    max_keywords_per_query: int = 10,
) -> list[dict]:
    """
    Google News search RSS. Combines keywords with OR; may split into multiple
    queries if the list is long (each query is cached separately).
    """
    kws = keywords or DEFAULT_GOOGLE_KEYWORDS
    # Batch keywords to avoid absurdly long URLs
    batches: list[list[str]] = []
    for i in range(0, len(kws), max_keywords_per_query):
        batches.append(kws[i : i + max_keywords_per_query])

    all_items: list[dict] = []
    for bi, batch in enumerate(batches):
        cache_key = f"google_news:{','.join(batch)}"
        cached = _cache_get(cache_key)
        if cached is not None:
            all_items.extend(cached)
            continue

        # OR query for Google News RSS
        q_parts = []
        for kw in batch:
            kw = kw.strip()
            if " " in kw:
                q_parts.append(f'"{kw}"')
            else:
                q_parts.append(kw)
        q = " OR ".join(q_parts)
        params = {"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"}
        url = "https://news.google.com/rss/search?" + urlencode(params)
        text = _http_get(url)
        if not text:
            _cache_set(cache_key, [])
            continue
        items = _parse_rss_items(text, "google_news", max_items=35)
        _cache_set(cache_key, items)
        all_items.extend(items)

    return deduplicate_headlines(all_items)


def fetch_forexlive_rss() -> list[dict]:
    cache_key = "forexlive"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    text = _http_get(_FOREXLIVE_URL)
    if not text:
        _cache_set(cache_key, [])
        return []
    items = _parse_rss_items(text, "forexlive", max_items=40)
    _cache_set(cache_key, items)
    return items


def fetch_fxstreet_rss() -> list[dict]:
    cache_key = "fxstreet"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    for url in _FXSTREET_URLS:
        text = _http_get(url)
        if text and ("<item" in text or "<entry" in text):
            items = _parse_rss_items(text, "fxstreet", max_items=40)
            _cache_set(cache_key, items)
            return items
    _log.warning("web_scraper: FXStreet RSS unreachable, returning empty")
    _cache_set(cache_key, [])
    return []


def fetch_reuters_rss() -> list[dict]:
    """World / business wire headlines. Tries Reuters, then BBC Business, Investing forex."""
    cache_key = "reuters_wire"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    for url in _REUTERS_URLS:
        text = _http_get(url)
        if not text:
            continue
        if "<item" not in text and "<entry" not in text:
            continue
        items = _parse_rss_items(text, "reuters", max_items=40)
        if items:
            _cache_set(cache_key, items)
            return items

    for label, url in _WIRE_FALLBACKS:
        text = _http_get(url)
        if not text or ("<item" not in text and "<entry" not in text):
            continue
        items = _parse_rss_items(text, label, max_items=40)
        if items:
            _log.info(
                "web_scraper: using wire fallback %s (Reuters feeds unavailable)",
                label,
            )
            _cache_set(cache_key, items)
            return items

    _log.warning("web_scraper: all wire RSS sources unreachable, returning empty")
    _cache_set(cache_key, [])
    return []


def gather_market_headlines(
    extra_keywords: Optional[list[str]] = None,
) -> list[dict]:
    """Fetch all configured sources, merge, dedupe. Order: Google, ForexLive, FXStreet, Reuters."""
    keywords = list(DEFAULT_GOOGLE_KEYWORDS)
    if extra_keywords:
        keywords.extend(extra_keywords)

    merged: list[dict] = []
    merged.extend(fetch_google_news(keywords))
    merged.extend(fetch_forexlive_rss())
    merged.extend(fetch_fxstreet_rss())
    merged.extend(fetch_reuters_rss())
    return deduplicate_headlines(merged)


def headlines_to_prompt_block(items: list[dict], max_lines: int = 80) -> str:
    """Format headlines for LLM consumption."""
    lines: list[str] = []
    for i, it in enumerate(items[:max_lines], 1):
        src = it.get("source", "?")
        title = it.get("title", "")
        summ = it.get("summary", "")
        if summ and len(summ) > 220:
            summ = summ[:217] + "..."
        extra = f" — {summ}" if summ else ""
        lines.append(f"{i}. [{src}] {title}{extra}")
    if not lines:
        return "(No headlines retrieved from web sources.)"
    return "\n".join(lines)
