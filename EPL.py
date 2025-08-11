#!/usr/bin/env python3
"""
EPL resilient scraper — FBref, Understat (improved, polite)

- Robust FBref schedule parsing (extracts match URLs where present)
- Optional best-effort enrichment from match pages (score, xG) — OFF by default
- Robust Understat JSON extraction with multiple fallback regexes
- Polite HTTP client with exponential backoff, jitter, Retry-After handling
- Caches pages under ./cache and writes CSVs to ./data

Usage:
    python EPL.py --seasons 2022-23 2023-24 --only all
    python EPL.py --seasons 2022-23 --only fbref --enrich --max-enrich 10 --rate-limit 4.0
"""
import argparse
import logging
import re
import time
import json
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin
from datetime import datetime, date
from io import StringIO

import requests
from bs4 import BeautifulSoup
import pandas as pd

# Optional understat + aiohttp path (if installed)
try:
    from understat import Understat
    import aiohttp
    HAS_UNDERSTAT_PKG = True
    HAS_AIOHTTP = True
except Exception:
    HAS_UNDERSTAT_PKG = False
    HAS_AIOHTTP = False

# selenium optional for oddsportal fallback (unchanged)
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    HAS_SELENIUM = True
except Exception:
    HAS_SELENIUM = False

import asyncio

# --- config ---
BASE_DIR = Path.cwd()
CACHE_DIR = BASE_DIR / "cache"
DATA_DIR = BASE_DIR / "data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; EPL-Scraper/1.0; +https://example.com)"}
DEFAULT_RATE_LIMIT = 3.0   # seconds between requests (increase to be polite)
DEFAULT_MAX_RETRIES = 5

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("epl_scraper")


def save_cache(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def load_cache(path: Path) -> Optional[bytes]:
    if path.exists():
        return path.read_bytes()
    return None


def _safe_cache_name_for_url(url: str) -> str:
    """Return a short hashed filename for any URL (safe for file systems)."""
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return f"url-{h}.html"


def request_get(url: str,
                cache_path: Optional[Path] = None,
                force: bool = False,
                rate_limit: float = DEFAULT_RATE_LIMIT,
                max_retries: int = DEFAULT_MAX_RETRIES) -> Optional[bytes]:
    """
    HTTP GET with caching, exponential backoff on 429, jitter, and Retry-After support.
    """
    if cache_path and not force:
        cached = load_cache(cache_path)
        if cached:
            logger.debug(f"Loaded from cache: {cache_path}")
            return cached

    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            logger.info(f"GET {url} (attempt {attempt})")
            resp = requests.get(url, headers=HEADERS, timeout=20)
            if resp.status_code == 200:
                content = resp.content
                if cache_path:
                    save_cache(cache_path, content)
                # polite small sleep after success
                time.sleep(rate_limit + random.uniform(0, 0.5))
                return content

            if resp.status_code == 429:
                retry_after = None
                if 'Retry-After' in resp.headers:
                    try:
                        retry_after = int(resp.headers['Retry-After'])
                    except Exception:
                        # could be HTTP-date string; ignore for simplicity
                        retry_after = None
                backoff = (2 ** (attempt - 1)) * rate_limit
                if retry_after:
                    backoff = max(backoff, retry_after)
                jitter = random.uniform(0, backoff * 0.2)
                wait = backoff + jitter
                logger.warning(f"Status 429 for {url} — backing off {wait:.1f}s (attempt {attempt})")
                time.sleep(wait)
                continue

            # other non-200: log and exponential backoff
            logger.warning(f"Status {resp.status_code} for {url}")
            backoff = (2 ** (attempt - 1)) * rate_limit
            time.sleep(backoff + random.uniform(0, 0.5))
        except Exception as e:
            logger.warning(f"Request error {e} on attempt {attempt} for {url}")
            backoff = (2 ** (attempt - 1)) * rate_limit
            time.sleep(backoff + random.uniform(0, 0.5))

    logger.error(f"Failed to fetch {url} after {max_retries} attempts")
    return None


# ---------------- FBref helpers ----------------
def fbref_season_url(season: str) -> str:
    # FBref expects a season slug like "2022-2023" or "2022-23".
    return f"https://fbref.com/en/comps/9/schedule/{season}/schedule"


def _flatten_columns(cols):
    if isinstance(cols, pd.MultiIndex):
        new = []
        for c in cols:
            parts = [str(x).strip() for x in c if str(x).strip() and str(x) != 'nan']
            new.append(" ".join(parts).strip())
        return new
    return [str(c).strip() for c in cols]


def _is_schedule_table(df: pd.DataFrame) -> bool:
    cols = [c.lower() for c in df.columns]
    has_home = any('home' in c for c in cols)
    has_away = any('away' in c for c in cols)
    has_date = any('date' in c for c in cols)
    has_score = any('score' in c or 'result' in c for c in cols)
    return (has_home and has_away) or (has_date and (has_score))


def find_schedule_table(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    # try id/caption heuristics
    for t in soup.find_all("table"):
        tid = (t.get("id") or "").lower()
        caption = t.find("caption").get_text(strip=True).lower() if t.find("caption") else ""
        if 'sched' in tid or 'schedule' in caption or 'fixtures' in caption:
            return t
    # fallback: heuristic over tables
    for t in soup.find_all("table"):
        try:
            df_try = pd.read_html(StringIO(str(t)))[0]
            df_try.columns = _flatten_columns(df_try.columns)
            if _is_schedule_table(df_try):
                return t
        except Exception:
            continue
    return None


def extract_match_links_from_table(table_soup: BeautifulSoup, base_url="https://fbref.com") -> Dict[int, str]:
    """
    Parses anchor tags in the schedule table to find match detail URLs.
    Returns mapping row_index -> absolute_url (row index based on order of <tr> inside table body).
    """
    links = {}
    tbody = table_soup.find("tbody")
    if not tbody:
        return links
    rows = tbody.find_all("tr", recursive=False)
    for i, row in enumerate(rows):
        # find first anchor that likely points to a match page
        a = row.find("a", href=True)
        if a:
            href = a['href']
            full = urljoin(base_url, href)
            links[i] = full
    return links


def parse_fbref_match_page_for_stats(html: bytes) -> Dict[str, Optional[str]]:
    """
    Best-effort extraction of Score and xG from a FBref match page.
    Returns dict with keys: score (e.g. '2-1'), home_xg, away_xg (strings or None).
    """
    out = {'score': None, 'home_xg': None, 'away_xg': None}
    soup = BeautifulSoup(html, "lxml")

    try:
        sb = soup.find(lambda tag: tag.name in ['div', 'h1'] and tag.get_text() and re.search(r'\d+\s*[-–]\s*\d+', tag.get_text()))
        if sb:
            m = re.search(r'(\d+)\s*[-–]\s*(\d+)', sb.get_text())
            if m:
                out['score'] = f"{m.group(1)}-{m.group(2)}"
    except Exception:
        pass

    try:
        text = soup.get_text(separator=' ')
        xg_matches = re.findall(r'xG[:\s]*([0-9]*\.?[0-9]+)', text, flags=re.IGNORECASE)
        if len(xg_matches) >= 2:
            out['home_xg'], out['away_xg'] = xg_matches[0], xg_matches[1]
        else:
            for tbl in soup.find_all('table'):
                try:
                    df_tbl = pd.read_html(StringIO(str(tbl)))[0]
                    cols = [str(c).lower() for c in df_tbl.columns]
                    if any('xg' in c for c in cols):
                        flat = df_tbl.astype(str)
                        nums = re.findall(r'([0-9]*\.?[0-9]+)', flat.to_string())
                        if len(nums) >= 2:
                            out['home_xg'], out['away_xg'] = nums[0], nums[1]
                            break
                except Exception:
                    continue
    except Exception:
        pass

    return out


def normalize_schedule_df(df: pd.DataFrame, table_soup: BeautifulSoup) -> pd.DataFrame:
    df.columns = _flatten_columns(df.columns)
    # candidates for Date/Home/Away
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    home_col = next((c for c in df.columns if c.lower() == 'home' or 'home' in c.lower()), None)
    away_col = next((c for c in df.columns if c.lower() == 'away' or 'away' in c.lower()), None)

    if not home_col:
        possible_home = [c for c in df.columns if any(k in c.lower() for k in ['home', 'team'])]
        if possible_home:
            home_col = possible_home[0]
    if not away_col:
        possible_away = [c for c in df.columns if any(k in c.lower() for k in ['away', 'opponent', 'opp', 'visitor'])]
        if possible_away:
            away_col = possible_away[0]

    key_cols = [c for c in (home_col, away_col) if c]

    if date_col:
        df = df[(df[date_col].notna()) | (df[key_cols].notna().any(axis=1))]
    elif key_cols:
        df = df[df[key_cols].notna().any(axis=1)]
    else:
        df = df.dropna(how='all')

    # attach match link column if we can map anchors from table_soup to rows
    links = extract_match_links_from_table(table_soup)
    if links:
        link_list = []
        tbody = table_soup.find("tbody")
        rows = tbody.find_all("tr", recursive=False) if tbody else []
        # align by sequence (best-effort)
        for idx in range(len(df)):
            link = links.get(idx)
            link_list.append(link if link else None)
        df['match_url'] = link_list[:len(df)]
    else:
        df['match_url'] = [None] * len(df)

    return df


def parse_fbref_schedule(html: bytes) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    table = find_schedule_table(soup)
    if table is None:
        # try comments (some FBref pages hide tables in comments)
        comments = soup.find_all(string=lambda text: isinstance(text, str) and "<table" in text)
        for c in comments:
            try:
                soup2 = BeautifulSoup(c, "lxml")
                table = find_schedule_table(soup2)
                if table:
                    break
            except Exception:
                continue
    if table is None:
        raise RuntimeError("Could not find schedule table on FBref page")

    df = pd.read_html(StringIO(str(table)))[0]
    df = normalize_schedule_df(df, table)

    logger.info(f"FBref parsed schedule columns: {list(df.columns)[:12]}")
    return df


def _parse_match_date(date_str: Any) -> Optional[date]:
    """
    Parse FBref date strings like '8/15/2025', '15 Aug 2025', '2025-08-15' to datetime.date.
    """
    if pd.isna(date_str):
        return None
    s = str(date_str).strip()
    for fmt in ("%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            continue
    try:
        dt = pd.to_datetime(s, dayfirst=False, errors='coerce')
        if pd.isna(dt):
            return None
        return dt.date()
    except Exception:
        return None


def enrich_matches_best_effort(df: pd.DataFrame,
                               enrich: bool = False,
                               max_enrich: int = 25,
                               rate_limit: float = DEFAULT_RATE_LIMIT) -> pd.DataFrame:
    """
    Enrich df in-place by fetching match pages for missing score/xG.
      - enrich: must be True to perform network calls
      - max_enrich: hard cap to avoid lots of requests
    """
    if not enrich:
        logger.info("Match-page enrichment disabled (use --enrich to enable).")
        return df

    score_col = next((c for c in df.columns if 'score' in c.lower()), None)
    xg_cols = [c for c in df.columns if c.lower().startswith('xg') or 'xg' in c.lower()]

    to_enrich = []
    today = datetime.now().date()

    date_col = next((c for c in df.columns if 'date' in c.lower()), None)

    for i, row in df.iterrows():
        url = row.get('match_url')
        if not url:
            continue
        # parse date; skip future matches
        parsed_date = _parse_match_date(row.get(date_col)) if date_col else None
        if parsed_date is None:
            # conservative: skip if date can't be parsed
            continue
        if parsed_date > today:
            continue

        score_missing = (not score_col) or pd.isna(row.get(score_col))
        xg_missing = (len(xg_cols) == 0) or all(pd.isna(row.get(c)) for c in xg_cols)
        if score_missing or xg_missing:
            to_enrich.append((i, url))

    if not to_enrich:
        logger.info("No past matches need enrichment.")
        return df

    logger.info(f"Attempting to enrich up to {min(len(to_enrich), max_enrich)} past matches (cached pages will be used when available).")

    enriched = 0
    consecutive_failures = 0

    for i, (idx, url) in enumerate(to_enrich):
        if enriched >= max_enrich:
            break

        logger.info(f"Enriching row {idx} -> {url} (#{enriched+1})")
        cache_file = CACHE_DIR / "fbref" / _safe_cache_name_for_url(url)
        html = request_get(url, cache_path=cache_file, force=False, rate_limit=rate_limit, max_retries=DEFAULT_MAX_RETRIES)
        if not html:
            consecutive_failures += 1
            logger.warning(f"Failed to fetch {url} — skipping. consecutive_failures={consecutive_failures}")
            if consecutive_failures >= 3:
                logger.error("Multiple consecutive fetch failures — aborting enrichment to avoid rate limiting.")
                break
            continue

        consecutive_failures = 0  # reset on success
        stats = parse_fbref_match_page_for_stats(html)

        if stats.get('score'):
            score_col = score_col or 'Score'
            df.at[idx, score_col] = stats['score']

        if stats.get('home_xg') and stats.get('away_xg'):
            if len(xg_cols) >= 2:
                df.at[idx, xg_cols[0]] = stats['home_xg']
                df.at[idx, xg_cols[1]] = stats['away_xg']
            else:
                df.at[idx, 'home_xG_fetched'] = stats['home_xg']
                df.at[idx, 'away_xG_fetched'] = stats['away_xg']

        enriched += 1
        time.sleep(rate_limit + random.uniform(0.5, 1.0))

    logger.info(f"Enrichment complete — enriched {enriched} matches (max {max_enrich}).")
    return df


def fetch_fbref_for_season(season: str, force: bool = False, enrich: bool = False, max_enrich: int = 25, rate_limit: float = DEFAULT_RATE_LIMIT) -> pd.DataFrame:
    url = fbref_season_url(season)
    cache_path = CACHE_DIR / "fbref" / f"schedule_{season}.html"
    content = request_get(url, cache_path=cache_path, force=force, rate_limit=rate_limit)
    if not content:
        raise RuntimeError("Failed to fetch FBref schedule")
    df = parse_fbref_schedule(content)
    # optionally enrich (best-effort)
    df = enrich_matches_best_effort(df, enrich=enrich, max_enrich=max_enrich, rate_limit=rate_limit)
    out_csv = DATA_DIR / f"fbref_schedule_{season}.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved FBref schedule -> {out_csv}")
    return df


# ---------------- Understat helpers (robust extraction) ----------------
def _extract_json_from_text_variants(text: str) -> Optional[Any]:
    """
    Try multiple regex patterns to extract JSON payloads often embedded in Understat pages.
    Return parsed object (list/dict) or None.
    """
    patterns = [
        r"var matchesData = JSON\.parse\('(?P<data>.+?)'\)",
        r"var\s+shotsData\s*=\s*JSON\.parse\('(?P<data>.+?)'\)",
        r"JSON\.parse\('(?P<data>\\\[\\{.+?\\}\\\\\])'\)",
        r"(\[\{.+?\}\])",
        r"data:\s*(\[\{.+?\}\])",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.DOTALL)
        if m:
            grp = m.groupdict().get('data') or (m.group(1) if m.groups() else None)
            if not grp:
                continue
            try:
                candidate = bytes(grp, 'utf-8').decode('unicode_escape')
                parsed = json.loads(candidate)
                return parsed
            except Exception:
                try:
                    parsed = json.loads(grp)
                    return parsed
                except Exception:
                    continue

    arrays = re.findall(r"(\[\{.+?\}\])", text, flags=re.DOTALL)
    if arrays:
        arrays.sort(key=lambda s: len(s), reverse=True)
        for arr in arrays:
            try:
                parsed = json.loads(arr)
                return parsed
            except Exception:
                try:
                    parsed = json.loads(bytes(arr, 'utf-8').decode('unicode_escape'))
                    return parsed
                except Exception:
                    continue
    return None


def understat_fetch_season_via_page(season: str, force: bool = False) -> pd.DataFrame:
    if '-' in season:
        end_year = season.split('-')[1]
        if len(end_year) == 2:
            end_year = '20' + end_year
    else:
        end_year = season
    url = f"https://understat.com/league/EPL/{end_year}"
    cache_path = CACHE_DIR / "understat" / f"league_EPL_{end_year}.html"
    content = request_get(url, cache_path=cache_path, force=force)
    if not content:
        raise RuntimeError("Failed to fetch Understat page")
    text = content.decode('utf-8', errors='ignore')

    parsed = _extract_json_from_text_variants(text)
    if parsed is None:
        raise RuntimeError("Couldn't locate Understat JSON payload (tried multiple patterns)")
    df = pd.json_normalize(parsed)
    out_csv = DATA_DIR / f"understat_EPL_{end_year}.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved Understat -> {out_csv}")
    return df


async def understat_fetch_season_async(season: str) -> List[Dict[str, Any]]:
    if not HAS_UNDERSTAT_PKG or not HAS_AIOHTTP:
        raise RuntimeError("understat + aiohttp required for async fetch")
    if '-' in season:
        end_year = season.split('-')[1]
        if len(end_year) == 2:
            end_year = '20' + end_year
    else:
        end_year = season
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        candidates = ['get_league_matches', 'get_league_fixtures', 'get_league_results']
        for name in candidates:
            if hasattr(understat, name):
                func = getattr(understat, name)
                try:
                    matches = await func('EPL', int(end_year))
                    return matches
                except Exception:
                    try:
                        matches = await func(int(end_year))
                        return matches
                    except Exception:
                        continue
        raise RuntimeError("Understat client has no compatible fetch method")


def fetch_understat_for_season(season: str, force: bool = False) -> pd.DataFrame:
    if HAS_UNDERSTAT_PKG and HAS_AIOHTTP:
        logger.info("Attempting understat async fetch")
        try:
            matches = asyncio.run(understat_fetch_season_async(season))
            df = pd.json_normalize(matches)
            out_csv = DATA_DIR / f"understat_EPL_{season}.csv"
            df.to_csv(out_csv, index=False)
            logger.info(f"Saved Understat -> {out_csv}")
            return df
        except Exception as e:
            logger.warning(f"Understat async failed ({e}), falling back to page-scrape")
            return understat_fetch_season_via_page(season, force=force)
    else:
        logger.info("Understat package not present — using page-scrape fallback")
        return understat_fetch_season_via_page(season, force=force)


# ---------------- Orchestration ----------------
def seasons_from_args(seasons: List[str]) -> List[str]:
    out = []
    for s in seasons:
        if s.isdigit() and len(s) == 4:
            start = int(s) - 1
            out.append(f"{start}-{str(s)[-2:]}")
        elif re.match(r"^\d{4}-\d{2}$", s) or re.match(r"^\d{4}-\d{4}$", s):
            out.append(s)
        else:
            logger.warning(f"Unrecognized season format: {s}")
    return out


def main(args):
    seasons = seasons_from_args(args.seasons)
    if not seasons:
        logger.error("No valid seasons")
        return

    # rate limit override from CLI
    rate_limit = args.rate_limit if hasattr(args, "rate_limit") and args.rate_limit else DEFAULT_RATE_LIMIT

    for season in seasons:
        logger.info(f"Processing season {season}")
        if 'fbref' in args.only or args.only == ['all']:
            fetch_fbref_for_season(season,
                                   force=args.force_refresh,
                                   enrich=args.enrich,
                                   max_enrich=args.max_enrich,
                                   rate_limit=rate_limit)
        if 'understat' in args.only or args.only == ['all']:
            fetch_understat_for_season(season, force=args.force_refresh)
    logger.info("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EPL Scraper: FBref + Understat (polite defaults)')
    parser.add_argument('--seasons', nargs='+', required=True, help='List seasons, e.g. 2022-23 or 2022')
    parser.add_argument('--force-refresh', action='store_true', help='Ignore cache and refetch pages')
    parser.add_argument('--only', nargs='+', default=['all'], help='Which sources to fetch: fbref understat oddsportal or all')
    parser.add_argument('--enrich', action='store_true', help='Attempt to fetch match detail pages to enrich score/xG (use with caution)')
    parser.add_argument('--max-enrich', type=int, default=25, help='Maximum number of match detail pages to fetch when enriching')
    parser.add_argument('--rate-limit', type=float, default=DEFAULT_RATE_LIMIT, help='Per-request delay in seconds (increase if you get 429s)')
    args = parser.parse_args()
    main(args)
