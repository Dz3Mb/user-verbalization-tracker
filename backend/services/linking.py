"""Entity linking service.

Aligns spaCy named entities with knowledge graphs:
- Wikidata (primary, reliable): resolves a surface form to a QID via the
  `wbsearchentities` API, then fetches the English Wikipedia sitelink.
- DBpedia: the resource URI is derived from the Wikidata `enwiki` sitelink
  (http://dbpedia.org/resource/<Article_Title>), which is robust and does not
  depend on the often-unavailable public DBpedia Spotlight service.
- DBpedia Spotlight (optional): full-text annotation, disabled by default.
  Enable via env vars to use a self-hosted (or public, when up) instance.

All network calls are best-effort: any failure degrades gracefully and the
pipeline still returns the local NLP results.

Privacy note: entity linking sends the entity text (and, for Spotlight, the
full transcript) to external services. It is configurable and can be turned
off with ENABLE_ENTITY_LINKING=false to keep processing fully local.
"""

import os
import logging
from urllib.parse import quote

import requests

logger = logging.getLogger("linking")

# Use the OS trust store so requests work behind corporate proxies that inject
# a custom root CA (otherwise certifi-based verification fails).
try:
    import truststore

    truststore.inject_into_ssl()
    _TRUSTSTORE = True
except Exception:  # pragma: no cover - optional dependency
    _TRUSTSTORE = False


def _flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


ENABLE_LINKING = _flag("ENABLE_ENTITY_LINKING", "true")
WIKIDATA_API = os.getenv("WIKIDATA_API_URL", "https://www.wikidata.org/w/api.php")

ENABLE_SPOTLIGHT = _flag("ENABLE_DBPEDIA_SPOTLIGHT", "false")
SPOTLIGHT_URL = os.getenv(
    "DBPEDIA_SPOTLIGHT_URL", "https://api.dbpedia-spotlight.org/en/annotate"
)
SPOTLIGHT_CONFIDENCE = float(os.getenv("DBPEDIA_SPOTLIGHT_CONFIDENCE", "0.5"))

HTTP_TIMEOUT = float(os.getenv("LINKING_HTTP_TIMEOUT", "8"))
DBPEDIA_RESOURCE_BASE = "https://dbpedia.org/resource/"

# spaCy labels worth linking. Includes English (en_core_web_sm: PERSON, ORG,
# GPE...) and French/multilingual (fr_core_news_sm: PER, LOC, ORG, MISC) labels.
LINKABLE_LABELS = {
    # English model
    "PERSON", "ORG", "GPE", "LOC", "NORP", "FAC", "PRODUCT", "EVENT",
    "WORK_OF_ART", "LAW", "LANGUAGE",
    # French / multilingual models
    "PER", "MISC",
}

# Number of Wikidata candidates to consider when type-filtering. Larger values
# improve recall on ambiguous surface forms at a small bandwidth cost.
WIKIDATA_SEARCH_LIMIT = int(os.getenv("WIKIDATA_SEARCH_LIMIT", "10"))

# Map a spaCy NER label to a set of Wikidata "instance of" (P31) class QIDs
# that the candidate must match. This rejects candidates of the wrong type
# (e.g. a Belgian commune named "Hugo" when the label is PERSON).
# Empty set means "no type filter for this label" (fallback to top-1).
SPACY_LABEL_TO_WIKIDATA_CLASSES: dict[str, set[str]] = {
    # People
    "PERSON": {"Q5"},          # human
    "PER":    {"Q5"},

    # Organizations / companies
    "ORG": {
        "Q43229",      # organization
        "Q4830453",    # business
        "Q783794",     # company
        "Q6881511",    # enterprise
        "Q891723",     # public company
        "Q161726",     # multinational corporation
        "Q31855",      # research institute
        "Q3918",       # university
        "Q15911314",   # association
        "Q163740",     # nonprofit organization
        "Q484652",     # international organization
        "Q327333",     # government agency
        "Q7278",       # political party
        "Q4438121",    # sports organization
    },

    # Countries / cities / administrative divisions
    "GPE": {
        "Q6256",       # country
        "Q3624078",    # sovereign state
        "Q515",        # city
        "Q1549591",    # big city
        "Q5119",       # capital
        "Q35657",      # state of the US
        "Q15634554",   # state of the United States (variant)
        "Q56061",      # administrative territorial entity
        "Q5107",       # continent
        "Q484170",     # commune of France
        "Q3957",       # town
        "Q486972",     # human settlement
        "Q1620908",    # historical country
    },

    # Generic locations (rivers, mountains, landmarks, buildings)
    "LOC": {
        "Q17334923",   # location
        "Q82794",      # geographic region
        "Q33837",      # river
        "Q23397",      # lake
        "Q8502",       # mountain
        "Q33146843",   # island
        "Q41176",      # building
        "Q811979",     # architectural structure
        "Q570116",     # tourist attraction
        "Q486972",     # human settlement (overlap with GPE on purpose)
        "Q515",        # city
        "Q6256",       # country
    },

    # Facilities (FAC in English model) -> buildings & physical infrastructure
    "FAC": {
        "Q41176", "Q811979", "Q174782", "Q44539", "Q23413", "Q12280",  # bridge
    },

    # Nationalities, religious or political groups
    "NORP": {"Q41710", "Q9174", "Q41397", "Q231002", "Q7278"},

    # Products
    "PRODUCT": {"Q2424752", "Q1183543"},

    # Events
    "EVENT": {"Q1190554", "Q1656682", "Q4504495"},

    # Works of art / creative works
    "WORK_OF_ART": {
        "Q838948",     # work of art
        "Q571",        # book
        "Q11424",      # film
        "Q482994",     # album
        "Q105543609",  # musical composition
        "Q7889",       # video game
    },

    # Laws / statutes
    "LAW": {"Q7748", "Q820655"},

    # Languages
    "LANGUAGE": {"Q34770", "Q1288568"},

    # MISC: French model only -> no reliable type filter, fall back to top-1
    "MISC": set(),
}

_session = requests.Session()
_session.headers.update(
    {"User-Agent": "VerbalizationTracker/0.1 (research prototype)"}
)


def _safe(fn, *args):
    """Run a network function, returning None on any failure."""
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001 - best effort by design
        logger.warning("%s failed: %s", getattr(fn, "__name__", fn), e)
        return None


def _normalize_query(text: str) -> str:
    """Strip a leading English article to improve search matching.

    e.g. "The Eiffel Tower" -> "Eiffel Tower" so the landmark (Q243) ranks
    above same-named works of art. The original entity text is preserved.
    """
    lowered = text.lower()
    for article in ("the ", "a ", "an "):
        if lowered.startswith(article):
            return text[len(article):]
    return text


def _wikidata_search_many(text: str, lang: str = "en", limit: int = WIKIDATA_SEARCH_LIMIT):
    """Return up to `limit` Wikidata candidates for a surface form."""
    resp = _session.get(
        WIKIDATA_API,
        params={
            "action": "wbsearchentities",
            "search": _normalize_query(text),
            "language": lang,
            "uselang": lang,
            "format": "json",
            "limit": limit,
            "type": "item",
        },
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    out = []
    for hit in resp.json().get("search", []):
        out.append({
            "id": hit.get("id"),
            "label": hit.get("label"),
            "description": hit.get("description"),
            "url": hit.get("concepturi") or f"https://www.wikidata.org/wiki/{hit.get('id')}",
        })
    return out


def _wikidata_claims_and_sitelinks(qids):
    """For a list of QIDs, fetch P31 (instance of) classes and the enwiki title.

    Returns {qid: {"p31": set[str], "title": str | None}}.
    Batched (up to 50 IDs per request, the Wikidata API limit).
    """
    out: dict[str, dict] = {}
    for i in range(0, len(qids), 50):
        chunk = qids[i : i + 50]
        resp = _session.get(
            WIKIDATA_API,
            params={
                "action": "wbgetentities",
                "ids": "|".join(chunk),
                "props": "claims|sitelinks",
                "sitefilter": "enwiki",
                "format": "json",
            },
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        entities = resp.json().get("entities", {})
        for qid, data in entities.items():
            classes: set[str] = set()
            for stmt in data.get("claims", {}).get("P31", []):
                snak = stmt.get("mainsnak", {})
                if snak.get("snaktype") != "value":
                    continue
                value = snak.get("datavalue", {}).get("value", {})
                if isinstance(value, dict) and value.get("id"):
                    classes.add(value["id"])
            sl = data.get("sitelinks", {}).get("enwiki")
            out[qid] = {
                "p31": classes,
                "title": (sl or {}).get("title"),
            }
    return out


def _pick_best_candidate(candidates: list[dict], spacy_label: str,
                         meta_by_qid: dict[str, dict]) -> tuple[dict | None, str]:
    """Pick the best Wikidata candidate using the spaCy type as a filter.

    Returns (candidate, reason) where `reason` is "type-match" if the type
    filter selected a non-top candidate, "top-1" if no filter applied or no
    candidate matched, or "no-candidate".
    """
    if not candidates:
        return None, "no-candidate"

    expected = SPACY_LABEL_TO_WIKIDATA_CLASSES.get(spacy_label, set())
    if not expected:
        return candidates[0], "top-1"

    for idx, cand in enumerate(candidates):
        cand_classes = meta_by_qid.get(cand["id"], {}).get("p31", set())
        if cand_classes & expected:
            return cand, ("type-match" if idx > 0 else "top-1-type-match")

    # No candidate had a matching P31 — fall back to the top result so we
    # always return something (with a clear "fallback" reason in the JSON).
    return candidates[0], "top-1-fallback"


def _dbpedia_uri_from_title(title: str) -> str:
    return DBPEDIA_RESOURCE_BASE + quote(title.replace(" ", "_"))


def _spotlight(text: str):
    """Annotate full text with DBpedia Spotlight (optional, best-effort)."""
    resp = _session.get(
        SPOTLIGHT_URL,
        params={"text": text, "confidence": SPOTLIGHT_CONFIDENCE},
        headers={"Accept": "application/json"},
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    resources = resp.json().get("Resources", []) or []
    out = []
    for res in resources:
        out.append(
            {
                "uri": res.get("@URI"),
                "surface_form": res.get("@surfaceForm"),
                "offset": int(res.get("@offset", -1)),
                "types": [t for t in (res.get("@types") or "").split(",") if t],
                "similarity_score": float(res.get("@similarityScore", 0) or 0),
            }
        )
    return out


def _match_spotlight(entity, spotlight):
    """Find a Spotlight annotation overlapping the entity span."""
    start, end = entity.get("start_char", -1), entity.get("end_char", -1)
    surface = entity["text"].lower()
    for ann in spotlight:
        off = ann.get("offset", -2)
        sf = (ann.get("surface_form") or "").lower()
        if off >= 0 and start <= off < end:
            return ann
        if sf and sf == surface:
            return ann
    return None


def enrich_entities(text: str, entities: list[dict], lang: str | None = None) -> dict:
    """Add `wikidata` and `dbpedia` links to entities where possible.

    `lang` (e.g. "en", "fr") is used as the Wikidata search language so
    non-English surface forms resolve correctly. The DBpedia URI is always
    derived from the entity's English Wikipedia sitelink (canonical DBpedia).

    Returns {"entities": [...], "meta": {...}} where meta reports which
    services were reached, so the caller/UI can show the linking status.
    """
    search_lang = (lang or "en").split("-")[0].lower() or "en"
    meta = {
        "linking_enabled": ENABLE_LINKING,
        "truststore": _TRUSTSTORE,
        "wikidata_used": False,
        "dbpedia_used": False,
        "spotlight_used": False,
    }

    if not ENABLE_LINKING or not entities:
        return {"entities": entities, "meta": meta}

    # 1. Wikidata search: get a few candidates per unique (text, label) so we
    #    can later filter by type. Caching avoids redundant calls for repeats.
    candidates_cache: dict[tuple[str, str], list[dict]] = {}
    for ent in entities:
        label = ent.get("label", "")
        if label not in LINKABLE_LABELS:
            continue
        key = (ent["text"], label)
        if key not in candidates_cache:
            candidates_cache[key] = (
                _safe(_wikidata_search_many, ent["text"], search_lang)
                or []
            )

    # 2. Batch-fetch P31 (instance of) + enwiki sitelink for every candidate.
    all_qids = {c["id"] for cands in candidates_cache.values() for c in cands if c.get("id")}
    meta_by_qid = (_safe(_wikidata_claims_and_sitelinks, list(all_qids)) or {}) if all_qids else {}

    # 3. Optional Spotlight annotation over the whole transcript.
    spotlight = _safe(_spotlight, text) if ENABLE_SPOTLIGHT else None

    enriched = []
    for ent in entities:
        e = dict(ent)
        label = ent.get("label", "")
        cands = candidates_cache.get((ent["text"], label), [])
        chosen, reason = _pick_best_candidate(cands, label, meta_by_qid) if cands else (None, "no-candidate")

        if chosen:
            meta["wikidata_used"] = True
            e["wikidata"] = {
                "id": chosen["id"],
                "label": chosen["label"],
                "description": chosen["description"],
                "url": chosen["url"],
                "match": reason,                # how the candidate was chosen
                "candidates": len(cands),       # how many were considered
            }
            title = meta_by_qid.get(chosen["id"], {}).get("title")
            if title:
                meta["dbpedia_used"] = True
                e["dbpedia"] = {
                    "uri": _dbpedia_uri_from_title(title),
                    "source": "wikidata-sitelink",
                }

        if spotlight:
            match = _match_spotlight(ent, spotlight)
            if match and match.get("uri"):
                meta["spotlight_used"] = True
                e["dbpedia"] = {
                    "uri": match["uri"],
                    "types": match.get("types", []),
                    "source": "spotlight",
                }
        enriched.append(e)

    return {"entities": enriched, "meta": meta}
