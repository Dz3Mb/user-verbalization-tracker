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

HTTP_TIMEOUT = float(os.getenv("LINKING_HTTP_TIMEOUT", "20"))
DBPEDIA_RESOURCE_BASE = "https://dbpedia.org/resource/"

# If true, the spaCy NER label is used as a *strict* filter when picking a
# Wikidata candidate. Disabled by default because spaCy small/medium models
# frequently mislabel rare or foreign proper nouns (e.g. a Polish first name
# labelled "LOC"), which would bias the search toward the wrong type entirely.
# When disabled, the linker still uses spaCy to find entity *boundaries* in
# the text, but the candidate selection ignores the predicted label and
# relies on Wikidata's own type (P31) instead.
USE_SPACY_TYPE = _flag("LINKING_USE_SPACY_TYPE", "false")

# Sentence-level disambiguation: rerank Wikidata candidates by the semantic
# similarity between their description and the sentence containing the entity.
# This resolves ambiguities like "Hugo" (Victor Hugo vs. Hugo the film) when
# the surrounding sentence gives enough context. Implementation lives below;
# it lazily loads a multilingual SentenceTransformer model.
ENABLE_SEMANTIC_RERANK = _flag("LINKING_SEMANTIC_RERANK", "true")
SEMANTIC_MODEL_NAME = os.getenv(
    "LINKING_SEMANTIC_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
# When the spread between candidates is small, semantic similarity is just
# noise. We only consider it meaningful if the best similarity exceeds this
# threshold and beats the next-best by SEMANTIC_REORDER_MARGIN. Tuned to
# conservatively prefer Wikidata's default ranking unless context is strong.
SEMANTIC_MIN_SCORE = float(os.getenv("LINKING_SEMANTIC_MIN_SCORE", "0.5"))
SEMANTIC_REORDER_MARGIN = float(os.getenv("LINKING_SEMANTIC_MARGIN", "0.15"))

# Walk P279 (subclass of) up to N hops when checking type matches, so a
# candidate of class "U.S. state" (Q35657) is recognized as a kind of
# "administrative territorial entity" (Q56061) without enumerating every
# subclass. Cached in memory.
P279_MAX_DEPTH = int(os.getenv("LINKING_P279_DEPTH", "2"))

# Labels we treat as "name-like" entities worth linking. We include both the
# English (OntoNotes) and French (UD/multilingual) label sets, plus we keep
# this list inclusive on purpose: when spaCy's predicted label is wrong, we
# do not want to drop the entity entirely just because the label is unusual.
LINKABLE_LABELS = {
    # English model
    "PERSON", "ORG", "GPE", "LOC", "NORP", "FAC", "PRODUCT", "EVENT",
    "WORK_OF_ART", "LAW", "LANGUAGE",
    # French / multilingual models
    "PER", "MISC",
}

# Labels we explicitly do NOT link (numerics, time expressions, etc. — they
# rarely have a meaningful Wikidata item that resolves from the surface form).
NON_LINKABLE_LABELS = {
    "DATE", "TIME", "CARDINAL", "ORDINAL", "PERCENT", "MONEY", "QUANTITY",
}

# Number of Wikidata candidates to consider when type-filtering. Larger values
# improve recall on ambiguous surface forms at a small bandwidth cost.
WIKIDATA_SEARCH_LIMIT = int(os.getenv("WIKIDATA_SEARCH_LIMIT", "6"))

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
        "Q1549591",    # big city
        "Q5119",       # capital
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

# Human-readable labels for the most common Wikidata P31 classes, so each
# linked entity can show its *actual* type from the knowledge graph,
# independently of spaCy's (sometimes wrong) NER label. See `_readable_type`.
WIKIDATA_CLASS_LABELS: dict[str, str] = {
    "Q5": "human",
    "Q43229": "organization",
    "Q4830453": "business",
    "Q783794": "company",
    "Q6881511": "enterprise",
    "Q891723": "public company",
    "Q161726": "multinational corporation",
    "Q3918": "university",
    "Q163740": "nonprofit organization",
    "Q484652": "international organization",
    "Q327333": "government agency",
    "Q7278": "political party",
    "Q6256": "country",
    "Q3624078": "sovereign state",
    "Q515": "city",
    "Q1549591": "big city",
    "Q5119": "capital",
    "Q35657": "U.S. state",
    "Q484170": "commune of France",
    "Q3957": "town",
    "Q486972": "human settlement",
    "Q33837": "river",
    "Q23397": "lake",
    "Q8502": "mountain",
    "Q41176": "building",
    "Q811979": "architectural structure",
    "Q570116": "tourist attraction",
    "Q11424": "film",
    "Q571": "book",
    "Q482994": "album",
    "Q7889": "video game",
    "Q34770": "language",
}

_session = requests.Session()
_session.headers.update(
    {"User-Agent": "VerbalizationTracker/0.1 (research prototype)"}
)


# --- Semantic re-ranker (lazy-loaded sentence transformer) -----------------

_embed_model = None
_embed_failed = False  # avoid re-trying on every request after a failure


def _get_embedder():
    """Return a (lazy-loaded) SentenceTransformer for context disambiguation.

    The model is downloaded the first time it is needed. If the package is
    not installed or the download fails (e.g. offline), we log once and
    return None so the linker degrades gracefully.
    """
    global _embed_model, _embed_failed
    if _embed_model is not None or _embed_failed:
        return _embed_model
    if not ENABLE_SEMANTIC_RERANK:
        return None
    try:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(SEMANTIC_MODEL_NAME)
        logger.info("Loaded semantic reranker: %s", SEMANTIC_MODEL_NAME)
    except Exception as e:  # noqa: BLE001 - best effort
        logger.warning("Semantic reranker unavailable: %s", e)
        _embed_failed = True
        _embed_model = None
    return _embed_model


def _cosine(a, b) -> float:
    """Cosine similarity between two numpy vectors (assumed non-zero)."""
    import numpy as np
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _semantic_rerank(candidates, context_sentence: str):
    """Score candidates by semantic similarity of their Wikidata description
    to the entity's surrounding sentence. Returns a dict {qid: similarity}.

    Returns an empty dict if the embedder is unavailable, the context is
    empty, or no candidate has a description.
    """
    if not candidates or not (context_sentence or "").strip():
        return {}
    model = _get_embedder()
    if model is None:
        return {}
    items = [(c["id"], c.get("description") or c.get("label") or "") for c in candidates]
    items = [(qid, desc) for qid, desc in items if desc]
    if not items:
        return {}
    try:
        ctx_emb = model.encode(context_sentence, normalize_embeddings=True)
        cand_embs = model.encode(
            [desc for _, desc in items], normalize_embeddings=True
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("semantic encoding failed: %s", e)
        return {}
    return {qid: float(ctx_emb @ cand_embs[i]) for i, (qid, _) in enumerate(items)}


# --- Wikidata P279 (subclass-of) traversal ---------------------------------

_p279_cache: dict[str, set[str]] = {}


def _expand_p279(qids: set[str]) -> set[str]:
    """Walk P279 (subclass-of) up to P279_MAX_DEPTH hops from the given QIDs.

    Returns the union of the input QIDs and all ancestor classes. Cached so
    repeated entities don't pay the network cost. Best-effort: a network
    failure short-circuits the expansion and we return what we have so far.
    """
    if P279_MAX_DEPTH <= 0 or not qids:
        return set(qids)
    visited: set[str] = set(qids)
    frontier: set[str] = set(qids) - set(_p279_cache.keys())
    # Seed with cached parents for already-known QIDs.
    for q in qids:
        if q in _p279_cache:
            visited |= _p279_cache[q]
    for _ in range(P279_MAX_DEPTH):
        new_qids = frontier - visited
        if not new_qids:
            break
        parents = _safe(_wikidata_p279_parents, list(new_qids))
        if not parents:
            break
        visited |= new_qids
        next_frontier: set[str] = set()
        for q in new_qids:
            ps = parents.get(q, set())
            _p279_cache[q] = ps
            next_frontier |= ps
        frontier = next_frontier
    visited |= frontier
    return visited


def _wikidata_p279_parents(qids):
    """Map QIDs to the set of their direct P279 parents (batched)."""
    out: dict[str, set[str]] = {}
    for i in range(0, len(qids), 50):
        chunk = qids[i : i + 50]
        resp = _session.get(
            WIKIDATA_API,
            params={
                "action": "wbgetentities",
                "ids": "|".join(chunk),
                "props": "claims",
                "format": "json",
            },
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        for qid, data in resp.json().get("entities", {}).items():
            classes: set[str] = set()
            for stmt in data.get("claims", {}).get("P279", []):
                snak = stmt.get("mainsnak", {})
                if snak.get("snaktype") != "value":
                    continue
                value = snak.get("datavalue", {}).get("value", {})
                if isinstance(value, dict) and value.get("id"):
                    classes.add(value["id"])
            out[qid] = classes
    return out


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
                         meta_by_qid: dict[str, dict],
                         context_sentence: str = "") -> tuple[dict | None, str]:
    """Pick the best Wikidata candidate, combining three signals:

    1. **Semantic similarity** between the candidate's description and the
       sentence containing the entity. Resolves homonyms when the surrounding
       sentence gives enough context (e.g. "Victor Hugo wrote Les Misérables"
       prefers the writer over the company "Hugo Boss"). Enabled by default
       via `LINKING_SEMANTIC_RERANK`.
    2. **Type-aware filter** using the spaCy NER label, with subclass
       expansion (P279). Opt-in via `LINKING_USE_SPACY_TYPE=true`; off by
       default because small spaCy models often mislabel rare entities.
    3. **Known-type fallback**: among candidates with any recognized P31
       class. This is the default selection when (1) is uninformative.

    Returns (candidate, reason) where `reason` reports how the pick was made.
    """
    if not candidates:
        return None, "no-candidate"

    known_classes = set(WIKIDATA_CLASS_LABELS.keys())

    # Quick check: does the top-1 already have a known P31 type? If so, it's
    # a "solid" candidate (Wikidata's default ranking surfaced an entity of a
    # recognized kind, which is usually the right answer for common names).
    # We then only let the semantic reranker override it on a strong signal.
    top = candidates[0]
    top_p31 = meta_by_qid.get(top["id"], {}).get("p31", set())
    top_is_solid = bool(top_p31 & known_classes) or bool(
        _expand_p279(top_p31) & known_classes
    )

    # 1. Semantic re-ranking with the context sentence (if any).
    sims = _semantic_rerank(candidates, context_sentence) if context_sentence else {}
    if sims:
        ranked = sorted(candidates, key=lambda c: sims.get(c["id"], 0.0), reverse=True)
        top_sim = sims.get(ranked[0]["id"], 0.0)
        second_sim = sims.get(ranked[1]["id"], 0.0) if len(ranked) > 1 else 0.0
        margin = top_sim - second_sim
        # Override the default top-1 only on a strong signal. When the default
        # top-1 is already solid, require an even larger margin to overrule it.
        required_margin = SEMANTIC_REORDER_MARGIN * (2 if top_is_solid else 1)
        if top_sim >= SEMANTIC_MIN_SCORE and margin >= required_margin:
            # Don't bother overriding if the reranker picks the same candidate.
            if ranked[0]["id"] != top["id"]:
                return ranked[0], f"semantic-rerank({top_sim:.2f})"

    # 2. Type-aware mode (opt-in): strict filter by spaCy label, P279 expanded.
    if USE_SPACY_TYPE:
        expected = SPACY_LABEL_TO_WIKIDATA_CLASSES.get(spacy_label, set())
        if expected:
            expected_expanded = _expand_p279(expected)
            for idx, cand in enumerate(candidates):
                cand_classes = meta_by_qid.get(cand["id"], {}).get("p31", set())
                cand_expanded = _expand_p279(cand_classes)
                if cand_expanded & expected_expanded:
                    return cand, ("type-match" if idx > 0 else "top-1-type-match")

    # 3. Type-agnostic (default): first candidate with any recognized P31
    #    class (P279-expanded so subclasses of known types also count).
    for idx, cand in enumerate(candidates):
        cand_classes = meta_by_qid.get(cand["id"], {}).get("p31", set())
        if cand_classes & known_classes:
            return cand, ("known-type" if idx > 0 else "top-1")
        cand_expanded = _expand_p279(cand_classes)
        if cand_expanded & known_classes:
            return cand, ("known-type-p279" if idx > 0 else "top-1-p279")

    # 4. Last resort: top-1.
    return candidates[0], "top-1-fallback"


def _dbpedia_uri_from_title(title: str) -> str:
    return DBPEDIA_RESOURCE_BASE + quote(title.replace(" ", "_"))


def _readable_type(p31_qids: set) -> tuple[str | None, list]:
    """Derive a human-readable type for an entity from its Wikidata P31 classes.

    Returns (label, qids) where `label` is a readable string (e.g. "human",
    "company", "city") taken from the first recognized class, or None if no
    class is recognized. `qids` is the raw list of P31 QIDs (always returned,
    for transparency). This is what lets the UI show the *actual* type from the
    knowledge graph, independently of spaCy's NER label.
    """
    qids = sorted(p31_qids)
    for qid in qids:
        if qid in WIKIDATA_CLASS_LABELS:
            return WIKIDATA_CLASS_LABELS[qid], qids
    return None, qids


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


def _sentence_around(text: str, start_char: int, end_char: int) -> str:
    """Return the sentence that contains [start_char, end_char) in `text`.

    Uses a simple punctuation-based split, which is sufficient for short
    transcripts and avoids loading a second NLP pipeline just for this.
    """
    if not text or start_char < 0 or end_char <= start_char:
        return text or ""
    # Find sentence boundaries before and after the entity span.
    left_punct = max(text.rfind(p, 0, start_char) for p in (".", "!", "?", "\n"))
    right_candidates = [text.find(p, end_char) for p in (".", "!", "?", "\n")]
    right_candidates = [r for r in right_candidates if r != -1]
    right_punct = min(right_candidates) if right_candidates else len(text)
    return text[left_punct + 1 : right_punct + 1].strip()


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
        # Skip numeric/time labels (rarely have a useful Wikidata item).
        if label in NON_LINKABLE_LABELS:
            continue
        # Be inclusive on the linkable side: if the label is not a known
        # "name-like" type either, we still attempt linking — spaCy's label
        # is unreliable for rare or foreign proper nouns, and we'd rather try
        # and let Wikidata decide than drop the entity silently.
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
        # Pass the sentence that contains the entity for context-aware
        # semantic disambiguation.
        context = _sentence_around(
            text, ent.get("start_char", -1), ent.get("end_char", -1)
        )
        chosen, reason = (
            _pick_best_candidate(cands, label, meta_by_qid, context)
            if cands else (None, "no-candidate")
        )

        if chosen:
            meta["wikidata_used"] = True
            kg_p31 = meta_by_qid.get(chosen["id"], {}).get("p31", set())
            kg_type, kg_type_qids = _readable_type(kg_p31)
            e["wikidata"] = {
                "id": chosen["id"],
                "label": chosen["label"],
                "description": chosen["description"],
                "url": chosen["url"],
                "match": reason,                # how the candidate was chosen
                "candidates": len(cands),       # how many were considered
                "kg_type": kg_type,             # actual type from Wikidata (P31)
                "kg_type_qids": kg_type_qids,   # raw P31 QIDs, for transparency
                "spacy_label": label,           # what spaCy thought (may differ)
                "label_mismatch": bool(
                    kg_type and label and label not in ("MISC",)
                    and not (meta_by_qid.get(chosen["id"], {}).get("p31", set())
                             & SPACY_LABEL_TO_WIKIDATA_CLASSES.get(label, set()))
                ),
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
