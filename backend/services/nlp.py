"""NLP service: entity extraction and rule-based relation extraction.

Public API (import these from `services.nlp`):
- analyze_text(text, lang)      -> {"entities": [...], "relations": [...]}
- extract_entities(text, lang)  -> list of entity dicts
- extract_relations(text, lang) -> list of relation triples
- token_breakdown(text, lang)   -> per-token analysis (debug / teaching)

How to extend:
- Add a language: register its spaCy model in `_MODEL_BY_LANG` (the loader
  prefers the `md` model and falls back to `sm`, then to English).
- Tune relations: edit `extract_relations` (dependency-label rules) or the
  helper `_phrase` (how a noun phrase is reconstructed).
- Tune which entities are kept: edit `extract_entities`.
"""

import logging

import spacy

logger = logging.getLogger("nlp")

# Map language codes to spaCy model candidates, in preference order.
# We prefer the largest CNN-based models available (lg > md > sm). The
# transformer-based "_trf" English model requires `curated-tokenizers` which
# does not yet build on Python 3.13, so we don't list it here.
_MODEL_BY_LANG = {
    "en": ["en_core_web_lg", "en_core_web_md", "en_core_web_sm"],
    "fr": ["fr_core_news_lg", "fr_core_news_md", "fr_core_news_sm"],
}
_DEFAULT_LANG = "en"
_nlp_cache: dict[str, "spacy.language.Language"] = {}

# Dependency labels for subjects, across English (OntoNotes) and French (UD).
_SUBJ_DEPS = {"nsubj", "nsubjpass", "nsubj:pass", "csubj"}

# Leading determiners/articles to trim from extracted phrases (EN + FR).
_LEADING_DETS = {
    "the", "a", "an",
    "le", "la", "les", "l'", "un", "une", "des", "du", "de", "l",
}


def _expand_conj(tokens: list) -> list:
    """Expand a list of tokens to include coordinated siblings (dep=conj).

    e.g. the subject head of "Microsoft and Apple" yields both tokens.
    """
    out = []
    for t in tokens:
        out.append(t)
        for child in t.children:
            if child.dep_ == "conj":
                out.append(child)
    return out


def _build_chunk_map(doc) -> dict:
    """Map each token index to the noun chunk span containing it (if any).

    Noun chunks give clean base noun phrases. Not all language models support
    them (French raises NotImplementedError), so this degrades gracefully.
    """
    chunk_map = {}
    try:
        for chunk in doc.noun_chunks:
            for tok in chunk:
                chunk_map[tok.i] = chunk
    except NotImplementedError:
        pass
    return chunk_map


def _strip_leading_det(text: str) -> str:
    """Remove a leading article/determiner for cleaner phrases."""
    parts = text.split(" ", 1)
    if len(parts) == 2 and parts[0].lower().strip("'") in _LEADING_DETS:
        return parts[1]
    return text


def _phrase(token, chunk_map: dict) -> str:
    """Return the full noun-phrase text for a token's head.

    Uses the noun chunk when available; otherwise gathers contiguous
    compound/name modifiers (e.g. "Barack Obama"). Leading determiners are
    trimmed.
    """
    chunk = chunk_map.get(token.i)
    if chunk is not None:
        return _strip_leading_det(chunk.text)

    idxs = [token.i] + [
        c.i for c in token.children
        if c.dep_ in ("compound", "flat", "flat:name", "name")
    ]
    lo, hi = min(idxs), max(idxs)
    return _strip_leading_det(token.doc[lo:hi + 1].text)


def _get_nlp(lang: str | None):
    """Return a loaded spaCy pipeline for `lang`, falling back gracefully.

    Tries the preferred models for the language in order (md, then sm). If
    none is installed, logs a warning and falls back to the default English
    model so the pipeline keeps working.
    """
    code = (lang or _DEFAULT_LANG).split("-")[0].lower()
    if code not in _MODEL_BY_LANG:
        code = _DEFAULT_LANG

    if code in _nlp_cache:
        return _nlp_cache[code]

    nlp = None
    for model_name in _MODEL_BY_LANG[code]:
        try:
            nlp = spacy.load(model_name)
            logger.info("Loaded spaCy model '%s' for language '%s'.", model_name, code)
            break
        except OSError:
            continue

    if nlp is None:
        logger.warning(
            "No spaCy model installed for language '%s' (%s); "
            "falling back to English.",
            code, ", ".join(_MODEL_BY_LANG[code]),
        )
        if _DEFAULT_LANG not in _nlp_cache:
            for model_name in _MODEL_BY_LANG[_DEFAULT_LANG]:
                try:
                    _nlp_cache[_DEFAULT_LANG] = spacy.load(model_name)
                    break
                except OSError:
                    continue
        nlp = _nlp_cache[_DEFAULT_LANG]

    _nlp_cache[code] = nlp
    return nlp


def extract_entities(text: str, lang: str | None = None) -> list[dict]:
    """Extract named entities from text using spaCy."""
    doc = _get_nlp(lang)(text)
    entities = []
    seen = set()
    for ent in doc.ents:
        key = (ent.text, ent.label_)
        if key not in seen:
            seen.add(key)
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
            })
    return entities


def extract_relations(text: str, lang: str | None = None) -> list[dict]:
    """Rule-based relation extraction over spaCy dependency parses.

    Improvements over a naive subject-verb-object scan:
    - subjects/objects are returned as full noun phrases (e.g. "Barack Obama",
      "the United States") instead of single head tokens;
    - coordinations are expanded ("Microsoft and Apple" -> two relations);
    - passive voice is detected and the triple direction is corrected;
    - copular sentences ("Paris is the capital") are captured;
    - the predicate includes the verb particle and/or preposition
      ("work" + "at" -> "work at"), making triples more meaningful.

    Works with both the English (OntoNotes labels) and French (Universal
    Dependencies labels) spaCy models.
    """
    doc = _get_nlp(lang)(text)
    chunk_map = _build_chunk_map(doc)

    relations = []
    seen = set()

    def add(subject, predicate, obj, sent, passive=False):
        subject, obj = subject.strip(), obj.strip()
        if not subject or not obj or subject.lower() == obj.lower():
            return
        key = (subject, predicate, obj)
        if key in seen:
            return
        seen.add(key)
        relations.append({
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "sentence": sent.strip(),
            "passive": passive,
        })

    for token in doc:
        # --- Copular sentences: "Paris is the capital" / "Paris est la capitale"
        # UD style: the predicate noun/adj is the head and has a `cop` child.
        cops = [c for c in token.children if c.dep_ == "cop"]
        if cops:
            subjects = _expand_conj(
                [c for c in token.children if c.dep_ in _SUBJ_DEPS]
            )
            for s in subjects:
                add(_phrase(s, chunk_map), cops[0].lemma_,
                    _phrase(token, chunk_map), token.sent.text)

        if token.pos_ not in ("VERB", "AUX"):
            continue

        subjects = _expand_conj(
            [c for c in token.children if c.dep_ in _SUBJ_DEPS]
        )
        if not subjects:
            continue

        passive = any(c.dep_ in ("nsubjpass", "nsubj:pass") for c in token.children)
        particle = next(
            (c.text for c in token.children if c.dep_ in ("prt", "compound:prt")),
            "",
        )

        # Collect objects, remembering the preposition that introduced each
        # and whether it is a passive *agent* (by-phrase / par-phrase).
        objects = []  # list of (token, preposition_text, is_agent)

        for c in token.children:
            if c.dep_ in ("dobj", "obj", "attr", "oprd", "dative"):
                objects.append((c, "", False))
            elif c.dep_ == "agent":  # English "by ..." in passives
                for gc in c.children:
                    if gc.dep_ in ("pobj", "obj"):
                        objects.append((gc, c.text, True))
            elif c.dep_ == "prep":  # English prepositional phrase
                prep = c.text
                for gc in c.children:
                    if gc.dep_ in ("pobj", "obj"):
                        objects.append((gc, prep, prep.lower() == "by"))
            elif c.dep_ in ("obl", "obl:arg", "obl:mod", "obl:agent"):  # French oblique
                case = next((cc.text for cc in c.children if cc.dep_ == "case"), "")
                objects.append((c, case, case.lower() in ("par", "by")))

        # Keep only meaningful object heads (nouns/proper nouns/pronouns/etc.),
        # filtering out stray function words.
        objects = [
            (o, prep, agent) for (o, prep, agent) in objects
            if o.pos_ in ("NOUN", "PROPN", "PRON", "ADJ", "NUM", "X")
        ]

        # Expand coordinated objects, keeping their preposition/agent flag.
        expanded_objects = []
        for obj, prep, agent in objects:
            for o in _expand_conj([obj]):
                expanded_objects.append((o, prep, agent))

        for s in subjects:
            for obj, prep, agent in expanded_objects:
                predicate = token.lemma_
                if particle:
                    predicate += f" {particle}"
                if prep:
                    predicate += f" {prep}"

                subj_text = _phrase(s, chunk_map)
                obj_text = _phrase(obj, chunk_map)
                # In a passive clause, only swap direction for the agent
                # ("eaten by John" -> John eat ...). Locatives/other obliques
                # keep the grammatical subject as the triple subject.
                if passive and agent:
                    subj_text, obj_text = obj_text, subj_text
                add(subj_text, predicate, obj_text, token.sent.text, passive)

    return relations


def analyze_text(text: str, lang: str | None = None) -> dict:
    """Run full NLP pipeline on text for a given language."""
    return {
        "entities": extract_entities(text, lang),
        "relations": extract_relations(text, lang),
    }


def token_breakdown(text: str, lang: str | None = None) -> list[dict]:
    """Return spaCy's per-token analysis of the text (debug / teaching tool).

    This shows *how spaCy reads the transcription*, token by token. The most
    relevant field for "where does an entity begin / continue / end" is the
    IOB tag (`ent_iob`):
      - "B" = Begin: first token of a named entity
      - "I" = Inside: a token in the middle or at the end of an entity
      - "O" = Outside: the token is not part of any entity
    (spaCy uses the IOB2 scheme; a single-token entity is just "B".)

    Other fields:
      - pos:  coarse part of speech (PROPN, VERB, NOUN, DET, ...)
      - dep:  syntactic dependency label to the head token
      - head: the token this one attaches to (drives relation extraction)
    """
    doc = _get_nlp(lang)(text)
    out = []
    for tok in doc:
        out.append({
            "i": tok.i,
            "text": tok.text,
            "lemma": tok.lemma_,
            "pos": tok.pos_,
            "dep": tok.dep_,
            "head": tok.head.text,
            "ent_iob": tok.ent_iob_,      # B / I / O  (begin / inside / outside)
            "ent_type": tok.ent_type_,    # entity label when part of an entity
            "is_sent_start": bool(tok.is_sent_start),
        })
    return out
