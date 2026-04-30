"""NLP service: entity extraction and simple rule-based relation extraction."""

import spacy

# Load the small English model. Install with: python -m spacy download en_core_web_sm
_nlp = spacy.load("en_core_web_sm")


def extract_entities(text: str) -> list[dict]:
    """Extract named entities from text using spaCy."""
    doc = _nlp(text)
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


def extract_relations(text: str) -> list[dict]:
    """
    Simple rule-based relation extraction using dependency parsing.

    Strategy: for each verb in the sentence, find its nominal subject (nsubj)
    and direct object (dobj). This captures basic SVO triples.
    """
    doc = _nlp(text)
    relations = []

    for token in doc:
        if token.pos_ == "VERB":
            subjects = [
                child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")
            ]
            objects = [
                child for child in token.children if child.dep_ in ("dobj", "attr", "pobj")
            ]

            # Also check for prepositional objects
            for child in token.children:
                if child.dep_ == "prep":
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            objects.append(grandchild)

            for subj in subjects:
                for obj in objects:
                    relations.append({
                        "subject": subj.text,
                        "predicate": token.lemma_,
                        "object": obj.text,
                        "sentence": token.sent.text.strip(),
                    })

    return relations


def analyze_text(text: str) -> dict:
    """Run full NLP pipeline on text."""
    return {
        "entities": extract_entities(text),
        "relations": extract_relations(text),
    }
