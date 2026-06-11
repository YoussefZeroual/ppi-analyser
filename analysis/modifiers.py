import re
import logging

logger = logging.getLogger(__name__)

UPOS_FR = {
    "ADV":   "adverbe",
    "PROPN": "nom propre",
    "ADJ":   "adjectif",
    "INTJ":  "interjection",
    "NOUN":  "nom",
    "VERB":  "verbe",
}

def _upos_fr(upos: str) -> str:
    return UPOS_FR.get(upos, upos)


def get_tree(target_lemma, doc, nlp, occurrence=0):
    w_id = None
    target_sent = None
    count = 0
    for s in doc.sentences:
        for w in s.words:
            if w.lemma == target_lemma:
                if count == occurrence:
                    w_id = w.id
                    target_sent = s
                    break
                count += 1
        if w_id:
            break
    if not target_sent or not w_id:
        return ""
    tree = [w.text for w in target_sent.words if (w.head == w_id) or (w.lemma == target_lemma)]
    return " ".join(tree)

def get_ppi_sent(tagged_ppi_nlp, text_nlp, nlp):
    tagged_ppi_lemmas = [w.lemma for s in tagged_ppi_nlp.sentences for w in s.words if w.upos != "PUNCT"]
    for s in text_nlp.sentences:
        s_lemmas = [w.lemma.strip() for w in s.words]
        if set(tagged_ppi_lemmas).issubset(set(s_lemmas)):
            return s, s_lemmas
    return None, None

def find_modifier(tagged_ppi_nlp, lemme_doc, text_nlp, nlp, occurrence=0):
    if nlp is None:
        logger.debug("find_modifier: no nlp object was passed, skipping modifier detection")
        return [], []

    ppi_sent = tagged_ppi_nlp
    if ppi_sent is None:
        logger.debug("find_modifier: PPI sentence not found in text")
        return [], []

    ppi_standard_form_lemmas = [w.lemma for s in lemme_doc.sentences for w in s.words]

    ppi_modifs = [
        w for w in ppi_sent.words
        if (
            w.upos in ("ADV", "PROPN", "ADJ", "INTJ")
            or w.deprel in ("obl:mod", "nmod", "acl:relcl", "dislocated", "amod")
        )
        and w.lemma not in ppi_standard_form_lemmas
    ]

    subtrees = [f"<MOD>{get_tree(w.lemma, text_nlp, nlp, occurrence)}</MOD>" for w in ppi_modifs]
    labels   = [_upos_fr(w.upos) for w in ppi_modifs]

    return labels, subtrees

def format_modifiers(labels: list[str], subtrees: list[str]) -> str:
    parts = [f"{label}: {subtree}" for label, subtree in zip(labels, subtrees) if subtree]
    return ", ".join(parts) if parts else "Aucun modifieur"


