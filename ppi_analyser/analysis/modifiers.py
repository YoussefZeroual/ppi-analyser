import re
import logging
import yaml
from pathlib import Path
from nltk.stem.snowball import FrenchStemmer
import nltk
nltk.download('punkt')
_stemmer = FrenchStemmer()

def load_modifier_rules(path: str | Path | None = None) -> dict:
    if path is None:
        path = Path(__file__).parent / "modifier_rules.yaml"
    with open(path, encoding="utf-8") as f:
        rules = yaml.safe_load(f)
    return {
        "upos": set(rules.get("upos", [])),
        "deprel": set(rules.get("deprel", [])),
        "lemma": set(rules.get("lemma", [])),
        "excluded_upos": set(rules.get("excluded_upos", [])),
        "excluded_deprel":set(rules.get("excluded_deprel", [])),
        "excluded_lemma":set(rules.get("excluded_lemma", []))
    }

MODIFIER_RULES = load_modifier_rules()
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
    def is_neg(standard_form_lemmas):
        for lemma in standard_form_lemmas:
            if lemma in ("rien","pas","nullement","pas"):
                return True
        return False
        
    rules = MODIFIER_RULES
    if nlp is None:
        logger.debug("find_modifier: no nlp object was passed, skipping modifier detection")
        return [], []
    ppi_sent = tagged_ppi_nlp
    if hasattr(ppi_sent, 'sentences'):
        ppi_sent = ppi_sent.sentences[0] if ppi_sent.sentences else None
    if ppi_sent is None:
        logger.debug("find_modifier: PPI sentence not found in text")
        return [], []
        
    ppi_standard_form_lemmas = [w.lemma for s in lemme_doc.sentences for w in s.words]
    ppi_standard_stems = {_stemmer.stem(w.lemma) for s in lemme_doc.sentences for w in s.words}
    ppi_standard_lemma_set = set(ppi_standard_form_lemmas)

    ppi_modifs = [
        w for s in text_nlp.sentences for w in s.words
        if (
            any(
                gov.lemma in ppi_standard_lemma_set or
                _stemmer.stem(gov.lemma) in ppi_standard_stems
                for gov in s.words if gov.id == w.head
            )
            and (
                w.upos in rules["upos"]
                or w.deprel in rules["deprel"]
                or w.lemma in rules["lemma"]
            )
        )
        and w.lemma not in ppi_standard_form_lemmas
        and _stemmer.stem(w.lemma) not in ppi_standard_stems
        and w.upos not in rules["excluded_upos"]
        and w.deprel not in rules["excluded_deprel"]
        and w.lemma not in rules["excluded_lemma"]
    ]
    logger.warning("%s", [f"{w.text}_{w.upos}:{w.deprel}:head={w.head}_id={w.id}" for s in text_nlp.sentences for w in s.words])
    logger.warning("ppi_lemma_set: %s", ppi_standard_lemma_set)
    logger.warning("ppi_stems: %s", ppi_standard_stems)
    for modif in ppi_modifs:
        if modif.lemma in ("ne","pas") and is_neg(ppi_standard_form_lemmas):
            ppi_modifs.remove(modif)

    if ppi_modifs is not None:
        subtrees = [f"<MOD>{get_tree(w.lemma, text_nlp, nlp, occurrence)}</MOD>" for w in ppi_modifs]
        labels   = [_upos_fr(w.upos) for w in ppi_modifs]
        logger.debug(
            "find_modifier | standard_form_lemmas: %s | ppi_sent lemmas: %s | detected modifs: %s",
            ppi_standard_form_lemmas,
            [w.lemma for w in ppi_sent.words],
            [(w.lemma, w.upos, w.deprel) for w in ppi_modifs],
        )
        return labels, subtrees
    return None, None

def format_modifiers(labels: list[str], subtrees: list[str]) -> str:
    if (labels is None) or (subtrees is None):
    	return "Aucun modifieur"
    parts = [f"{label}: {subtree}" for label, subtree in zip(labels, subtrees) if subtree]
    return ", ".join(parts) if parts else "Aucun modifieur"


