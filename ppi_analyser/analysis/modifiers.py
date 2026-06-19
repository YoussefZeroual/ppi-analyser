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

    rules = MODIFIER_RULES
    if nlp is None:
        logger.debug("find_modifier: no nlp object was passed, skipping modifier detection")
        return [], []

    ppi_sent = tagged_ppi_nlp
    if hasattr(ppi_sent, 'sentences'):  # Document passed instead of Sentence
        ppi_sent = ppi_sent.sentences[0] if ppi_sent.sentences else None
    if ppi_sent is None:
        logger.debug("find_modifier: PPI sentence not found in text")
        return [], []
    ppi_standard_form_lemmas = [w.lemma for s in lemme_doc.sentences for w in s.words]
    ppi_form_heads_ids = [w.id for s in lemme_doc.sentences for w in s.words if w.lemma in ppi_standard_form_lemmas]
    
    logger.warning("ppi_standard_form_lemmas %s",ppi_standard_form_lemmas)
    ppi_standard_stems = {_stemmer.stem(w.lemma) for s in lemme_doc.sentences for w in s.words} #<-- utilisation des radicaux car le lemme est différent pour désolé et désolée (probleme stanza)
    logger.warning("ppi_standard_stems %s",ppi_standard_stems)
    ppi_sent_stems =  {_stemmer.stem(w.lemma) for w in ppi_sent.words}
    ppi_modifs = [
        w for w in ppi_sent.words
        if (
            w.head in ppi_form_heads_ids and
            (w.upos in rules["upos"]
            or w.deprel in rules["deprel"]
            or w.lemma in rules["lemma"])
        )
        and w.lemma not in ppi_standard_form_lemmas
        and _stemmer.stem(w.lemma) not in ppi_standard_stems
        and w.upos not in rules["excluded_upos"]
        and w.deprel not in rules["excluded_deprel"]
        and w.lemma not in rules["excluded_lemma"]
        
    ]

    subtrees = [f"<MOD>{get_tree(w.lemma, text_nlp, nlp, occurrence)}</MOD>" for w in ppi_modifs]
    labels   = [_upos_fr(w.upos) for w in ppi_modifs]
    logger.debug(
    "find_modifier | standard_form_lemmas: %s | ppi_sent lemmas: %s | detected modifs: %s",
    ppi_standard_form_lemmas,
    [w.lemma for w in ppi_sent.words],
    [(w.lemma, w.upos, w.deprel) for w in ppi_modifs],
)
    return labels, subtrees

def format_modifiers(labels: list[str], subtrees: list[str]) -> str:
    parts = [f"{label}: {subtree}" for label, subtree in zip(labels, subtrees) if subtree]
    return ", ".join(parts) if parts else "Aucun modifieur"


