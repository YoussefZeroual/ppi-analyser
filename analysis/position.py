# analysis/position.py

import re
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers — détection d'expansion sur objets Stanza natifs
# (logique portée depuis expansion.py, sans appel réseau supplémentaire)
# ---------------------------------------------------------------------------

def _extract_ppi_text(conv: str) -> str | None:
    match = re.search(r'<PPI>(.*?)</PPI>', conv, re.IGNORECASE)
    return match.group(1).strip() if match else None


def _get_ppi_ids_stanza(sentence, ppi_text: str) -> set:
    """
    Identifie les ids des tokens correspondant à ppi_text dans une phrase
    Stanza native (attributs .id, .text sur les Word objects).
    """
    words = sentence.words
    ppi_clean = re.sub(r'\s*-\s*', '-', ppi_text.lower()).strip()
    for i in range(len(words)):
        for j in range(i + 1, len(words) + 1):
            window = words[i:j]
            surface = re.sub(r'\s*-\s*', '-', " ".join(w.text for w in window).lower())
            if surface == ppi_clean:
                return {w.id for w in window}
    return set()


def _get_expansion_tokens_stanza(sentence, ppi_ids: set) -> list:
    """
    Retourne les tokens de la première expansion syntaxique détectée
    (infinitive, complétive, groupe nominal/prépositionnel) à partir de la
    tête de la PPI, en travaillant sur un objet sentence Stanza natif.
    Renvoie une liste vide si aucune expansion n'est trouvée.
    """
    words = sentence.words

    # Tête de la PPI : le token de la PPI dont le gouverneur est hors de la PPI
    ppi_head = None
    for w in words:
        if w.id in ppi_ids and w.head not in ppi_ids:
            ppi_head = w
            break
    if not ppi_head:
        return []

    # Dépendants directs de la tête, hors PPI
    dependants = [w for w in words if w.head == ppi_head.id and w.id not in ppi_ids]

    for dep in dependants:
        deprel = dep.deprel
        upos = dep.upos
        if (
            (deprel == "xcomp" and upos == "VERB")
            or deprel in ("ccomp", "csubj")
            or (deprel in ("nmod", "obl", "obl:arg", "obj") and upos in ("NOUN", "PRON", "ADP"))
        ):
            # Sous-arbre complet du dépendant, en excluant les tokens de la PPI
            subtree_ids = {dep.id}
            changed = True
            while changed:
                changed = False
                for w in words:
                    if w.head in subtree_ids and w.id not in subtree_ids and w.id not in ppi_ids:
                        subtree_ids.add(w.id)
                        changed = True
            return sorted(
                [w for w in words if w.id in subtree_ids],
                key=lambda w: w.id,
            )
    return []


# ---------------------------------------------------------------------------
# Fonction principale
# ---------------------------------------------------------------------------

def get_pos(conv: str, mode: str, tokenization_mode: str = "nlp", nlp=None) -> tuple | None:
    from ppi_analyser.preprocessing.speakers import get_loc_full_turn, detect_speakers
    full_turn, sent = get_loc_full_turn(conv, mode)
    loc, interloc = detect_speakers(conv, mode)

    if is_clausative(conv, mode):
        return ("Totale", f"La PPI {sent} occupe la totalité du tour de parole de **{loc}**: *{full_turn}*")

    full_turn_clean = (
        "".join(full_turn).replace("\n", " ").replace("  ", " ")
        .replace(f"[{loc}]", "").replace("<PPI>", "").replace("</PPI>", "").strip().lower().replace(".", " ")
    )
    sent_clean = (
        sent.replace(f"[{loc}]", "").replace("<PPI>", "").replace("</PPI>", "").strip().lower()
    )

    if tokenization_mode == "nlp" and nlp is not None:
        # Tokenisation Stanza — une seule passe par texte
        doc_turn = nlp(full_turn_clean.replace("\'", " "))
        doc_sent = nlp(sent_clean.replace("\'", " "))

        nlp_turn = [
            w.text
            for sentence in doc_turn.sentences
            for w in sentence.words
            if w.upos != "PUNCT"
        ]
        nlp_sent = [
            w.text
            for sentence in doc_sent.sentences
            for w in sentence.words
            if w.upos != "PUNCT"
        ]

        # Détection de l'expansion sur doc_sent déjà parsé — pas de second appel Stanza
        ppi_text = _extract_ppi_text(conv)
        if ppi_text:
            for sentence in doc_sent.sentences:
                ppi_ids = _get_ppi_ids_stanza(sentence, ppi_text)
                if ppi_ids:
                    exp_tokens = _get_expansion_tokens_stanza(sentence, ppi_ids)
                    if exp_tokens:
                        expansion_words = [w.text for w in exp_tokens if w.upos != "PUNCT"]
                        nlp_sent = nlp_sent + expansion_words
                        logger.debug(
                            "Expansion détectée pour '%s' : %s",
                            ppi_text,
                            " ".join(expansion_words),
                        )
                    break

    else:
        # Tokenisation simple par espace
        nlp_turn = full_turn_clean.replace("\'", " ").split()
        nlp_sent = sent_clean.replace("\'", " ").split()

    turn_l = len(nlp_turn)
    sent_l = len(nlp_sent)
    indices = []
    for i in range(turn_l - sent_l + 1):
        if nlp_turn[i:i + sent_l] == nlp_sent:
            indices = list(range(i, i + sent_l))
            break
    if not indices:
        return None

    full_turn_display, sent_display = get_loc_full_turn(conv, mode)
    full_turn_display = (
        full_turn_display
        .replace("<PPI>", "<strong>")
        .replace("</PPI>", "</strong>")
    )
    full_turn_display = re.sub(r'\[(.*?)\]', '', full_turn_display).strip()

    if indices[0] < 5 and turn_l - indices[-1] > 5:
        return ("Initiale", f"La PPI {sent_clean} démarre dans les 5 premiers tokens du tour de parole de <strong>{loc}</strong>: *{full_turn_display}*")
    elif indices[0] >= 5 and turn_l - indices[-1] >= 5:
        return ("Médiane", f"La PPI {sent_clean} apparaît après les 5 premiers tokens du tour de parole de <strong>{loc}</strong>: *{full_turn_display}*")
    elif indices[0] >= 5 and turn_l - indices[-1] >= 1:
        return ("Finale", f"La PPI {sent_clean} apparaît dans les 5 derniers tokens du tour de parole de <strong>{loc}</strong>: *{full_turn_display}*")
    elif indices[0] <= 5 and turn_l - indices[-1] >= 1:
        return ("Totale", f"La PPI {sent_clean} occupe quasiment la totalité du tour de parole de <strong>{loc}</strong>: *{full_turn_display}*")
    else:
        return ("Indéterminé", "Impossible de déterminer la position de la PPI")


def is_clausative(conv: str, mode: str) -> bool:
    from ppi_analyser.preprocessing.speakers import get_loc_full_turn
    turn = get_loc_full_turn(conv, mode)[0].replace("\n", "")
    for line in turn.split("\n"):
        if "<PPI>" in line and "</PPI>" in line:
            target = re.sub(r'[\[({].*?[\]})]', '', line).strip()
            before = target[:target.find("<PPI>")]
            after = target[target.find("</PPI>") + 7:]
            return len(before) + len(after) == 0
    return False
