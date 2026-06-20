# analysis/position.py

import re
import logging
from ppi_analyser.config import AnalysisMode
from ppi_analyser.analysis.expansion import detect_expansion
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers — détection d'expansion sur objets Stanza natifs
# (logique portée depuis expansion.py, sans appel réseau supplémentaire)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Fonction principale
# ---------------------------------------------------------------------------

def get_pos(conv: str, mode: str, tokenization_mode: str = "nlp", nlp=None, state=None,sent_id:int =None) -> tuple | None:
    from ppi_analyser.preprocessing.speakers import get_loc_full_turn, detect_speakers
    from ppi_analyser.analysis.expansion import detect_expansion
    full_turn_lemmas = [w.lemma for s in state.nlp_preprocessed_turn[sent_id]["full_turn_stripped_nlp_doc"].sentences for w in s.words]
    forme_doc = state.nlp_preprocessed_turn[sent_id]["forme_nlp_doc"]
    form_lemmas = [w.lemma for s in forme_doc.sentences for  w in s.words] if forme_doc is not None else []
    from ppi_analyser.analysis.expansion import extract_ppi_sentence,detect_expansion
    ppi_text, _ = extract_ppi_sentence(conv)
    expansion = detect_expansion(state.nlp_preprocessed_turn[sent_id]["full_turn_nlp_doc"], ppi_text,state.nlp_preprocessed_turn[sent_id]["ppi_occurrence"])
    
    
    
    expansion_tokens = [w.text.lower() for w in expansion[0]["tokens"] if w.upos != "PUNCT"] 
    expression_tokens = [w.text.lower() for s in state.nlp_preprocessed_turn[sent_id]["surface_sent_nlp"].sentences for w in s.words if w.upos != "PUNCT"]
    extended_expression = expression_tokens+expansion_tokens
    
    
    full_turn, sent = get_loc_full_turn(conv, mode)
    loc, interloc = detect_speakers(conv, mode)

    if nlp is None:
        logger.info("Warning: no nlp object was passed for position detection!")

    # Récupérer les docs précalculés
    full_turn_doc = state.nlp_preprocessed_turn[sent_id]["full_turn_nlp_doc"]
    forme_doc = state.nlp_preprocessed_turn[sent_id]["surface_sent_nlp"]

    # Tokens du tour complet
    nlp_turn = [w.text.lower() for s in full_turn_doc.sentences for w in s.words if w.upos != "PUNCT"]
    #logger.debug("traitement de la position sentid= %s:full turn %s , expression + expansion %s , %s",sent_id,nlp_turn,extended_expression)
    # Tokens de la forme PPI
    forme_lemmas = [w.lemma for s in forme_doc.sentences for w in s.words if w.upos != "PUNCT"]

   
    nlp_sent = extended_expression
    
    logger.warning("%s",extended_expression)
    
    # Calculer la position
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
    sent_clean = re.sub(r'</?PPI>', '', sent).strip()

    start = indices[0]
    end = turn_l - indices[-1] - 1

    if start < 5 and end < 5:
        return ("Totale", f"La PPI {sent_clean} occupe quasiment la totalité du tour de parole de <strong>{loc}</strong>: *{full_turn_display}*")
    elif start < 5 and end >= 5:
        return ("Initiale", f"La PPI {sent_clean} démarre dans les 5 premiers tokens du tour de parole de <strong>{loc}</strong>: *{full_turn_display}*")
    elif start >= 5 and end >= 5:
        return ("Médiane", f"La PPI {sent_clean} apparaît après les 5 premiers tokens du tour de parole de <strong>{loc}</strong>: *{full_turn_display}*")
    elif start >= 5 and end < 5:
        return ("Finale", f"La PPI {sent_clean} apparaît dans les 5 derniers tokens du tour de parole de <strong>{loc}</strong>: *{full_turn_display}*")
    else:
        return ("Indéterminé", "Impossible de déterminer la position de la PPI")


