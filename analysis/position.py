# analysis/position.py

import re
import logging

logger = logging.getLogger(__name__)


def get_pos(conv: str, mode: str, tokenization_mode: str = "simple", nlp=None) -> tuple | None:
    from ppi_analyser.preprocessing.speakers import get_loc_full_turn, detect_speakers
    full_turn, sent = get_loc_full_turn(conv, mode)
    loc, interloc = detect_speakers(conv, mode)
    if is_clausative(conv, mode):
        return ("Totale", f"La PPI {sent} occupe la totalité du tour de parole de **{loc}**: *{full_turn}*")
    full_turn_clean = (
        "".join(full_turn).replace("\n", " ").replace("  ", " ")
        .replace(f"[{loc}]", "").replace("<PPI>", "").replace("</PPI>", "").strip().lower().replace("."," ")
    )
    sent_clean = (
        sent.replace(f"[{loc}]", "").replace("<PPI>", "").replace("</PPI>", "").strip().lower()
    )

    
    # Tokenization based on mode
    if tokenization_mode == "nlp" and nlp is not None:
        # Use Stanza for tokenization
        doc_turn = nlp(full_turn_clean.replace("\'", " "))
        doc_sent = nlp(sent_clean.replace("\'", " "))
        
        # Extract tokens excluding punctuation
        nlp_turn = [word.text for sentence in doc_turn.sentences for word in sentence.words if word.upos != "PUNCT"]
        nlp_sent = [word.text for sentence in doc_sent.sentences for word in sentence.words if word.upos != "PUNCT"]
    else:
        # Simple tokenization (split by whitespace)
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
