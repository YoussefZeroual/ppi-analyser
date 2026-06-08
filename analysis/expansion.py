import sys
import re
import pandas as pd
from stanza_client import StanzaClient
from format_excel import format_ppi_bold

client = StanzaClient()


def extract_ppi_sentence(tagged_line):
    match = re.search(r'<PPI>(.*?)</PPI>', tagged_line, re.IGNORECASE)
    if not match:
        return None, None
    ppi_text = match.group(1).strip()
    # Trouve la limite gauche : dernier séparateur avant la balise
    #pre = tagged_line[:match.start()]
    post = tagged_line[match.end():]
    # Coupe à gauche sur / ou début de ligne
    #left = re.split(r'/', pre)[-1]
    # Coupe à droite sur / 
    right = re.split(r'/', post)[0]
    clean_seg = re.sub(r'</?PPI>', '', right, flags=re.IGNORECASE).strip() # <-- removed left be cause exp is always in the right
    return ppi_text, clean_seg

def get_ppi_ids(sentence, ppi_text):
    words = sentence["words"]
    ppi_clean = re.sub(r'\s*-\s*', '-', ppi_text.lower()).strip()
    for i in range(len(words)):
        for j in range(i+1, len(words)+1):
            window = words[i:j]
            surface = re.sub(r'\s*-\s*', '-', " ".join(w["text"] for w in window).lower())
            if surface == ppi_clean:
                return set(w["id"] for w in window)
    return set()


def get_ppi_head(sentence, ppi_ids):
    for w in sentence["words"]:
        if w["id"] in ppi_ids and w["head"] not in ppi_ids:
            return w
    return None


def get_subtree(head_word, words, exclude_ids=set()):
    subtree_ids = {head_word["id"]}
    changed = True
    while changed:
        changed = False
        for w in words:
            if w["head"] in subtree_ids and w["id"] not in subtree_ids and w["id"] not in exclude_ids:
                subtree_ids.add(w["id"])
                changed = True
    return sorted([w for w in words if w["id"] in subtree_ids], key=lambda w: w["id"])


def detect_expansion(text, ppi_text):
    text_nlp = client.process(text)
    for sentence in text_nlp["sentences"]:
        ppi_ids = get_ppi_ids(sentence, ppi_text)
        if not ppi_ids:
            continue
        ppi_head = get_ppi_head(sentence, ppi_ids)
        if not ppi_head:
            continue
        words = sentence["words"]
        dependants = [w for w in words if w["head"] == ppi_head["id"] and w["id"] not in ppi_ids]
        expansions = []
        for dep in dependants:
            deprel = dep["deprel"]
            upos = dep["upos"]
            if deprel == "xcomp" and upos == "VERB":
                subtree = get_subtree(dep, words, exclude_ids=ppi_ids)
                expansions.append({"type": "infinitive", "tokens": subtree})
            elif deprel in ("ccomp", "csubj"):
                subtree = get_subtree(dep, words, exclude_ids=ppi_ids)
                expansions.append({"type": "completive_que", "tokens": subtree})
            elif deprel in ("nmod", "obl", "obl:arg", "obj") and upos in ("NOUN", "PRON", "ADP"):
                subtree = get_subtree(dep, words, exclude_ids=ppi_ids)
                expansions.append({"type": "nominal_prep", "tokens": subtree})
        return expansions[:1] if expansions else [{"type": None, "tokens": []}]
    return [{"type": None, "tokens": []}]


def process_file(input_path):
    if input_path.endswith(".csv"):
        df_in = pd.read_csv(input_path)
    else:
        df_in = pd.read_excel(input_path)
    print(df_in.columns.tolist())
    print(df_in.head(2))
    rows = []
    for _, row in df_in.iterrows():
        conversation = str(row.get("Conversation", ""))
        lines = conversation.split("\n")
        for line in lines:
            if not re.search(r'<PPI>', line, re.IGNORECASE):
                continue
            ppi_text, clean_seg = extract_ppi_sentence(line)
            if not ppi_text:
                continue
            expansions = detect_expansion(clean_seg, ppi_text)
            exp = expansions[0]
            expansion_text = " ".join(w["text"] for w in exp["tokens"]) if exp["tokens"] else ""
            print(f"Tour : {line.strip()}")
            print(f"  Type     : {exp['type']}")
            print(f"  Expansion: {expansion_text}\n")
            rows.append({
                "Tour": line.strip(),
                "PPI": ppi_text,
                "Type_expansion_1": exp["type"] if len(expansions) > 0 else "",
                "Expansion_1": expansion_text,
                "Type_expansion_2": expansions[1]["type"] if len(expansions) > 1 else "",
                "Expansion_2": " ".join(w["text"] for w in expansions[1]["tokens"]) if len(expansions) > 1 and expansions[1]["tokens"] else "",
            })

    df_out = pd.DataFrame(rows)
    base = re.sub(r'\.(xlsx|csv)$', '', input_path)
    output_path = f"{base}_expansion.xlsx"
    format_ppi_bold(df_out, output_path)
    print(f"Résultat enregistré : {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python expansion.py <fichier.csv|xlsx>")
        sys.exit(1)
    process_file(sys.argv[1])
