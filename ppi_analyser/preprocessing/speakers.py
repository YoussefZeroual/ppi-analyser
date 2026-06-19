# preprocessing/speakers.py

import re


def detect_speakers(text: str, mode: str) -> tuple[str, list[str]]:
    locuteur = ""
    for line in text.splitlines():
        if "</PPI>" in line:
            match = re.search(r'\[\s*([^\]]+?)\s*\]', line)
            locuteur = match.group(0).strip() if match else "Locuteur 1"

    interlocuteurs = list(set(re.findall(r'\[(.*?)\]', text)) - {locuteur})
    return locuteur, interlocuteurs


def is_clausative(conv: str, mode: str) -> bool:
    turn = get_loc_full_turn(conv, mode)[0].replace("\n", "")
    for line in turn.split("\n"):
        if ("<PPI>" in line) and ("</PPI>" in line):
            target = re.sub(r'[\[({].*?[\]})]', '', line).strip()
            before = target[:target.find("<PPI>")]
            after = target[target.find("</PPI>") + 7:]
            return len(before) + len(after) == 0
    return False


def get_loc_full_turn(conv: str, mode: str) -> tuple[str, str]:
    if mode == "écrit_ia":
    	logger.warning("%s",conv)
        conv = re.sub(r'(?<!\n)\[', '\n[', conv)
        for line in conv.split("\n"):
            if re.findall(r'<PPI>(.*?)</PPI>', line):
                sent = re.findall(r'<PPI>(.*?)</PPI>', line)[0]
                return line, sent

    conv_lines = conv.split("\n")
    loc, interloc = detect_speakers(conv, mode)
    match = re.search(r'<PPI>(.*?)</PPI>', conv, re.DOTALL)
    expr = match.group(1) if match else "Aucune balise <PPI> détectée"

    before = []
    after = []
    sent = ""
    l_size = len(conv_lines)

    try:
        for i in range(l_size):
            if "<PPI>" in conv_lines[i] and not (
                loc in conv_lines[i + 1] or loc in conv_lines[i - 1]
            ):
                sent = conv_lines[i]
                expr = sent[sent.find("<PPI>"):sent.find("</PPI>")]
                break
            elif loc and "<PPI>" in conv_lines[i] and (
                loc in conv_lines[i + 1] or loc in conv_lines[i - 1]
            ):
                sent = conv_lines[i]
                for k in range(i - 1, 0, -1):
                    if loc not in conv_lines[k]:
                        break
                    before.append(conv_lines[k])
                for j in range(i + 1, l_size):
                    if loc not in conv_lines[j]:
                        break
                    after.append(conv_lines[j])
    except IndexError:
        if "<PPI>" in conv_lines[i]:
            return conv_lines[i], expr

    before = "\n".join(before[::-1])
    after = "\n".join(after)
    loc_turn = before + "\n" + sent + "\n" + after
    return loc_turn, expr
