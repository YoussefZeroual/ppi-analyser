# preprocessing/conversation.py

import re
import importlib
import logging
logger = logging.getLogger("ppi_analyser")
def load_sentences(file: str, sent_list: list[int] | None = None) -> list[str]:
    pd = importlib.import_module("pandas")
    df = pd.read_excel(file)
    if sent_list is not None:
        df = df.iloc[sent_list, :]
    sentences = []
    for i, row in df.iterrows():
        left = row["left"].strip().replace("<span class='selectedSent'>", "")
        node = row["node"].strip().replace("<span class='selectedSent'>", "")
        node = node.replace("' ","'") # c' est problem
        right = row["right"].strip().replace("<span class='selectedSent'>", "")
        node = ''.join(c for c in node )#if c.isalnum() or c.isspace())
        if re.match(r'^\[', right):
            right_1 = right[:2]
        else:
            right_1 = ''.join(c for c in right[:2]) #if c.isalnum() or c.isspace())
        right = right_1 + right[2:]
        sentence = " ".join([left, "<PPI>", node, "</PPI>", right])
        sentences.append(sentence)
    return sentences
    return sentences
# preprocessing/conversation.py


def clean_conv(conv: str, mode: str) -> str:
    if mode == "écrit":
        #logger.debug("conv is being cleaned in écrit mode")
        try:
            conv = re.sub(r' -(\w+)', r' – \1', conv)
            conv = re.sub(r'- (\w+)', r' – \1', conv)
            conv = re.sub(r"- ", "–", conv)
            conv = re.sub(r" - ", "–", conv)
        except IndexError:
            pass

        if "–" not in conv:
            return "[Locuteur_1] " + conv

        speaker = 1
        new_text = []
        conv = re.sub(r'(?<!\n)–', '\n–', conv)
        conv = re.sub(r'(?<!\n)«', '\n«', conv)
        conv = re.sub(r'»(?!\n)', '»\n', conv)
        conv = conv.replace("\n«","")
        for line in conv.split("\n"):
            if line.startswith("-") or line.startswith("–") or line.startswith("«"):
                line = f"[Locuteur_{speaker}] {line[1:]}"
                speaker = 3 - speaker
            new_text.append(line)
        return "\n".join(new_text)

    elif mode == "oral":
        return conv

    else:
        return conv


def fix_speaker_turns(conv: str, mode: str = "oral") -> str:
    #logger.debug("fix xpeaker turn processing this raw shit%s",conv)
    # Fix missing opening bracket e.g. VE2] -> [VE2]
    conv = re.sub(r'(?<!\S)([A-Z]{2,3}\d+\])', r'[\1', conv)
    # Fix formatting: collapse newlines after ] and before – or «
    conv = re.sub(r'\]\s*\n+', '] ', conv)
    conv = re.sub(r'\]\s*–', '] ', conv)
    conv = re.sub(r'\n\s*–', ' ', conv)
    conv = re.sub(r'\n\s*«', ' «', conv)
    # Split turns onto separate lines
    conv = re.sub(r'(?<!\n)\[', '\n[', conv)
    conv_lines = conv.split("\n")

    #logger.debug("fix speaker turn is called in mode:%s", mode)
    #logger.debug("conv lines %s", conv_lines)

    # check if already cleaned
    already_cleaned = True
    previous = None
    for line in conv_lines:
        if line:
            speaker = "".join(re.findall(r'\[.*?\]', line))
            if speaker and previous and speaker.strip() == previous.strip():
                already_cleaned = False
                break
            if speaker:
                previous = speaker
    if already_cleaned:
        return conv

    previous_speaker = None
    for i, line in enumerate(conv_lines):
        if line:
            conv_lines[i] = line.strip()
            current_speaker = "".join(re.findall(r'\[.*?\]', line))
            if current_speaker and previous_speaker and current_speaker.strip() == previous_speaker.strip():
                if mode == "oral":
                    conv_lines[i] = line.replace(current_speaker.strip(), '/').strip()
                else:
                    # mark for merging onto previous line
                    conv_lines[i] = '\x00' + line.replace(current_speaker.strip(), '').strip()
            if current_speaker:
                previous_speaker = current_speaker

    # Join continuation lines back onto their speaker line
    # oral: lines starting with / are continuations
    # non-oral: lines marked with \x00 are same-speaker continuations to merge
    result_lines = []
    for line in conv_lines:
        if not line:
            continue
        if line.startswith('/'):
            if result_lines:
                result_lines[-1] += ' ' + line
            else:
                result_lines.append(line)
        elif line.startswith('\x00'):
            if result_lines:
                result_lines[-1] += ' ' + line[:]
            else:
                result_lines.append(line[1:])
        else:
            result_lines.append(line)

    result = "\n".join(line for line in result_lines if line.strip())
    return result
