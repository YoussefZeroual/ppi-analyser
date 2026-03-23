# preprocessing/conversation.py

import re
import importlib

def load_sentences(file: str, sent_list: list[int] | None = None) -> list[str]:
    pd = importlib.import_module("pandas")
    df = pd.read_excel(file)
    if sent_list is not None:
        df = df.iloc[sent_list, :]

    sentences = []
    for i, row in df.iterrows():
        left = row["left"].strip().replace("<span class='selectedSent'>", "")
        node = row["node"].strip().replace("<span class='selectedSent'>", "")
        right = row["right"].strip().replace("<span class='selectedSent'>", "")

        node = ''.join(c for c in node if c.isalnum() or c.isspace())
        right_1 = ''.join(c for c in right[:2] if c.isalnum() or c.isspace())
        right = right_1 + right[2:]

        sentence = " ".join([left, "<PPI>", node, "</PPI>", right])
        sentences.append(sentence)

    return sentences
# preprocessing/conversation.py


def clean_conv(conv: str, mode: str) -> str:
    if mode == "écrit":
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
        conv = conv.replace("–", "\n–").replace("«", "\n«").replace("»", "»\n")
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
    # Fix missing opening bracket e.g. VE2] -> [VE2]
    conv = re.sub(r'(?<!\[)([A-Z]{2,3}\d+\])', r'[\1', conv)

    conv = conv.replace("[", "\n[")
    conv_lines = conv.split("\n")

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
                    conv_lines[i] = line.replace(current_speaker.strip(), '').strip()
            if current_speaker:
                previous_speaker = current_speaker

    # Join continuation lines (starting with /) back onto their speaker line
    result_lines = []
    for line in conv_lines:
        if line.startswith('/'):
            if result_lines:
                result_lines[-1] += ' ' + line
            else:
                result_lines.append(line)
        elif line:
            result_lines.append(line)

    result = "\n".join(result_lines)

    print(conv)
    print("---")
    print(result)
    input("zebbi")

    return result
