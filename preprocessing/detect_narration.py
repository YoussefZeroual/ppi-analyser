
import re

# fix 10/3 12:07 removed import stanza
def detect_segments(text: str,nlp) -> str:
    # --- NEW: Protect PPI tags by replacing them with placeholders ---
    ppi_contents = []
    text = re.sub(r"»", "", text)
    text = re.sub(r"»", "", text)
    def protect_ppi_tags(match):
        # Store the full tag (including content) and replace with placeholder
        ppi_contents.append(match.group(0))
        return f"__PPI_{len(ppi_contents)-1}__"
    
    # Protect both opening tags, closing tags, and self-closing tags
    # This regex matches <PPI>, </PPI>, and <PPI>anything</PPI>
    text_with_placeholders = re.sub(r'(<PPI>.*?</PPI>|</?PPI>)', protect_ppi_tags, text, flags=re.DOTALL)
    # --- END NEW ---

    # --- 1. New Logic: Direct Quote Processing ---
    # This identifies blocks within « » and tags them if > 3 words
    def process_quotes(t):
        pattern = r'«([^»]+)»'
        segments = []
        last_idx = 0
        for match in re.finditer(pattern, t):
            # Narration: text before the quote
            narr = t[last_idx:match.start()].strip()
            if narr: segments.append(f"<narration>{narr}</narration>")
            
            # Dialogue: content between quotes
            content = match.group(1).strip()
            if len(content.split()) > 3:
                segments.append(f"<dialogue>«{content}»</dialogue>")
            else:
                segments.append(f"«{content}»")
            last_idx = match.end()
            
        # Remaining text after last quote
        remaining = t[last_idx:].strip()
        if remaining: segments.append(f"<narration>{remaining}</narration>")
        return " ".join(segments)

    # If the text contains French quotes, we prioritize this simple logic
    if '«' in text_with_placeholders and '»' in text_with_placeholders:
        result = process_quotes(text_with_placeholders)
    else:
        # --- 2. Original Logic: Deep Linguistic Analysis ---
        DIALOGUE_CUES = {
            'je', 'me', 'moi', 'mon', 'ma', 'mes', 'tu', 'te', 'toi', 'ton', 'ta', 'tes',
            'nous', 'notre', 'nos', 'vous', 'votre', 'vos', 'on'
        }
        
        VERBES_INTRODUCTEURS = {
            'dire', 'demander', 'répondre', 'crier', 'chuchoter', 'murmurer',
            'ajouter', 'continuer', 'reprendre', 'lancer', 'déclarer',
            'hasarder', 'souffler', 'répliquer', 'interrompre', 'rétorquer','siffler'
        }

        def is_passe_simple(word):
            return word.feats and 'Tense=Past' in word.feats and 'VerbForm=Fin' in word.feats

        def is_imparfait(word):
            return word.feats and 'Tense=Imp' in word.feats and 'VerbForm=Fin' in word.feats

        def is_present(word):
            return word.feats and 'Tense=Pres' in word.feats and 'VerbForm=Fin' in word.feats

        def is_dialogue_segment(text_seg, words):
            if not text_seg.strip(): return False
            if '?' in text_seg or '!' in text_seg: return True
            if any(w.text.lower() in DIALOGUE_CUES for w in words): return True
            
            has_imp = any(is_imparfait(w) for w in words)
            has_pres = any(is_present(w) for w in words)
            if has_imp and not has_pres: return False
            
            if re.match(r'^\s*[–—"-]', text_seg) and not any(is_passe_simple(w) for w in words):
                return True
            return False
            
        doc = nlp(text_with_placeholders)
        final_output = []

        for sent in doc.sentences:
            s_start = sent.tokens[0].start_char
            s_end = sent.tokens[-1].end_char
            sent_text = text_with_placeholders[s_start:s_end]
            words = sent.words
            
            narr_anchor = next((w for w in words if w.lemma.lower() in VERBES_INTRODUCTEURS or is_passe_simple(w) or is_imparfait(w)), None)

            if narr_anchor and any(m in sent_text for m in ['–', '—']):
                anchor_token = next(t for t in sent.tokens if any(w.id == narr_anchor.id for w in t.words))
                break_point = anchor_token.start_char
                
                rel_break = break_point - s_start
                part1_text = sent_text[:rel_break].rstrip()
                part2_text = sent_text[rel_break:].lstrip()
                
                p1_words = [w for w in words if next(t for t in sent.tokens if any(word.id == w.id for word in t.words)).start_char < break_point]
                p2_words = [w for w in words if next(t for t in sent.tokens if any(word.id == w.id for word in t.words)).start_char >= break_point]

                tag1 = "dialogue" if is_dialogue_segment(part1_text, p1_words) else "narration"
                tag2 = "narration" if (narr_anchor.lemma.lower() in VERBES_INTRODUCTEURS or is_passe_simple(narr_anchor) or is_imparfait(narr_anchor)) else "dialogue"
                
                final_output.append(f"<{tag1}>{part1_text}</{tag1}> <{tag2}>{part2_text}</{tag2}>")
            else:
                tag = "dialogue" if is_dialogue_segment(sent_text, words) else "narration"
                final_output.append(f"<{tag}>{sent_text}</{tag}>")

        result = "\n".join(final_output)

    # --- NEW: Restore original PPI tags from placeholders ---
    for i, original_tag in enumerate(ppi_contents):
        result = result.replace(f"__PPI_{i}__", original_tag)
    # --- END NEW ---

    return result


def get_descendants(head_id, words):
    """ دالة كترجع كاع الـ IDs ديال الكلمات اللي تابعة لواحد الـ head """
    descendants = set()
    for w in words:
        if w.head == head_id:
            descendants.add(w.id)
            descendants.update(get_descendants(w.id, words))
    return descendants
    

def clean_dialogue(text):
    pattern = r'(?m)^[-–]|(?<=\s)[-–]'
    cleaned = re.sub(pattern, r'\n–', text).strip()
    

    return cleaned

# fix 7/3 9:30 added <PPI></PPI>
# fix 7/3 10:34 added restoring ppi tags in returned value
def get_dialogue_ecrit(text):

    ppi = "".join(re.findall(r'<PPI>(.*?)</PPI>',text)) if re.findall(r'<PPI>(.*?)</PPI>',text) else None
    all_tags = re.findall(r'<(?:dialogue|PPI)>(.*?)</(?:dialogue|PPI)>', text)
    return " ".join(all_tags).replace(ppi,"<PPI>"+ppi+"</PPI>")
# fix 8/3 14:30 created a specefic func for écrit mode because it broke the normal one and in many cases
# it captured only the ppi instead of the whole dialogue
def get_dialogue(text):

    all_tags = re.findall(r'<dialogue>(.*?)</dialogue>', text)
    
    return " ".join(all_tags)

if __name__ == "__main__":

    text = "<narration>blabla <PPI>dfdsfsdfs</PPI></narration>helo"
    print(get_dialogue(text))
    

