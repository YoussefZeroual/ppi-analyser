# preprocessing/segmentation.py

import re
import logging
from ppi_analyser.preprocessing.segmentation_cache import get as cache_get, set as cache_set

logger = logging.getLogger(__name__)


def detect_segments_ia(text: str, model: str) -> str:
    if not re.search(r'[«"\u2018\u2019][^»"\u2018\u2019]*[»"\u2018\u2019]|[\u2014\u2013-]\s*\w+|«[^»]*»', text):
        return "<dialogue>[Locuteur1] " + text + "</dialogue>"

    cached = cache_get(text, model)
    if cached is not None:
        return cached

    system_prompt = """\
Put narration parts inside <narration> tags and dialogal parts inside <dialogue> tags
narration tags indlcude also inserted segments like: "dit-il", "répondit Jean", etc.
<PPI> tagged segments should always be inside <dialogue> tags
dialogues should start with a speaker label (e.g [Jean] if it can be infered from the context, otherwise just a generic label e.g: [speaker 1], [speaker 2], etc

dont add anything that isnt already in the text

few shots example:
* entrée:
"– Par tous les dieux !  s'écria-t-il.  Il est impossible de manipuler une épée aussi longue sans se blesser soi-même !  – Il faut s'entraîner pendant des années avant de savoir la manier de façon adéquate, assura Kira avec amusement.  Si tu es patient, je te l'enseignerai.  – Pendant que l'ennemi franchira nos frontières sans que je puisse l'arrêter parce que je ne sais pas m'en servir ? <PPI>Très peu pour moi</PPI>!  Il rendit l'épée à Sage sans cacher sa frustration de ne pouvoir l'utiliser correctement malgré son expérience des armes.  Le jeune guerrier se tourna alors vers Jasson qui l'observait toujours avec suspicion.  – Veux-tu l'essayer aussi ? ?  lui offrit Sage.  – Pourquoi pas ?  répondit-il en relevant fièrement la tête..  L'hybride la lui tendit et dégaina l'épée à lame unique qu'il avait reçue du Roi d'Émeraude lors de son adoubement. 
"
* sortie: 
"<dialogue>[locuteur 1] Par tous les dieux !</dialogue><narration>s'écria-t-il.</narration><dialogue>Il est impossible de manipuler une épée aussi longue sans se blesser soi-même !</dialogue><dialogue> [locuteur 2] Il faut s'entraîner pendant des années avant de savoir la manier de façon adéquate, assura Kira avec amusement.  Si tu es patient, je te l'enseignerai.</dialogue>  <dialogue>[locuteur 1] Pendant que l'ennemi franchira nos frontières sans que je puisse l'arrêter parce que je ne sais pas m'en servir ? <PPI>Très peu pour moi</PPI>!</dialogue>  <narration>Il rendit l'épée à Sage sans cacher sa frustration de ne pouvoir l'utiliser correctement malgré son expérience des armes.  Le jeune guerrier se tourna alors vers Jasson qui l'observait toujours avec suspicion.</narration> <dialogue>[locuteur 2] Veux-tu l'essayer aussi ? ?</dialogue><narration>  lui offrit Sage.</narration><dialogue> [locuteur 1] Pourquoi pas ?</dialogue><narration>répondit-il en relevant fièrement la tête..  L'hybride la lui tendit et dégaina l'épée à lame unique qu'il avait reçue du Roi d'Émeraude lors de son adoubement.</narration> "
---

"""

    prompt = (
        text
    )

    from ppi_analyser.models.factory import get_provider
    provider = get_provider(model.split('_')[0], model.split('_')[1])
    result = provider.complete(system_prompt, prompt)
    result = result.replace("—", "")

    cache_set(text, model, result)
    return result


_BATCH_SYSTEM_PROMPT = """\
for each text, Put narration parts inside <narration> tags and dialogal parts inside <dialogue> tags
narration tags indlcude also inserted segments like: "dit-il", "répondit Jean", etc.
<PPI> tagged segments should always be inside <dialogue> tags
dialogues should start with a speaker label (e.g [Jean] if it can be infered from the context, otherwise just a generic label e.g: [speaker 1], [speaker 2], etc

dont add anything that isnt already in the text

few shots example:
* entrée:
"– Par tous les dieux !  s'écria-t-il.  Il est impossible de manipuler une épée aussi longue sans se blesser soi-même !  – Il faut s'entraîner pendant des années avant de savoir la manier de façon adéquate, assura Kira avec amusement.  Si tu es patient, je te l'enseignerai.  – Pendant que l'ennemi franchira nos frontières sans que je puisse l'arrêter parce que je ne sais pas m'en servir ? <PPI>Très peu pour moi</PPI>!  Il rendit l'épée à Sage sans cacher sa frustration de ne pouvoir l'utiliser correctement malgré son expérience des armes.  Le jeune guerrier se tourna alors vers Jasson qui l'observait toujours avec suspicion.  – Veux-tu l'essayer aussi ? ?  lui offrit Sage.  – Pourquoi pas ?  répondit-il en relevant fièrement la tête..  L'hybride la lui tendit et dégaina l'épée à lame unique qu'il avait reçue du Roi d'Émeraude lors de son adoubement. 
"
* sortie: 
"<dialogue>[locuteur 1] Par tous les dieux !</dialogue><narration>s'écria-t-il.</narration><dialogue>Il est impossible de manipuler une épée aussi longue sans se blesser soi-même !</dialogue><dialogue> [locuteur 2] Il faut s'entraîner pendant des années avant de savoir la manier de façon adéquate, assura Kira avec amusement.  Si tu es patient, je te l'enseignerai.</dialogue>  <dialogue>[locuteur 1] Pendant que l'ennemi franchira nos frontières sans que je puisse l'arrêter parce que je ne sais pas m'en servir ? <PPI>Très peu pour moi</PPI>!</dialogue>  <narration>Il rendit l'épée à Sage sans cacher sa frustration de ne pouvoir l'utiliser correctement malgré son expérience des armes.  Le jeune guerrier se tourna alors vers Jasson qui l'observait toujours avec suspicion.</narration> <dialogue>[locuteur 2] Veux-tu l'essayer aussi ? ?</dialogue><narration>  lui offrit Sage.</narration><dialogue> [locuteur 1] Pourquoi pas ?</dialogue><narration>répondit-il en relevant fièrement la tête..  L'hybride la lui tendit et dégaina l'épée à lame unique qu'il avait reçue du Roi d'Émeraude lors de son adoubement.</narration> "
---

Une fois tous les textes traités, contrôle :

- [ ] Le nombre de blocs `<TEXTE> </TEXTE>` est égal au nombre de textes reçus
- [ ] Chaque bloc est séparé du suivant par `===SEPARATOR===` (exactement, sans guillemets, sans espace avant ou après)
- [ ] Aucun texte n'a été omis ou traité deux fois
- [ ] Aucune explication, commentaire ou texte libre n'apparaît en dehors des balises

---

## FORMAT DE SORTIE — IMPÉRATIF

Traite **tous** les textes et retourne **une seule réponse**, structurée ainsi :

```
<TEXTE>[texte 1 traité]</TEXTE>
===SEPARATOR===
<TEXTE>[texte 2 traité]</TEXTE>
===SEPARATOR===
<TEXTE>[texte 3 traité]</TEXTE>
```

> ⚠️ `===SEPARATOR===` doit apparaître **entre** les blocs, jamais avant le premier ni après le dernier.

**NE DONNE AUCUNE EXPLICATION SUPPLÉMENTAIRE.**
"""


def _parse_batch_result(raw: str, expected: int) -> list[str]:
    chunks = raw.split("===SEPARATOR===")
    results = []
    for chunk in chunks:
        m = re.search(r'<TEXTE>(.*?)</TEXTE>', chunk.strip(), re.DOTALL)
        if m:
            results.append(m.group(1).strip())
    if len(results) != expected:
        logger.warning(
            "_parse_batch_result: expected %d results, got %d", expected, len(results)
        )
    return results


def detect_segments_ia_batch(texts: list[str], model: str) -> list[str]:
    if not texts:
        return []

    processed = [None] * len(texts)
    batch_texts = []
    batch_indices = []

    for i, text in enumerate(texts):
        if not re.search(r'[«"\u2018\u2019][^»"\u2018\u2019]*[»"\u2018\u2019]|[\u2014\u2013-]\s*\w+|«[^»]*»', text):
            processed[i] = "<dialogue>[Locuteur1] " + text + "</dialogue>"
            continue
        cached = cache_get(text, model)
        if cached is not None:
            processed[i] = cached
            continue
        batch_texts.append(text)
        batch_indices.append(i)

    if batch_texts:
        batch_prompt = (
            "Analyse maintenant les textes suivants et produis une sortie "
            "avec les balises selon le format spécifié:\n\n"
        )
        for i, text in enumerate(batch_texts):
            batch_prompt += f"--- TEXTE {i + 1} ---\n{text}\n\n"

        from ppi_analyser.models.factory import get_provider
        provider = get_provider(model.split('_')[0], model.split('_')[1])
        raw_result = provider.complete(_BATCH_SYSTEM_PROMPT, batch_prompt)
        parsed = _parse_batch_result(raw_result, expected=len(batch_texts))

        for list_pos, original_idx in enumerate(batch_indices):
            if list_pos < len(parsed):
                result = parsed[list_pos]
                processed[original_idx] = result
                cache_set(batch_texts[list_pos], model, result)
            else:
                logger.warning("No parsed result for text index %d, using fallback", original_idx)
                processed[original_idx] = "<dialogue>[Locuteur1] " + texts[original_idx] + "</dialogue>"

    return processed


def detect_segments(text: str, nlp) -> str:
    ppi_contents = []
    text = re.sub(r"»", "", text)

    def protect_ppi_tags(match):
        ppi_contents.append(match.group(0))
        return f"__PPI_{len(ppi_contents) - 1}__"

    text_with_placeholders = re.sub(
        r'(<PPI>.*?</PPI>|</?PPI>)', protect_ppi_tags, text, flags=re.DOTALL
    )

    def process_quotes(t):
        pattern = r'«([^»]+)»'
        segments = []
        last_idx = 0
        for match in re.finditer(pattern, t):
            narr = t[last_idx:match.start()].strip()
            if narr:
                segments.append(f"<narration>{narr}</narration>")
            content = match.group(1).strip()
            if len(content.split()) > 3:
                segments.append(f"<dialogue>«{content}»</dialogue>")
            else:
                segments.append(f"«{content}»")
            last_idx = match.end()
        remaining = t[last_idx:].strip()
        if remaining:
            segments.append(f"<narration>{remaining}</narration>")
        return " ".join(segments)

    if '«' in text_with_placeholders and '»' in text_with_placeholders:
        result = process_quotes(text_with_placeholders)
    else:
        DIALOGUE_CUES = {
            'je', 'me', 'moi', 'mon', 'ma', 'mes', 'tu', 'te', 'toi', 'ton', 'ta', 'tes',
            'nous', 'notre', 'nos', 'vous', 'votre', 'vos', 'on'
        }
        VERBES_INTRODUCTEURS = {
            'dire', 'demander', 'répondre', 'crier', 'chuchoter', 'murmurer',
            'ajouter', 'continuer', 'reprendre', 'lancer', 'déclarer',
            'hasarder', 'souffler', 'répliquer', 'interrompre', 'rétorquer', 'siffler'
        }

        def is_passe_simple(word):
            return word.feats and 'Tense=Past' in word.feats and 'VerbForm=Fin' in word.feats

        def is_imparfait(word):
            return word.feats and 'Tense=Imp' in word.feats and 'VerbForm=Fin' in word.feats

        def is_present(word):
            return word.feats and 'Tense=Pres' in word.feats and 'VerbForm=Fin' in word.feats

        def is_dialogue_segment(text_seg, words):
            if not text_seg.strip():
                return False
            if '?' in text_seg or '!' in text_seg:
                return True
            if any(w.text.lower() in DIALOGUE_CUES for w in words):
                return True
            has_imp = any(is_imparfait(w) for w in words)
            has_pres = any(is_present(w) for w in words)
            if has_imp and not has_pres:
                return False
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

            narr_anchor = next(
                (w for w in words
                 if w.lemma.lower() in VERBES_INTRODUCTEURS
                 or is_passe_simple(w) or is_imparfait(w)),
                None
            )

            if narr_anchor and any(m in sent_text for m in ['–', '—']):
                anchor_token = next(
                    t for t in sent.tokens
                    if any(w.id == narr_anchor.id for w in t.words)
                )
                break_point = anchor_token.start_char
                rel_break = break_point - s_start
                part1_text = sent_text[:rel_break].rstrip()
                part2_text = sent_text[rel_break:].lstrip()

                p1_words = [
                    w for w in words
                    if next(t for t in sent.tokens
                            if any(word.id == w.id for word in t.words)).start_char < break_point
                ]
                p2_words = [
                    w for w in words
                    if next(t for t in sent.tokens
                            if any(word.id == w.id for word in t.words)).start_char >= break_point
                ]

                tag1 = "dialogue" if is_dialogue_segment(part1_text, p1_words) else "narration"
                tag2 = (
                    "narration"
                    if (narr_anchor.lemma.lower() in VERBES_INTRODUCTEURS
                        or is_passe_simple(narr_anchor) or is_imparfait(narr_anchor))
                    else "dialogue"
                )
                final_output.append(f"<{tag1}>{part1_text}</{tag1}> <{tag2}>{part2_text}</{tag2}>")
            else:
                tag = "dialogue" if is_dialogue_segment(sent_text, words) else "narration"
                final_output.append(f"<{tag}>{sent_text}</{tag}>")

        result = "\n".join(final_output)

    for i, original_tag in enumerate(ppi_contents):
        result = result.replace(f"__PPI_{i}__", original_tag)

    return result
