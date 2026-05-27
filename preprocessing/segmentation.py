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
Prompt_Formatage_conversation
Tu es un expert en analyse linguistique et en traitement automatique du langage naturel. Ta mission est d'analyser un texte en français et d'y identifier deux types de segments distincts :

1.  **Segments de dialogue** : paroles prononcées par les personnages (répliques, questions, exclamations, etc.), y compris les incises de dialogue lorsqu'elles sont intégrées.
2.  **Segments de narration** : tout ce qui n'est pas du dialogue (description, actions, pensées non verbalisées, éléments narratifs, etc.).

## Indices linguistiques à utiliser

| Dialogue | Narration |
| :--- | :--- |
| Ponctuation expressive (! ?) | Temps du récit : passé simple, imparfait, plus-que-parfait |
| Temps verbaux du discours : présent, **passé composé (OBLIGATOIREMENT CONSIDÉRÉ COMME DIALOGUE!)**, futur, conditionnel, impératif | Absence de ponctuation expressive |
| Présence de verbes introducteurs (dit-il, s'exclama-t-elle, etc.) | Description d'actions, de pensées non rapportées |
| Guillemets (« ») ou tirets (‑) en début de réplique | |

## Règles strictes à respecter

### 1. Préservation du texte original
- Ne supprime **aucun** élément du texte.
- N'ajoute que les balises `<dialogue>` et `<narration>` et, si nécessaire, les étiquettes de locuteurs entre crochets `[ ]`.
- Conserve intégralement la ponctuation, la mise en forme et les éventuelles balises `<PPI> </PPI>` (ne pas les déplacer ni les supprimer).
- N'ajoute pas des balises <PPI> qui n'existaient pas dans le texte d'entrée.
### 2. Délimitation des segments
- Encadre chaque segment de dialogue par `<dialogue> </dialogue>`.
- Encadre chaque segment de narration par `<narration> </narration>`.
- Si un segment contient à la fois narration et dialogue (ex. : *dit‑il, « je viens »*), divise‑le en segments distincts.
- Les verbes introducteurs (y compris les formes rares comme *gémit‑elle, rectifia‑t‑il, hula*) sont à conserver et à placer dans le segment auquel ils appartiennent (souvent narration).

### 3. Règles automatiques impératives
- Tout segment contenant une ponctuation expressive (**!** ou **?**) est automatiquement du dialogue.
- Le segment contenant les balises `<PPI>` est automatiquement du dialogue.
- La présence du **passé composé** indique **OBLIGATOIREMENT** un dialogue. Ne jamais classer un passé composé dans la narration.
- La présence du **passé simple** indique **OBLIGATOIREMENT** une narration.

### 4. Gestion des tours de parole
- Dans le dialogue, chaque nouveau locuteur est généralement introduit par un tiret (**‑**) en début de réplique. **Supprime ces tirets** car ils ne font pas partie du dialogue proprement dit.
- Si le texte ne contient aucun signe formel de changement de locuteur (tiret, guillemets), considère qu'il s'agit d'un seul tour de parole attribué à un locuteur unique.
- Si un locuteur enchaîne plusieurs répliques successives, regroupe‑les en **un seul tour de parole** en séparant les répliques par une barre oblique (**/**).
- Lorsqu'un verbe introducteur indique une rectification ou une hésitation du même locuteur, ne pas créer de nouveau locuteur.

### 5. Gestion spécifique des balises PPI
- Lorsqu'une balise `<PPI>` est suivie immédiatement d'un trait d'union (**‑**) et d'un pronom sujet inversé (*-il*, *-elle*, *-on*, *-je*, *-tu*, *-ils*, *-elles*, *-nous*, *-vous*), ces éléments font **partie intégrante de la PPI** et doivent être inclus **à l'intérieur** de la balise `<PPI>`.
  - *Exemple* : `<PPI>comment ça se fait</PPI>-il` → `<PPI>comment ça se fait-il</PPI>`
- Lorsqu'une balise `<PPI>` est suivie d'un pronom sujet inversé **sans trait d'union** (erreur typographique ou absence de ponctuation), le pronom doit néanmoins être inclus à l'intérieur de la balise `<PPI>` si la structure syntaxique indique clairement une interrogation inversée.
  - *Exemple* : `<PPI>comment ça se fait</PPI> il` → `<PPI>comment ça se fait il</PPI>`
- **RÈGLE ABSOLUE** : Après une balise `</PPI>`, si le mot suivant est un pronom sujet parmi (*il, elle, on, je, tu, ils, elles, nous, vous*), ce pronom appartient toujours à la PPI. Ne jamais laisser un pronom isolé immédiatement après `</PPI>`.

### 6. Identification des locuteurs
- Si le nom du locuteur est identifiable dans le texte, place‑le entre crochets devant la réplique (ex. : `[Jean] bonjour`).
- Sinon, attribue des étiquettes numérotées : `[Locuteur 1]`, `[Locuteur 2]`, etc., dans l'ordre d'apparition.
- Conserve les guillemets français (**« »**) à l'intérieur des balises `<dialogue>`.

### 7. Cas particuliers
- Les incises de dialogue sont à traiter ainsi : le dialogue inclut les paroles avec leurs guillemets ; l'incise est placée dans la narration.
- Les appellatifs placés avant le tour de parole font partie du dialogue.

## Format de sortie
- La réponse doit être un **texte brut**, sans aucun formatage supplémentaire.
- Seuls les ajouts autorisés sont les balises et, éventuellement, les étiquettes de locuteurs.
"""

    prompt = (
        "Analyse maintenant le texte suivant et produis une sortie "
        "avec les balises `<dialogue>` et `<narration>` :\n\n" + text
    )

    from ppi_analyser.models.factory import get_provider
    provider = get_provider(model.split('_')[0], model.split('_')[1])
    result = provider.complete(system_prompt, prompt)
    result = result.replace("—", "")

    cache_set(text, model, result)
    return result


_BATCH_SYSTEM_PROMPT = """\
Prompt_Formatage_conversation
Tu es un expert en analyse linguistique et en traitement automatique du langage naturel. Ta mission est d'analyser un texte en français et d'y identifier deux types de segments distincts :

1.  **Segments de dialogue** : paroles prononcées par les personnages (répliques, questions, exclamations, etc.), y compris les incises de dialogue lorsqu'elles sont intégrées.
2.  **Segments de narration** : tout ce qui n'est pas du dialogue (description, actions, pensées non verbalisées, éléments narratifs, etc.).

## Indices linguistiques à utiliser

| Dialogue | Narration |
| :--- | :--- |
| Ponctuation expressive (! ?) | Temps du récit : passé simple, imparfait, plus-que-parfait |
| Temps verbaux du discours : présent, passé composé, futur, conditionnel, impératif | Absence de ponctuation expressive |
| Présence de verbes introducteurs (dit-il, s'exclama-t-elle, etc.) | Description d'actions, de pensées non rapportées |
| Guillemets (« ») ou tirets (‑) en début de réplique | |

## Règles strictes à respecter

### 1. Préservation du texte original
- Ne supprime **aucun** élément du texte.
- N'ajoute que les balises `<dialogue>` et `<narration>` et, si nécessaire, les étiquettes de locuteurs entre crochets `[ ]`.
- Conserve intégralement la ponctuation, la mise en forme et les éventuelles balises `<PPI> </PPI>` (ne pas les déplacer ni les supprimer).
- N'ajoute pas des balises <PPI> qui n'existaient pas dans le texte d'entrée.
### 2. Délimitation des segments
- Encadre chaque segment de dialogue par `<dialogue> </dialogue>`.
- Encadre chaque segment de narration par `<narration> </narration>`.
- Si un segment contient à la fois narration et dialogue, divise‑le en segments distincts.

### 3. Règles automatiques impératives
- Tout segment contenant une ponctuation expressive (**!** ou **?**) est automatiquement du dialogue.
- Le segment contenant les balises `<PPI>` est automatiquement du dialogue.
- La présence du **passé composé** indique **OBLIGATOIREMENT** un dialogue.
- La présence du **passé simple** indique **OBLIGATOIREMENT** une narration.

### 4. Gestion des tours de parole
- Supprime les tirets de début de réplique.
- Sans signe formel de changement de locuteur : un seul tour de parole, un seul locuteur.
- Répliques successives du même locuteur : regroupe‑les avec **/** comme séparateur.

### 5. Gestion spécifique des balises PPI
- Lorsqu'une balise `<PPI>` est suivie d'un trait d'union (**‑**) et d'un pronom sujet inversé (*-il*, *-elle*, *-on*, *-je*, *-tu*, *-ils*, *-elles*, *-nous*, *-vous*), ces éléments font **partie intégrante de la PPI** et doivent être inclus **à l'intérieur** de la balise `<PPI>`.
  - *Exemple* : `<PPI>comment ça se fait</PPI>-il` → `<PPI>comment ça se fait-il</PPI>`
- Lorsqu'une balise `<PPI>` est suivie d'un pronom sujet inversé **sans trait d'union**, le pronom doit néanmoins être inclus à l'intérieur de la balise `<PPI>`.
  - *Exemple* : `<PPI>comment ça se fait</PPI> il` → `<PPI>comment ça se fait il</PPI>`
- **RÈGLE ABSOLUE** : Après `</PPI>`, si le mot suivant est un pronom parmi (*il, elle, on, je, tu, ils, elles, nous, vous*), ce pronom appartient toujours à la PPI.

### 6. Identification des locuteurs
- Nom identifiable → entre crochets (ex. `[Jean]`).
- Sinon → `[Locuteur 1]`, `[Locuteur 2]`, etc.

### 7. Cas particuliers
- Les appellatifs placés avant le tour de parole font partie du dialogue.
- Les incises sont placées dans la narration.

## FORMAT DE SORTIE — TRÈS IMPORTANT
Tu dois traiter TOUS les textes fournis et retourner UNE SEULE réponse.
Pour chaque texte :
1. Applique toutes les règles ci-dessus.
2. Encadre le résultat avec `<TEXTE> </TEXTE>`.
3. Sépare chaque résultat par `===SEPARATOR===` (exactement, sans guillemets).

Exemple :
<TEXTE>[texte 1 traité]</TEXTE>
===SEPARATOR===
<TEXTE>[texte 2 traité]</TEXTE>

NE DONNE AUCUNE EXPLICATION SUPPLÉMENTAIRE.
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
