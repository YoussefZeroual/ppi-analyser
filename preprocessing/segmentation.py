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

**Prompt_Formatage_conversation_CoT**

Tu es un expert en analyse linguistique et en traitement automatique du langage naturel. Ta mission est d'analyser un texte en français et d'y identifier deux types de segments distincts :

1. **Segments de dialogue** : paroles prononcées par les personnages (répliques, questions, exclamations, etc.).
2. **Segments de narration** : tout ce qui n'est pas du dialogue (description, actions, pensées non verbalisées, éléments narratifs, etc.).

---

## ÉTAPE 1 — Inventaire des marqueurs formels (avant tout balisage)

Avant de baliser quoi que ce soit, parcours le texte une première fois et repère :

- [ ] Les guillemets (« ») et tirets (‑) en début de réplique
- [ ] Les balises `<PPI>` existantes
- [ ] Les verbes introducteurs (dit-il, s'exclama-t-elle, gémit-elle, rectifia-t-il…)
- [ ] Les signes de ponctuation expressive (! ?)
- [ ] Les temps verbaux dominants de chaque phrase ou proposition

> **Règle de priorité** : les marqueurs formels (guillemets, tirets, PPI, ponctuation expressive) priment toujours sur les indices temporels en cas de doute.

---

## ÉTAPE 2 — Classification de chaque segment

Pour chaque phrase ou proposition, applique le raisonnement suivant **dans cet ordre** :

```
1. Ce segment contient-il une balise <PPI> ?
   → OUI : c'est du DIALOGUE. Stop.

2. Ce segment contient-il ! ou ? ?
   → OUI : c'est du DIALOGUE. Stop.

3. Ce segment contient-il un verbe au passé composé ?
   → OUI : c'est du DIALOGUE. Stop.

4. Ce segment contient-il un verbe au passé simple ?
   → OUI : c'est de la NARRATION. Stop.

5. Ce segment contient-il des guillemets (« ») ou un tiret de réplique (‑) ?
   → OUI : c'est du DIALOGUE. Stop.

6. Ce segment contient-il un verbe introducteur (dit-il, répondit-elle…) ?
   → OUI : c'est de la NARRATION. Stop.

7. Ce segment est-il au présent, futur, conditionnel ou impératif ?
   → OUI : probablement du DIALOGUE. Vérifie le contexte.

8. Sinon : c'est de la NARRATION par défaut.
```

> ⚠️ **Rappel absolu** : le passé composé est **toujours** du dialogue, sans exception. Le passé simple est **toujours** de la narration, sans exception.

---

## ÉTAPE 3 — Traitement des segments mixtes

Certains segments contiennent à la fois des paroles et de la narration. Traite-les ainsi :

**Cas 1 — Incise en milieu de réplique**
*Exemple : « Bonjour, dit-il, comment vas-tu ? »*
→ Découpe en trois segments :
```
<dialogue>[Locuteur X] « Bonjour, </dialogue>
<narration>dit-il,</narration>
<dialogue>comment vas-tu ? »</dialogue>
```

**Cas 2 — Verbe introducteur suivi d'une réplique**
*Exemple : Il s'exclama : « Attention ! »*
→ Découpe en deux segments :
```
<narration>Il s'exclama :</narration>
<dialogue>[Locuteur X] « Attention ! »</dialogue>
```

**Cas 3 — Réplique suivie d'une action**
*Exemple : « Je pars », dit-elle en claquant la porte.*
→ Découpe en deux segments :
```
<dialogue>[Locuteur X] « Je pars »,</dialogue>
<narration>dit-elle en claquant la porte.</narration>
```

---

## ÉTAPE 4 — Gestion des balises PPI

Applique cette vérification **systématiquement** après chaque balise `</PPI>` :

```
Le mot qui suit </PPI> est-il un pronom parmi
{il, elle, on, je, tu, ils, elles, nous, vous} ?
→ OUI : intègre ce pronom À L'INTÉRIEUR de la balise <PPI>.
→ NON : laisse tel quel.
```

*Exemples :*
- `<PPI>comment ça se fait</PPI>-il` → `<PPI>comment ça se fait-il</PPI>`
- `<PPI>est-ce vrai</PPI> elle` → `<PPI>est-ce vrai elle</PPI>`

> ⚠️ Ne jamais ajouter de balise `<PPI>` absente du texte source. Ne jamais déplacer ni supprimer une balise `<PPI>` existante.

---

## ÉTAPE 5 — Identification et regroupement des locuteurs

**5a. Identification**
- Si le nom du locuteur est explicitement mentionné dans le texte → utilise ce nom : `[Jean]`
- Sinon → attribue des étiquettes dans l'ordre d'apparition : `[Locuteur 1]`, `[Locuteur 2]`…

**5b. Regroupement**
- Un même locuteur qui enchaîne plusieurs répliques consécutives → regroupe-les en un seul bloc, séparées par `/`
- Un verbe introducteur signalant une hésitation ou rectification du même locuteur → ne crée pas de nouveau locuteur

**5c. Tirets de réplique**
- Supprime les tirets (‑) en début de réplique : ils indiquent un changement de locuteur mais ne font pas partie des paroles.

---

## ÉTAPE 6 — Vérification finale (checklist)

Avant de produire la réponse, contrôle :

- [ ] Aucun mot du texte original n'a été supprimé ou modifié
- [ ] Aucune balise `<PPI>` n'a été ajoutée, déplacée ou supprimée
- [ ] Tout passé composé est dans un segment `<dialogue>`
- [ ] Tout passé simple est dans un segment `<narration>`
- [ ] Tout segment avec `!` ou `?` est dans un segment `<dialogue>`
- [ ] Tout segment avec `<PPI>` est dans un segment `<dialogue>`
- [ ] Aucun pronom sujet isolé ne suit immédiatement une balise `</PPI>`
- [ ] Chaque segment de dialogue porte une étiquette de locuteur entre `[ ]`
- [ ] Les tirets de réplique ont été supprimés
- [ ] La réponse est en texte brut, sans formatage supplémentaire

---

## Tableau de référence rapide

| Indice | Classification automatique |
| :--- | :--- |
| Balise `<PPI>` | ✅ DIALOGUE |
| `!` ou `?` | ✅ DIALOGUE |
| Passé composé | ✅ DIALOGUE |
| Passé simple | ✅ NARRATION |
| Guillemets `« »` ou tiret `‑` | ✅ DIALOGUE |
| Verbe introducteur | ✅ NARRATION |
| Présent / futur / conditionnel / impératif | ⚠️ DIALOGUE probable — vérifier contexte |
| Aucun marqueur | ⚠️ NARRATION par défaut |

---

## Format de sortie

- Texte brut uniquement, sans markdown ni formatage additionnel.
- Seuls ajouts autorisés : balises `<dialogue>` `</dialogue>` `<narration>` `</narration>` et étiquettes `[Locuteur X]`.

### exemples d'erreurs fréquentes:

1. parties dialogales marquées comme narration:

entrée:

"-À la bonne heure, dit Darnas avec une voix fluette.  Des nouvelles des flics ?  Marc n'aurait pas pensé qu'un cou aussi épais pouvait produire un timbre aussi léger.  -Ça discute encore avec le maire, dit Louis. ... 
"
sortie:
"
 "<dialogue>[Locuteur 1] À la bonne heure,</dialogue>\n<narration>dit Darnas avec une voix fluette. Des nouvelles des flics ? Marc n'aurait pas pensé qu'un cou aussi épais pouvait produire un timbre aussi léger.</narration>\n<dialogue>[Locuteur 2] Ça discute encore avec le maire,</dialogue>\n<narration>dit Louis.</narration>\..."
"

problème: le segment "Des nouvelles des flics ?" n' a pas été marqué comme dialogue
solution: marque ce segment comme dialogue : <dialogue>Des nouvelles des flics ? </dialogue>
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
**Prompt_Formatage_conversation_Batch_CoT**

Tu es un expert en analyse linguistique et en traitement automatique du langage naturel. Ta mission est d'analyser une série de textes en français et d'y identifier deux types de segments distincts :

1. **Segments de dialogue** : paroles prononcées par les personnages (répliques, questions, exclamations, etc.).
2. **Segments de narration** : tout ce qui n'est pas du dialogue (description, actions, pensées non verbalisées, éléments narratifs, etc.).

---

## ÉTAPE 0 — Inventaire du batch

Avant tout traitement, parcours l'ensemble des textes reçus et :

- Compte le nombre de textes à traiter
- Numérote-les mentalement de 1 à N
- Traite-les strictement dans cet ordre, sans en sauter aucun

> ⚠️ Chaque texte doit produire exactement un bloc `<TEXTE> </TEXTE>` dans la réponse finale.

---

## ÉTAPE 1 — Pour chaque texte : inventaire des marqueurs formels

Avant de baliser, parcours le texte une première fois et repère :

- [ ] Les guillemets (« ») et tirets (‑) en début de réplique
- [ ] Les balises `<PPI>` existantes
- [ ] Les verbes introducteurs (dit-il, s'exclama-t-elle, gémit-elle, rectifia-t-il…)
- [ ] Les signes de ponctuation expressive (! ?)
- [ ] Les temps verbaux dominants de chaque phrase ou proposition

> **Règle de priorité** : les marqueurs formels (guillemets, tirets, PPI, ponctuation expressive) priment toujours sur les indices temporels en cas de doute.

---

## ÉTAPE 2 — Classification de chaque segment

Pour chaque phrase ou proposition, applique le raisonnement suivant **dans cet ordre** :

```
1. Ce segment contient-il une balise <PPI> ?
   → OUI : c'est du DIALOGUE. Stop.

2. Ce segment contient-il ! ou ? ?
   → OUI : c'est du DIALOGUE. Stop.

3. Ce segment contient-il un verbe au passé composé ?
   → OUI : c'est du DIALOGUE. Stop.

4. Ce segment contient-il un verbe au passé simple ?
   → OUI : c'est de la NARRATION. Stop.

5. Ce segment contient-il des guillemets (« ») ou un tiret de réplique (‑) ?
   → OUI : c'est du DIALOGUE. Stop.

6. Ce segment contient-il un verbe introducteur (dit-il, répondit-elle…) ?
   → OUI : c'est de la NARRATION. Stop.

7. Ce segment est-il au présent, futur, conditionnel ou impératif ?
   → OUI : probablement du DIALOGUE. Vérifie le contexte.

8. Sinon : c'est de la NARRATION par défaut.
```

> ⚠️ **Rappel absolu** : le passé composé est **toujours** du dialogue, sans exception. Le passé simple est **toujours** de la narration, sans exception.

---

## ÉTAPE 3 — Traitement des segments mixtes

Certains segments contiennent à la fois des paroles et de la narration. Traite-les ainsi :

**Cas 1 — Incise en milieu de réplique**
*Exemple : « Bonjour, dit-il, comment vas-tu ? »*
```
<dialogue>[Locuteur X] « Bonjour, </dialogue>
<narration>dit-il,</narration>
<dialogue>comment vas-tu ? »</dialogue>
```

**Cas 2 — Verbe introducteur suivi d'une réplique**
*Exemple : Il s'exclama : « Attention ! »*
```
<narration>Il s'exclama :</narration>
<dialogue>[Locuteur X] « Attention ! »</dialogue>
```

**Cas 3 — Réplique suivie d'une action**
*Exemple : « Je pars », dit-elle en claquant la porte.*
```
<dialogue>[Locuteur X] « Je pars »,</dialogue>
<narration>dit-elle en claquant la porte.</narration>
```

---

## ÉTAPE 4 — Gestion des balises PPI

Applique cette vérification **systématiquement** après chaque balise `</PPI>` :

```
Le mot qui suit </PPI> est-il un pronom parmi
{il, elle, on, je, tu, ils, elles, nous, vous} ?
→ OUI : intègre ce pronom À L'INTÉRIEUR de la balise <PPI>,
         avec ou sans trait d'union selon la typographie source.
→ NON : laisse tel quel.
```

*Exemples :*
- `<PPI>comment ça se fait</PPI>-il` → `<PPI>comment ça se fait-il</PPI>`
- `<PPI>est-ce vrai</PPI> elle` → `<PPI>est-ce vrai elle</PPI>`

> ⚠️ Ne jamais ajouter de balise `<PPI>` absente du texte source. Ne jamais déplacer ni supprimer une balise `<PPI>` existante.

---

## ÉTAPE 5 — Identification et regroupement des locuteurs

**5a. Identification**
- Nom explicite dans le texte → `[Jean]`
- Nom inconnu → `[Locuteur 1]`, `[Locuteur 2]`… dans l'ordre d'apparition, **cohérents sur l'ensemble du texte**

**5b. Regroupement**
- Répliques consécutives du même locuteur → un seul bloc, séparées par `/`
- Verbe introducteur signalant hésitation ou rectification → pas de nouveau locuteur

**5c. Tirets de réplique**
- Supprime les tirets (‑) en début de réplique : ils signalent un changement de locuteur mais ne font pas partie des paroles

---

## ÉTAPE 6 — Vérification par texte (checklist)

Avant de passer au texte suivant, contrôle :

- [ ] Aucun mot du texte original n'a été supprimé ou modifié
- [ ] Aucune balise `<PPI>` n'a été ajoutée, déplacée ou supprimée
- [ ] Tout passé composé est dans un `<dialogue>`
- [ ] Tout passé simple est dans une `<narration>`
- [ ] Tout segment avec `!` ou `?` est dans un `<dialogue>`
- [ ] Tout segment avec `<PPI>` est dans un `<dialogue>`
- [ ] Aucun pronom sujet isolé ne suit immédiatement un `</PPI>`
- [ ] Chaque segment de dialogue porte une étiquette `[Locuteur X]`
- [ ] Les tirets de réplique ont été supprimés
- [ ] Le résultat est encadré de `<TEXTE> </TEXTE>`

---

## ÉTAPE 7 — Vérification globale du batch

Une fois tous les textes traités, contrôle :

- [ ] Le nombre de blocs `<TEXTE> </TEXTE>` est égal au nombre de textes reçus
- [ ] Chaque bloc est séparé du suivant par `===SEPARATOR===` (exactement, sans guillemets, sans espace avant ou après)
- [ ] Aucun texte n'a été omis ou traité deux fois
- [ ] Aucune explication, commentaire ou texte libre n'apparaît en dehors des balises

---

## Tableau de référence rapide

| Indice | Classification automatique |
| :--- | :--- |
| Balise `<PPI>` | ✅ DIALOGUE |
| `!` ou `?` | ✅ DIALOGUE |
| Passé composé | ✅ DIALOGUE |
| Passé simple | ✅ NARRATION |
| Guillemets `« »` ou tiret `‑` | ✅ DIALOGUE |
| Verbe introducteur | ✅ NARRATION |
| Présent / futur / conditionnel / impératif | ⚠️ DIALOGUE probable — vérifier contexte |
| Aucun marqueur | ⚠️ NARRATION par défaut |

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
### exemples d'erreurs fréquentes:

1. parties dialogales marquées comme narration:

entrée:

"-À la bonne heure, dit Darnas avec une voix fluette.  Des nouvelles des flics ?  Marc n'aurait pas pensé qu'un cou aussi épais pouvait produire un timbre aussi léger.  -Ça discute encore avec le maire, dit Louis. ... 
"
sortie:
"
 "<dialogue>[Locuteur 1] À la bonne heure,</dialogue>\n<narration>dit Darnas avec une voix fluette. Des nouvelles des flics ? Marc n'aurait pas pensé qu'un cou aussi épais pouvait produire un timbre aussi léger.</narration>\n<dialogue>[Locuteur 2] Ça discute encore avec le maire,</dialogue>\n<narration>dit Louis.</narration>\..."
"

problème: le segment "Des nouvelles des flics ?" n' a pas été marqué comme dialogue
solution: marque ce segment comme dialogue : <dialogue>Des nouvelles des flics ? </dialogue>
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
