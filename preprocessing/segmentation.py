# preprocessing/segmentation.py

import re
import logging
logger = logging.getLogger(__name__)

def detect_segments_ia(text,model,misc=None):
	# fix 6/3 if text has no quotes or dialogue cues
        if not (re.search(r'[«"''][^»"'']*[»"'']|[—–-]\s*\w+|«[^»]*»',text)):
            return "<dialogue>"+"[Locuteur1] "+text+"</dialogue>"

        # fixe 6/3 10:47 ajout des indices formels, indices implicites, verbes intro rares, combination tour parole avec exemple
        # fix 6/3 20:42 Le segment contenant les balises <PPI> est automatiquement un dialogue
        # fix 6/3 20:42 Please process the text without removing the <PPI></PPI> tags or moving them
        # fix 6/3 20:42 Si l'extrait ne contient aucun signe formel de changement de tour de paroles: tiret,guillemets, il s'agit alors d'un seul tour de parole que tu doit attribuer à un seul locuteur
        # fix 6/3 changed whole prompt with deepseek deepthink
        # fix 11/3 changed whole prompt with deepseek (restructured prompt with md)
        system_prompt =f""" 
        Prompt_Formatage_conversation
Tu es un expert en analyse linguistique et en traitement automatique du langage naturel. Ta mission est d'analyser un texte en français et d'y identifier deux types de segments distincts :

1.  **Segments de dialogue** : paroles prononcées par les personnages (répliques, questions, exclamations, etc.), y compris les incises de dialogue lorsqu'elles sont intégrées.
2.  **Segments de narration** : tout ce qui n'est pas du dialogue (description, actions, pensées non verbalisées, éléments narratifs, etc.).

## Indices linguistiques à utiliser

| Dialogue | Narration |
| :--- | :--- |
| Ponctuation expressive (! ?) | Temps du récit : passé simple, imparfait, plus-que-parfait |
| Temps verbaux du discours : présent, **passé composé (OBLIGATOIREMENT CONSIDÉRÉ COMME DIALOGUE! :e.g vous avez vu?/tu as vu? etc.)** , futur, conditionnel, impératif | Absence de ponctuation expressive |
| Présence de verbes introducteurs (dit-il, s'exclama-t-elle, etc.) | Description d'actions, de pensées non rapportées |
| Guillemets (« ») ou tirets (‑) en début de réplique | |

## Règles strictes à respecter

### 1. Préservation du texte original
- Ne supprime **aucun** élément du texte.
- N'ajoute que les balises `<dialogue>` et `<narration>` et, si nécessaire, les étiquettes de locuteurs entre crochets `[ ]`.
- Conserve intégralement la ponctuation, la mise en forme et les éventuelles balises `<PPI> </PPI>` (ne pas les déplacer ni les supprimer).

### 2. Délimitation des segments
- Encadre chaque segment de dialogue par `<dialogue> </dialogue>`.
- Encadre chaque segment de narration par `<narration> </narration>`.
- Si un segment contient à la fois narration et dialogue (ex. : *dit‑il, « je viens »*), divise‑le en segments distincts.
- Les verbes introducteurs (y compris les formes rares comme *gémit‑elle, rectifia‑t‑il, hula*) sont à conserver et à placer dans le segment auquel ils appartiennent (souvent narration).

### 3. Règles automatiques impératives
- Tout segment contenant une ponctuation expressive (**!** ou **?**) est automatiquement du dialogue.
- Le segment contenant les balises `<PPI>` est automatiquement du dialogue.
- La présence du **passé composé** (ex. : *il a regardé, elle est partie, ils ont mangé*) indique **OBLIGATOIREMENT** un dialogue. Ne jamais classer un passé composé dans la narration.
- La présence du **passé simple** (ex. : *il regarda, elle partit*) indique **OBLIGATOIREMENT** une narration.

### 4. Gestion des tours de parole
- Dans le dialogue, chaque nouveau locuteur est généralement introduit par un tiret (**‑**) en début de réplique. **Supprime ces tirets** car ils ne font pas partie du dialogue proprement dit.
- Si le texte ne contient aucun signe formel de changement de locuteur (tiret, guillemets), considère qu'il s'agit d'un seul tour de parole attribué à un locuteur unique.
- Si un locuteur enchaîne plusieurs répliques successives (par exemple avec des tirets successifs mais sans changement de personne), regroupe‑les en **un seul tour de parole** en séparant les répliques par une barre oblique (**/**).
    - *Exemple* : `[Locuteur 1] je suis content aujourd'hui / j'ai réussi !`
- Lorsqu'un verbe introducteur indique une rectification ou une hésitation du même locuteur (ex. : *se rectifia‑t‑il*), ne pas créer de nouveau locuteur.

### 5. Gestion spécifique des balises PPI
- **Cas particulier :** Lorsqu'une balise `<PPI>` est suivie immédiatement d'un trait d'union (**‑**) et d'un pronom (ex. : *‑il*, *‑elle*, *‑on*, *‑je*, *‑tu*, etc.), ces éléments (**‑** + pronom) font **partie intégrante de la PPI** et doivent être inclus **à l'intérieur** de la balise `<PPI>`.
    - *Exemple* : `<PPI> comment cela se fait</PPI>-il` → `<dialogue><PPI> comment cela se fait-il</PPI></dialogue>`
- **Cas des erreurs typographiques :** Il peut arriver que, pour des raisons typographiques (espace manquante, erreur de formatage), le trait d'union soit absent entre la balise PPI et le pronom qui la suit. Dans ce cas, si la structure syntaxique indique clairement que le pronom fait partie intégrante de la PPI (par exemple, dans une forme interrogative inversée comme *"comment cela se fait il"* au lieu de *"comment cela se fait-il"*), le pronom doit néanmoins être inclus dans la PPI et placé à l'intérieur de la balise `<PPI>`.
    - *Exemple* : `<PPI> comment cela se fait</PPI> il` (avec espace) → `<dialogue><PPI> comment cela se fait il</PPI></dialogue>`
- Dans tous les autres cas, le trait d'union et les éléments qui suivent une balise PPI doivent être examinés avec attention pour déterminer s'ils appartiennent au dialogue (parole rapportée) ou à la narration (incise).

### 6. Identification des locuteurs
- Si le nom du locuteur est identifiable dans le texte, place‑le entre crochets devant la réplique (ex. : `[Jean] bonjour`).
- Sinon, attribue des étiquettes numérotées : `[Locuteur 1]`, `[Locuteur 2]`, etc., dans l'ordre d'apparition.
- Conserve les guillemets français (**« »**) à l'intérieur des balises `<dialogue>`.

### 7. Cas particuliers
- Les incises de dialogue (ex. : *« Je viens », dit‑il*) sont à traiter ainsi : le dialogue inclut les paroles avec leurs guillemets ; l'incise est placée dans la narration.
- Si un passage contient des guillemets ouvrants sans fermants (ou l'inverse), interprète‑le comme du dialogue jusqu'à la fin logique du discours.
- Les appellatifs placés avant le tour de parole font partie du dialogue (ex. : *‑ Monsieur Jean, la vie est belle* → `[Locuteur] Monsieur Jean, la vie est belle`).

## Format de sortie
- La réponse doit être un **texte brut**, sans aucun formatage supplémentaire (pas de markdown, pas de code, pas d'explications).
- Seuls les ajouts autorisés sont les balises et, éventuellement, les étiquettes de locuteurs.

---

### Exemple de transformation

**Entrée :**
« Je n'en sais rien », répondit-il en haussant les épaules. Puis il ajouta : « Peut-être devrions-nous partir. »

**Sortie attendue :**
<dialogue>[Jean]« Je n'en sais rien »</dialogue> <narration>, répondit Jean en haussant les épaules.</narration> <narration>Puis il ajouta :</narration> <dialogue>« Peut-être devrions-nous partir. »</dialogue>

**Entrée avec PPI (trait d'union présent) :**
<PPI> comment cela se fait</PPI>-il demanda-t-il.

**Sortie attendue :**
<dialogue><PPI> comment cela se fait</PPI>-il</dialogue> <narration>demanda-t-il.</narration>

**Entrée avec PPI (erreur typographique) :**
<PPI> comment cela se fait</PPI> il demanda-t-il.

**Sortie attendue :**
<dialogue><PPI> comment cela se fait il</PPI></dialogue> <narration>demanda-t-il.</narration>

**Entrée avec passé composé :**
Il a décidé de partir. « Pourquoi tant d'hésitation ? » demanda-t-elle.

**Sortie attendue :**
<dialogue>[Locuteur 1]Il a décidé de partir.</dialogue> <narration>« Pourquoi tant d'hésitation ? » demanda-t-elle.</narration>

---

### Texte à analyser
Analyse maintenant le texte suivant et produis une sortie avec les balises `<dialogue>` et `<narration>` :

{text}
"""
### Texte à analyser
        prompt = f"""
Analyse maintenant le texte suivant et produis une sortie avec les balises `<dialogue>` et `<narration>` :

{text}

        """
        from ppi_analyser.models.factory import get_provider
        provider = get_provider(model.split('_')[0], model.split('_')[1])
        result = provider.complete(system_prompt, prompt)
        return result.replace("—", "")        
        

def detect_segments_ia_batch( texts, model, misc=None):
        log("Detecting segments in batch mode")
        """
        Batch version of detect_segments_ia that processes multiple texts at once.
        
        Args:
            texts: List of text strings to analyze
            model: Model identifier string
            misc: Optional miscellaneous parameter
        
        Returns:
            List of processed strings, each containing the text with <dialogue> and <narration> tags
        """
        if not texts:
            return []
        
        # Apply the simple rule for texts without quotes or dialogue cues
        processed_texts = []
        batch_texts = []
        batch_indices = []
        
        for i, text in enumerate(texts):
            # fix 6/3 if text has no quotes or dialogue cues
            if not (re.search(r'[«"''][^»"'']*[»"'']|[—–-]\s*\w+|«[^»]*»', text)):
                processed_texts.append("<dialogue>" + "[Locuteur1] " + text + "</dialogue>")
            else:
                # Add to batch for processing
                batch_texts.append(text)
                batch_indices.append(i)
                processed_texts.append(None)  # Placeholder
        
        # If there are texts to process with the model
        if batch_texts:
            system_prompt = """ 
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

    ### 2. Délimitation des segments
    - Encadre chaque segment de dialogue par `<dialogue> </dialogue>`.
    - Encadre chaque segment de narration par `<narration> </narration>`.
    - Si un segment contient à la fois narration et dialogue (ex. : *dit‑il, « je viens »*), divise‑le en segments distincts.
    - Les verbes introducteurs (y compris les formes rares comme *gémit‑elle, rectifia‑t‑il, hula*) sont à conserver et à placer dans le segment auquel ils appartiennent (souvent narration).

    ### 3. Règles automatiques
    - Tout segment contenant une ponctuation expressive (**!** ou **?**) est automatiquement du dialogue.
    - Le segment contenant les balises `<PPI>` est automatiquement du dialogue.
    - La présence du **passé composé** indique un dialogue (ex. : *il a regardé, elle est partie*).
    - La présence du **passé simple** indique une narration (ex. : *il regarda, elle partit*).

    ### 4. Gestion des tours de parole
    - Dans le dialogue, chaque nouveau locuteur est généralement introduit par un tiret (**‑**) en début de réplique. **Supprime ces tirets** car ils ne font pas partie du dialogue proprement dit.
    - Si le texte ne contient aucun signe formel de changement de locuteur (tiret, guillemets), considère qu'il s'agit d'un seul tour de parole attribué à un locuteur unique.
    - Si un locuteur enchaîne plusieurs répliques successives (par exemple avec des tirets successifs mais sans changement de personne), regroupe‑les en **un seul tour de parole** en séparant les répliques par une barre oblique (**/**).
        - *Exemple* : `[Locuteur 1] je suis content aujourd'hui / j'ai réussi !`
    - Lorsqu'un verbe introducteur indique une rectification ou une hésitation du même locuteur (ex. : *se rectifia‑t‑il*), ne pas créer de nouveau locuteur.

    ### 5. Identification des locuteurs
    - Si le nom du locuteur est identifiable dans le texte, place‑le entre crochets devant la réplique (ex. : `[Jean] bonjour`).
    - Sinon, attribue des étiquettes numérotées : `[Locuteur 1]`, `[Locuteur 2]`, etc., dans l'ordre d'apparition.
    - Conserve les guillemets français (**« »**) à l'intérieur des balises `<dialogue>`.

    ### 6. Cas particuliers
    - Les incises de dialogue (ex. : *« Je viens », dit‑il*) sont à traiter ainsi : le dialogue inclut les paroles avec leurs guillemets ; l'incise est placée dans la narration.
    - Si un passage contient des guillemets ouvrants sans fermants (ou l'inverse), interprète‑le comme du dialogue jusqu'à la fin logique du discours.

    ### 7. Attention:
    - Attention aux appellatifs qui peuvent être avant le tour du parole, ils font partie du dialogue: e.g: - Monsieur Jean, la vie est belle 

    ## FORMAT DE SORTIE - TRÈS IMPORTANT

    Tu dois traiter TOUS les textes fournis et retourner UNE SEULE réponse contenant TOUS les résultats.

    Pour chaque texte analysé, tu dois:
    1. Appliquer toutes les règles ci-dessus
    2. Encadrer le résultat complet avec des balises <TEXTE> </TEXTE>
    3. Séparer chaque texte traité avec la chaîne exacte: "===SEPARATOR===" (sans les guillemets)

    Exemple de format de sortie:
    <TEXTE>[texte 1 traité avec ses balises dialogue/narration]</TEXTE>
    ===SEPARATOR===
    <TEXTE>[texte 2 traité avec ses balises dialogue/narration]</TEXTE>
    ===SEPARATOR===
    <TEXTE>[texte 3 traité avec ses balises dialogue/narration]</TEXTE>

    NE DONNE AUCUNE EXPLICATION SUPPLÉMENTAIRE. Retourne uniquement les résultats formatés comme ci-dessus.
    ---

    """
            
            # Prepare batch prompt with all texts
            batch_prompt = "Analyse maintenant les textes suivants et produis une sortie avec les balises selon le format spécifié:\n\n"
            
            for i, text in enumerate(batch_texts):
                batch_prompt += f"--- TEXTE {i+1} ---\n{text}\n\n"
            
            # Get batch result from model
            batch_result = use_model(
                system_prompt=system_prompt,
                prompt=batch_prompt,
                model=model.split('_')[0],
                submodel=model.split('_')[1],
                conv=None,
                expression=None,
                sent_index=None,
                misc=misc,
                mode="écrit"
            )
            
            # Parse the batch result
            parsed_results = _parse_batch_result(batch_result)
            
            # Fill in the processed texts at the correct indices
            result_index = 0
            for i in range(len(processed_texts)):
                if processed_texts[i] is None and result_index < len(parsed_results):
                    processed_texts[i] = parsed_results[result_index]
                    result_index += 1
        
        # Remove the placeholder None values (shouldn't happen if parsing worked)
        #print(processed_texts)
        
        return [text for text in processed_texts if text is not None]


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
