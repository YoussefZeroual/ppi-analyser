# PPI Analyser

A semi-automatic pipeline for the linguistic analysis of **Interactional prefabricated sentences (phrases préfabriquées des interactions) ** (PPI) in French. The pipeline orchestrates LLM API calls and classical NLP tools (Stanza) to produce structured, human-verifiable analyses of conversational formulae according to the PREFAB project's linguistic annotation grid.

---

## Table of Contents

1. [Background and Motivation](#1-background-and-motivation)
2. [Architecture Overview](#2-architecture-overview)
3. [Installation and Requirements](#3-installation-and-requirements)
4. [Configuration](#4-configuration)
   - [Environment Variables](#41-environment-variables)
   - [PipelineConfig Parameters](#42-pipelineconfig-parameters)
5. [Input Format](#5-input-format)
6. [Analysis Modes](#6-analysis-modes)
   - [ECRIT](#61-ecrit-written-literary-dialogues)
   - [ECRIT_IA](#62-ecrit_ia-written-dialogues-with-ai-speaker-detection)
   - [ORAL](#63-oral-authentic-spoken-interactions)
   - [ECRIT_TEST](#64-ecrit_test-development-mode)
7. [Processing Modes](#7-processing-modes)
   - [Sequential (default)](#71-sequential-mode-default)
   - [Batch Mode](#72-batch-mode)
   - [Mistral Async Batch](#73-mistral-async-batch-mode)
8. [Supported Models and Providers](#8-supported-models-and-providers)
9. [Linguistic Properties Analysed](#9-linguistic-properties-analysed)
10. [Local NLP Modules](#10-local-nlp-modules)
    - [Position Detection](#101-position-detection)
    - [Expansion Detection](#102-expansion-detection)
    - [Modifier Detection](#103-modifier-detection)
11. [Prompt System](#11-prompt-system)
12. [Output Files](#12-output-files)
13. [Analysis Cache](#13-analysis-cache)
14. [Usage Examples](#14-usage-examples)
    - [Minimal Example](#141-minimal-example)
    - [Batch Mode with Mistral](#142-batch-mode-with-mistral)
    - [Oral Corpus with Ollama](#143-oral-corpus-with-ollama)
    - [Selecting a Sentence Subset](#144-selecting-a-sentence-subset)
    - [Custom Properties Only](#145-custom-properties-only)
    - [With Analysis Cache](#146-with-analysis-cache)
    - [Multiple Models](#147-multiple-models)
15. [Module Reference](#15-module-reference)
16. [Error Handling and Resuming Interrupted Runs](#16-error-handling-and-resuming-interrupted-runs)

---

## 1. Background and Motivation

Interactional prefabricated sentences (phrases préfabriquées des interactions) are formulaic conversational expressions in French (utterances such as *comment ça se fait*, *tu te rends compte*, or *c'est pas possible*) whose pragmatic and syntactic properties must be described systematically across large corpora. Manual annotation of hundreds of concordances is time-consuming; the PPI Analyser pipeline accelerates this work by delegating classification tasks to LLMs while keeping a human-in-the-loop for validation.

The pipeline implements the approach described in Morin & Marttinen Larsson (2025): LLMs (autoregressive decoder models) excel at detecting abstract linguistic properties that are beyond the reach of classical embedding models. Their main limitation is throughput, addressed here through parallelism, batch grouping, and asynchronous API calls.

Input concordances are extracted from corpus tools such as **Lexicoscope** and processed against the PREFAB project's full linguistic annotation grid: acception, sentence type, enunciation modality, syntactic properties, modifiers, co-occurrents, expansions, scope, triggering, global function, specific functions, and miscellaneous remarks.

---

## 2. Architecture Overview

```
Input (Excel/CSV concordances)
        │
        ▼
┌──────────────────────┐
│   _load_sentences    │  Loads concordances; optionally reads lemma per row
│   _validate_range    │  Applies start_sent / max_sentences / sent_list filters
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│              Preprocessing                   │
│  • clean_conv      — tag normalisation       │
│  • detect_segments — turn detection (ECRIT)  │
│  • detect_segments_ia — LLM turn detection   │
│  • fix_speaker_turns — oral correction       │
│  • detect_speakers — locuteur extraction     │
│  • _fill_nlp_preprocessed — Stanza parsing  │
└──────────┬───────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│              Analysis                        │
│  Sequential  │  Batch  │  Mistral Async Batch│
│                                              │
│  For each property:                          │
│  • NON_IA properties → local computation    │
│    (Forme, Lemme, Position, Expansion, ...)  │
│  • IA properties → LLM API call             │
│    with system_prompt + user_prompt          │
└──────────┬───────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│        Result Assembly & Export              │
│  • JSON parsing / cleaning (results.py)      │
│  • DataFrame construction (pandas)           │
│  • Excel export (simple + extended)          │
│  • PDF export (detailed report)              │
└──────────────────────────────────────────────┘
```

The main entry point is `PPIAnalyser.process_sentences(config)` in `core.py`.

---

## 3. Installation and Requirements

```bash
pip install ppi_analyser

or docker compose up -d (using the included docker-compose.yml)

```

Key dependencies:

- `stanza` — French NLP pipeline (tokenisation, POS, lemmatisation, dependency parsing)
- `mistralai` — Mistral batch API client
- `pandas`, `openpyxl` — data handling and Excel export
- `nltk` — French stemmer for modifier detection
- `pyyaml` — modifier rule loading
- `python-dotenv` — environment variable management
- `reportlab` or equivalent — PDF export

A Stanza French model must be downloaded once:

```python
import stanza
stanza.download('fr')
```

---

## 4. Configuration

### 4.1 Environment Variables

Create a `.env` file in the project root (or export to your shell):

```env
MISTRAL_API_KEY=your_mistral_key
GROQ_API_KEY=your_groq_key
DEEPSEEK_API_KEY=your_deepseek_key
GEMINI_API_KEY=your_gemini_key
OLLAMA_HOST=http://localhost:11434   # default
```

### 4.2 PipelineConfig Parameters

`PipelineConfig` is a dataclass defined in `config.py`. All parameters:

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `models` | `list[str]` | ✓ | — | List of model identifiers (see §8) |
| `expression` | `str` | ✓ | — | Target PPI lemma (e.g. `"comment ça se fait"`) |
| `sentence_file` | `str` | ✓ | — | Path to input Excel/CSV concordance file |
| `mode` | `AnalysisMode` | ✓ | — | Analysis mode (see §6) |
| `output_dir` | `str` | ✓ | — | Directory for output files |
| `expressions_from_file` | `bool` | | `False` | Read lemma per row from a `Lemme` column |
| `start_sent` | `int` | | `0` | Index of first sentence to process |
| `max_sentences` | `int\|"all"` | | `"all"` | Maximum number of sentences |
| `n_threads` | `int` | | `4` | Number of parallel threads for API calls |
| `ollama_host` | `str` | | `"localhost"` | Ollama server host |
| `max_reqs` | `int` | | `-1` | Max API requests per minute (-1 = unlimited) |
| `sent_list` | `list[int]\|None` | | `None` | Explicit list of row indices to process |
| `speaker_detection_model` | `str\|None` | | `None` | Model for AI speaker detection (required for ECRIT_IA) |
| `custom_properties` | `list[str]\|None` | | `None` | Run only these property names |
| `batch_mode` | `bool` | | `False` | Enable batch grouping of sentences per API call |
| `batch_size` | `int` | | `5` | Number of sentences per batch |
| `preprocessed_json` | `str\|None` | | `None` | Path to pre-saved preprocessing JSON (resume support) |
| `properties` | `list[str]\|None` | | `None` | Alias for custom_properties at pipeline level |
| `use_analysis_cache` | `bool` | | `False` | Enable persistent analysis cache |
| `analysis_cache_path` | `str\|None` | | `~/.ppi_analyser/analysis_cache.json` | Cache file path |
| `non_ia` | `list[int]\|None` | | `None` | Override which property indices are computed locally |
| `exporting_mode` | `str` | | `"simple"` | `"simple"` (Excel only) or `"full"` (Excel + PDF) |

---

## 5. Input Format

The pipeline expects an **Excel** (`.xlsx`) or **CSV** file with concordances in Lexicoscope format:

| left | node | right |
|---|---|---|
| – Comment ça se fait qu'ils soient pas contaminés ? je demande, intrigué. […] Enfin bref, | comment ça se fait | qu'ils attrapent pas l'infection ? |
| [...] moi j'avais un copain sans papier enfin copain une connaissance / | comment ça se fait | qu'il est si discret quoi [...] |

- **left**: left context (preceding text, possibly including dialogue turns)
- **node**: the target PPI occurrence (the pivot)
- **right**: right context (following text)

The pipeline concatenates these three columns and inserts `<PPI>...</PPI>` tags around the node automatically.

If you have multiple PPI lemmas in the same file (e.g. a multi-expression study), set `expressions_from_file=True` and add a `Lemme` column with the standard lemma for each row. The pipeline will use the appropriate lemma for each sentence.

---

## 6. Analysis Modes

### 6.1 ECRIT — Written Literary Dialogues

```python
mode=AnalysisMode.ECRIT
```

Designed for romanesque dialogue corpora where speaker turns are indicated by typographic conventions (em-dashes `–`, guillemets `«»`). The pipeline uses Stanza's dependency parser to identify turn boundaries and assigns generic speaker labels `[locuteur 1]`, `[locuteur 2]`, etc. This mode can be noisy — use `ECRIT_IA` for better results.

**No `speaker_detection_model` required.**

Example input row (left + node + right):

```
Dans la pénombre [...] Toi qui disais mépriser le luxe ,
comment as-tu pu
aménager un endroit aussi délicieux ?
```

Becomes after preprocessing:

```
[locuteur 1] Toi qui disais mépriser le luxe , <PPI>comment as-tu pu</PPI> aménager un endroit aussi délicieux ?
[locuteur 2] Ce n'est pas moi , c'est ma sœur .
```

### 6.2 ECRIT_IA — Written Dialogues with AI Speaker Detection

```python
mode=AnalysisMode.ECRIT_IA,
speaker_detection_model="deepseek_deepseek"
```

The recommended mode for literary corpora. An LLM is called to:

1. Separate narrative from dialogue.
2. Detect and label speaker turns, assigning character names from context where possible.
3. Strip stage directions and incises.

The result is a clean, labelled dialogue, identical in structure to the oral format, on which all subsequent analyses run.

**Requires `speaker_detection_model`** — any provider string accepted by the pipeline (e.g. `"deepseek_deepseek"`, `"mistral_mistral-medium-latest"`, `"ollama_mistral:7b"`).

Example — from raw literary extract to formatted dialogue:

```
Raw:
  – Toi qui disais mépriser le luxe , comment as-tu pu aménager un endroit aussi délicieux ? ?
  – Ce n'est pas moi , c'est ma sœur .

After ECRIT_IA preprocessing:
  [Grue des Nuages] Toi qui disais mépriser le luxe , <PPI>comment as tu pu</PPI> aménager un endroit aussi délicieux ?
  [Locuteur 2] Ce n'est pas moi , c'est ma sœur .
```

### 6.3 ORAL — Authentic Spoken Interactions

```python
mode=AnalysisMode.ORAL
```

For transcribed corpora (e.g. ORFEO, CLAPI) where speaker turns are already demarcated and there is no narrative to strip. The preprocessing:

1. Corrects turns that span multiple sentences by joining them with `/` separators.
2. Does not call any LLM for speaker detection.
3. Excludes properties 0, 1, and 5 (Forme, Lemme, Position) from local computation by default, since these are handled differently in oral data.

Example formatted turn (ORFEO corpus):

```
[Michel_Chevrier] moi j'avais un copain sans papier enfin copain une connaissance /
[Michel_Chevrier] et euh et je me suis toujours dit mais(MD) <PPI>comment ça se fait</PPI> qu'il est si discret quoi (Expansion)
```

### 6.4 ECRIT_TEST — Development Mode

```python
mode=AnalysisMode.ECRIT_TEST
```

A lightweight mode for quick testing. Wraps the raw text with a single generic `[locuteur 1]` tag and processes it as oral. No speaker detection or segmentation is performed. Useful for debugging prompts or verifying output format.

---

## 7. Processing Modes

### 7.1 Sequential Mode (default)

```python
batch_mode=False  # default
```

Each concordance is preprocessed and analysed in sequence. All properties for sentence N are sent to the LLM before moving to sentence N+1. Property-level calls within a sentence are parallelised across `n_threads` threads.

Best for: small runs, debugging, or when exact control over per-sentence timing is needed.

Rate limiting: set `max_reqs` (e.g. `max_reqs=60`) to insert a sleep between sentences and stay within API rate limits.

### 7.2 Batch Mode

```python
batch_mode=True,
batch_size=5
```

Groups `batch_size` concordances into a single API call per property. The batch prompt lists all sentences sequentially; the LLM returns a JSON object keyed by `sentence no.0`, `sentence no.1`, etc. This dramatically reduces token costs because:

- The system prompt is sent once per batch, not once per sentence.
- If the provider has a prompt cache (e.g. Mistral), the system prompt tokens are cached after the first call.

Best for: large corpora (50+ sentences), cost efficiency, Mistral or DeepSeek providers.

Response parsing (`_parse_batch_response`) handles malformed JSON, strips markdown code fences, and falls back to regex extraction.

### 7.3 Mistral Async Batch Mode

```python
models=["mistral_batch_mistral-medium-latest"],
batch_mode=True,
batch_size=10
```

Uses Mistral's asynchronous batch API, which costs **50% less** than synchronous calls. Jobs are submitted, the pipeline polls for completion, and results are assembled once all jobs finish.

If the run is interrupted before all jobs complete, the pipeline saves a `mistral_batch_job.json` state file in `output_dir`. Re-running with the same config will automatically resume from the saved job IDs — no sentences need to be resubmitted.

State file structure:

```json
{
  "job_map": {"c0_p2": "job-abc123", "c0_p3": "job-def456", ...},
  "preprocessed_json": "/path/to/output/mistral_batch_preprocessed.json"
}
```

---

## 8. Supported Models and Providers

Model strings follow the pattern `{provider}_{submodel}` or `{provider}`:

| Model string | Provider | Notes |
|---|---|---|
| `"mistral_mistral-medium-latest"` | Mistral API | Standard sync calls |
| `"mistral_mistral-large-latest"` | Mistral API | Larger, slower, more accurate |
| `"mistral_batch_mistral-medium-latest"` | Mistral Batch API | Async, 50% cheaper |
| `"deepseek_deepseek"` | DeepSeek API | DeepSeek-V3, strong reasoning |
| `"groq_moonshotai/kimi-k2-instruct"` | Groq API | Fast inference |
| `"gemini_gemini-3-flash-preview"` | Google Gemini | Multimodal capable |
| `"ollama_mistral:7b"` | Ollama (local) | Fully local, no API key needed |
| `"ollama_gemma3:27b"` | Ollama (local) | Large local model |
| `"no_model"` | — | Returns placeholder responses (testing) |

Multiple models can be specified in the `models` list; each one will produce its own column set in the output. This enables side-by-side comparison of models on the same data:

```python
models=["deepseek_deepseek", "mistral_mistral-medium-latest", "ollama_gemma3:27b"]
```

Default submodels (used if only the provider prefix is given) are defined in `config.py`:

```python
DEFAULT_SUBMODELS = {
    "ollama":   "mistral:7b",
    "mistral":  "mistral-large-latest",
    "groq":     "moonshotai/kimi-k2-instruct",
    "gemini":   "gemini-3-flash-preview",
    "no_model": "no_model",
}
```

---

## 9. Linguistic Properties Analysed

The pipeline analyses each PPI occurrence across the full PREFAB grid. Properties are indexed 0–N in the order they appear in `system_prompts.txt`.

| Index | Property | Computed by |
|---|---|---|
| 0 | Forme | Local (regex extraction from `<PPI>` tags) |
| 1 | Lemme | Local (from config expression) |
| 2 | Acception | LLM |
| 3 | Type de phrase | LLM |
| 4 | Modalité d'énonciation | LLM |
| 5 | Position | Local (Stanza + `position.py`) |
| 6 | Propriétés syntaxiques | LLM |
| 7 | Expansion | Local (Stanza + `expansion.py`) |
| 8 | Modifieurs | Local (Stanza + `modifiers.py`) |
| 9 | Cooccurrents | LLM |
| 10 | Portée | LLM |
| 11 | Déclenchement | LLM |
| 12 | Fonction globale | LLM |
| 13 | Fonctions spécifiques | LLM |
| 14 | Remarques diverses | LLM |

The `non_ia` list in `SessionState` (defaulting to `[0, 1, 5, 7, 8]` for written modes, `[0, 1, 5]` for oral) controls which indices are handled locally. You can override this via `PipelineConfig.non_ia`.

Each property produces two output columns: `{Property}` and `{Property} Justification`.

---

## 10. Local NLP Modules

Three properties are computed locally using Stanza dependency trees, without LLM calls.

### 10.1 Position Detection

**File:** `analysis/position.py`

Detects where the PPI sits within its speaker turn: **Initiale**, **Médiane**, **Finale**, or **Totale**. The logic:

1. Tokenise the full speaker turn and the PPI (+ its expansion, in ECRIT_IA mode) removing punctuation.
2. Locate the PPI token sequence within the turn tokens.
3. Count tokens before (`start`) and after (`end`):
   - `start < 5 AND end < 5` → **Totale** (PPI fills almost the entire turn)
   - `start < 5 AND end ≥ 5` → **Initiale**
   - `start ≥ 5 AND end ≥ 5` → **Médiane**
   - `start ≥ 5 AND end < 5` → **Finale**

Example output:

```
Position: Initiale
Justification: La PPI comment ça se fait démarre dans les 5 premiers tokens du tour de parole
de <strong>Michel_Chevrier</strong>: *[Michel_Chevrier] <strong>comment ça se fait</strong>
qu'il est si discret quoi*
```

### 10.2 Expansion Detection

**File:** `analysis/expansion.py`

Identifies syntactic expansions attached to the PPI head in the dependency tree. Three expansion types are detected:

| Type | Trigger | Example |
|---|---|---|
| `infinitive` | `xcomp` dependent with `VERB` | *comment ça se fait **d'oublier ça*** |
| `completive_que` | `ccomp` or `csubj` dependent | *comment ça se fait **qu'il soit là*** |
| `nominal_prep` | `nmod`, `obl`, `obj`, `advcl` with NOUN/PRON/VERB | *comment ça se fait **avec lui*** |

Algorithm:

1. Find the PPI token span in the sentence (by surface matching).
2. Identify the PPI's syntactic head (the word whose governor is outside the PPI).
3. Collect the head's dependants outside the PPI span.
4. For each dependant matching the type conditions, extract its full subtree.

Only the first detected expansion is returned (the most relevant one syntactically).

### 10.3 Modifier Detection

**File:** `analysis/modifiers.py`

Finds lexical items that modify the PPI standard form. Modifier rules are loaded from `modifier_rules.yaml`:

```yaml
upos:       [ADV, ADJ, NOUN]       # POS tags of valid modifiers
deprel:     [obl:mod, nmod, amod, acl:relcl, dislocated]  # dependency relations
lemma:      [dieu, diable]          # specific lemmas always counted as modifiers

excluded_upos:   [PUNKT]
excluded_deprel: []
excluded_lemma:  []
```

A word `w` is a modifier if:
- Its governor's lemma is in the PPI standard form's lemma set (or shares a stem), **and**
- `w`'s POS, dependency relation, or lemma matches the inclusion rules, **and**
- `w`'s lemma is not already part of the standard PPI form, **and**
- `w` is not in any exclusion list.

Example output:

```
Modifieurs: adverbe: <MOD>vraiment comment ça se fait</MOD>
```

Negative markers (`pas`, `rien`, etc.) are removed if they are already part of the standard PPI form.

---

## 11. Prompt System

Prompts are loaded from two plain-text files:

- `system_prompts.txt` — one system prompt per property, delimited by `start_prompt` / `end_prompt` markers, containing a `Prompt_{PropertyName}` identifier.
- `prompts.txt` — additional user-level prompt templates (legacy, optional).

Each system prompt begins with a general instruction block (`GENERAL_PROMPT` or `GENERAL_PROMPT_BATCH`) injected at runtime, followed by the property-specific instructions.

User prompts are assembled dynamically per sentence from templates in `get_prompts()` / `get_prompts_batch()` (`analysis/prompts.py`). Three template variants are used depending on the property type:

**Template A — conversation context** (for Acception, Portée, Déclenchement, Fonctions, Remarques):

```
Analyse de la propriété: Acception
**Contexte de la conversation** :
- **Locuteur** : Michel_Chevrier
- **Interlocuteurs** : ['Locuteur 2']
- **Conversation** : [Michel_Chevrier] et je me suis toujours dit mais(MD) <PPI>comment ça se fait</PPI> qu'il est si discret quoi [...]
**Expression à analyser** : **comment ça se fait**
```

**Template B — speaker turn context** (for Type de phrase, Modalité, Propriétés syntaxiques, Expansions, Modifieurs, Cooccurrents):

```
Analyse de la propriété: type_phrase
**Contexte de la conversation** :
- **Locuteur** : Michel_Chevrier
- **Tour de parole** : et je me suis toujours dit mais(MD) comment ça se fait qu'il est si discret quoi
**Expression à analyser** : **comment ça se fait** (forme relevée)
**Lemme** : **comment ça se fait** (forme par défaut)
```

**Template C — minimal** (for Forme, Lemme):

```
Analyse de la propriété: Forme
**Expression à analyser** : **comment ça se fait**
```

LLMs are instructed to respond exclusively in valid JSON:

```json
{"Propriété": "Interrogative", "Justification": "La PPI est une tournure interrogative indirecte."}
```

---

## 12. Output Files

Three files are generated per run, named using the pattern `{expression}_{range}_{mode}`:

### Excel (Simple) — `{expression}_{range}_{mode}_simple.xlsx`

The standard PREFAB annotation grid: one row per concordance, one column per property (without justifications). Colour coding for quick human review:

- **Red** — Cooccurrents
- **Orange** — Modifieurs  
- **Green** — Expansions
- **Underlined** — Portée

### Excel (Extended) — `{expression}_{range}_{mode}.xlsx`

The full grid with both property values and LLM justifications (one column pair per property). Only produced when `exporting_mode="full"`.

### PDF Report — `{expression}_{range}_{mode}.pdf`

A human-readable report combining all concordance information, model responses, and justifications in a structured layout. Only produced when `exporting_mode="full"`.

---

## 13. Analysis Cache

The analysis cache avoids redundant API calls by persisting LLM responses to disk. It is keyed by an MD5 hash of `(conversation, expression, model, submodel, prompt_type)`.

Enable it in `PipelineConfig`:

```python
use_analysis_cache=True,
analysis_cache_path="/home/user/.ppi_analyser/analysis_cache.json"
# If analysis_cache_path is omitted, defaults to ~/.ppi_analyser/analysis_cache.json
```

Cache behaviour:

- **HIT**: if an identical prompt was already answered by the same model, the stored response is returned immediately at zero cost.
- **MISS**: the API is called, the response is stored, and the cache is saved to disk.
- The cache file is a flat JSON dictionary; it can be inspected or cleared manually.
- `analysis_cache.clear()` wipes all entries.

The cache is particularly useful when:
- Re-running a pipeline after changing only the export format.
- Comparing multiple models on the same sentences (only the non-cached model incurs API costs).
- Recovering from interrupted runs (already-analysed sentences are not re-queried).

---

## 14. Usage Examples

### 14.1 Minimal Example

```python
import os
from ppi_analyser.core import PPIAnalyser
from ppi_analyser.config import PipelineConfig, AnalysisMode

expression = "comment ça se fait"
out_dir = "results/comment_ca_se_fait"
os.makedirs(out_dir, exist_ok=True)

analyser = PPIAnalyser(tokenization_mode="nlp")

config = PipelineConfig(
    models=["deepseek_deepseek"],
    expression=expression,
    sentence_file="data/concordances.xlsx",
    mode=AnalysisMode.ECRIT_IA,
    output_dir=out_dir,
    speaker_detection_model="deepseek_deepseek",
)

df, state = analyser.process_sentences(config)
print(f"Processed {len(df)} concordances in {state.total_time:.1f}s")
```

### 14.2 Batch Mode with Mistral

Groups 10 concordances per API call using synchronous Mistral. Reduces prompt-token costs by ~80%.

```python
config = PipelineConfig(
    models=["mistral_mistral-medium-latest"],
    expression="tu te rends compte",
    sentence_file="data/concordances.xlsx",
    mode=AnalysisMode.ECRIT_IA,
    output_dir="results/",
    speaker_detection_model="mistral_mistral-medium-latest",
    batch_mode=True,
    batch_size=10,
    n_threads=4,
    exporting_mode="full",  # produce Excel + PDF
)
```

### 14.3 Mistral Async Batch (cheapest option)

Uses Mistral's asynchronous batch endpoint at 50% discount. Ideal for very large runs (500+ sentences).

```python
config = PipelineConfig(
    models=["mistral_batch_mistral-medium-latest"],
    expression="c'est pas possible",
    sentence_file="data/corpus_ecrit.xlsx",
    mode=AnalysisMode.ECRIT_IA,
    output_dir="results/async_run",
    speaker_detection_model="mistral_mistral-medium-latest",
    batch_mode=True,
    batch_size=15,
)
# If the run is interrupted, re-run the same script — it will resume automatically.
```

### 14.4 Oral Corpus with Ollama

Fully local run, no API keys required, using a local Gemma model via Ollama.

```python
config = PipelineConfig(
    models=["ollama_gemma3:27b"],
    expression="comment ça se fait",
    sentence_file="data/orfeo_concordances.xlsx",
    mode=AnalysisMode.ORAL,
    output_dir="results/oral_local",
    n_threads=2,          # local GPU may not parallelise well
    ollama_host="localhost",
)
```

### 14.5 Selecting a Sentence Subset

Process only sentences 10 through 20 of the file:

```python
config = PipelineConfig(
    models=["deepseek_deepseek"],
    expression="comment ça se fait",
    sentence_file="data/concordances.xlsx",
    mode=AnalysisMode.ECRIT_IA,
    output_dir="results/",
    speaker_detection_model="deepseek_deepseek",
    start_sent=10,
    max_sentences=11,   # exclusive upper bound: processes rows 10–20
)
```

Or process arbitrary non-contiguous rows by index:

```python
config = PipelineConfig(
    ...
    sent_list=[0, 5, 12, 47, 103],  # exact row indices
)
```

### 14.6 Custom Properties Only

Run only specific properties (e.g. to re-run failed properties without re-querying everything):

```python
config = PipelineConfig(
    models=["deepseek_deepseek"],
    expression="comment ça se fait",
    sentence_file="data/concordances.xlsx",
    mode=AnalysisMode.ECRIT_IA,
    output_dir="results/",
    speaker_detection_model="deepseek_deepseek",
    custom_properties=["Acception", "Fonction_globale", "Remarques_diverses"],
)
```

Property names must match the `Prompt_{Name}` identifiers in `system_prompts.txt`.

### 14.7 With Analysis Cache

Enable caching to avoid re-querying sentences already analysed in a previous run:

```python
config = PipelineConfig(
    models=["deepseek_deepseek"],
    expression="comment ça se fait",
    sentence_file="data/concordances.xlsx",
    mode=AnalysisMode.ECRIT_IA,
    output_dir="results/",
    speaker_detection_model="deepseek_deepseek",
    use_analysis_cache=True,
    analysis_cache_path="/home/user/.ppi_analyser/analysis_cache.json",
)
```

### 14.8 Multiple Models (Comparison Study)

Run three models in parallel on the same concordances. Each model produces its own result columns:

```python
config = PipelineConfig(
    models=[
        "deepseek_deepseek",
        "mistral_mistral-medium-latest",
        "ollama_gemma3:27b",
    ],
    expression="tu te rends compte",
    sentence_file="data/concordances.xlsx",
    mode=AnalysisMode.ECRIT_IA,
    output_dir="results/comparison",
    speaker_detection_model="deepseek_deepseek",
    n_threads=8,
    exporting_mode="full",
)
```

### 14.9 Rate-Limited Run

Respect a provider's rate limit of 60 requests per minute across 4 threads:

```python
config = PipelineConfig(
    models=["mistral_mistral-medium-latest"],
    expression="comment ça se fait",
    sentence_file="data/concordances.xlsx",
    mode=AnalysisMode.ECRIT_IA,
    output_dir="results/",
    speaker_detection_model="mistral_mistral-medium-latest",
    n_threads=4,
    max_reqs=60,  # pipeline will sleep between sentences to stay within limit
)
```

### 14.10 Multi-Expression File

When the input file contains concordances for different PPI lemmas mixed together, store the lemma for each row in a `Lemme` column and enable `expressions_from_file`:

```python
config = PipelineConfig(
    models=["deepseek_deepseek"],
    expression="",          # ignored when expressions_from_file=True
    sentence_file="data/multi_expression_corpus.xlsx",
    mode=AnalysisMode.ECRIT_IA,
    output_dir="results/multi",
    speaker_detection_model="deepseek_deepseek",
    expressions_from_file=True,
)
```

---

## 15. Module Reference

| Module | Role |
|---|---|
| `core.py` — `PPIAnalyser` | Main class; orchestrates the full pipeline |
| `config.py` — `PipelineConfig`, `AnalysisMode` | Configuration dataclass and mode enum |
| `state.py` — `SessionState` | Mutable run-time state shared across modules |
| `analysis/pipeline.py` | Sentence loading, preprocessing, analysis orchestration, export |
| `analysis/sentence.py` | Per-sentence (and batch) LLM calls |
| `analysis/prompts.py` | Prompt assembly: `get_prompts()`, `get_prompts_batch()` |
| `analysis/results.py` | JSON cleaning, normalisation, DataFrame construction |
| `analysis/position.py` | Local position detection (Initiale/Médiane/Finale/Totale) |
| `analysis/expansion.py` | Local expansion detection via Stanza dependency trees |
| `analysis/modifiers.py` | Local modifier detection with YAML-configurable rules |
| `analysis/analysis_cache.py` | MD5-keyed persistent response cache |
| `analysis/mistral_batch_pipeline.py` | Mistral async batch job submission, polling, resumption |
| `preprocessing/conversation.py` | Raw text loading and cleaning |
| `preprocessing/segmentation.py` | Turn segmentation (rule-based and LLM-based) |
| `preprocessing/speakers.py` | Speaker/interlocutor extraction |
| `preprocessing/detect_narration.py` | LLM-based narrative/dialogue separation |
| `exporters/excel.py` | Excel export (simple and extended) |
| `exporters/pdf.py` | PDF report generation |
| `models/factory.py` | Provider instantiation from model string |
| `modifier_rules.yaml` | YAML configuration for modifier detection rules |
| `system_prompts.txt` | System prompts for each linguistic property |
| `prompts.txt` | Optional additional user prompt templates |

---

## 16. Error Handling and Resuming Interrupted Runs

**Mistral Async Batch interruptions:** If a run using `mistral_batch_*` models is stopped before all jobs complete, the pipeline saves `mistral_batch_job.json` and `mistral_batch_preprocessed.json` in `output_dir`. Re-running with the same `PipelineConfig` will automatically detect this file and resume polling, skipping resubmission.

**Missing `<PPI>` tags:** If a concordance does not contain `<PPI>` tags after preprocessing, a `PPITagMissingError` is raised. Check the input row and the speaker detection output in the logs.

**Malformed LLM responses:** `results.py` applies multiple fallback strategies before giving up on a JSON response:
1. Strip markdown code fences.
2. Extract the first `{...}` block with regex.
3. Remove stray backslashes and trailing commas.
4. Attempt a second `json.loads()` after re-escaping quotes.
5. Fall back to regex extraction of `"Propriété"` and `"Justification"` values individually.
6. If all fail, the cell is left as `None` (empty in the output).

**Rate limits:** Set `max_reqs` in `PipelineConfig`. The pipeline computes a per-sentence sleep interval as `60 / (max_reqs / n_threads)` seconds.

**Stanza server unavailable:** The `PPIAnalyser` constructor falls back to a local in-process Stanza pipeline if the API server at `http://localhost:5000` is not reachable. Performance will be lower for large runs; starting the server is recommended.

**Logging:** All modules use Python's standard `logging`. Configure the level in your application:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Session-specific log files are written to the session directory using `setup_logging(session_id=...)`.
