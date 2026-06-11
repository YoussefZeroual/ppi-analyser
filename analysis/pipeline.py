# analysis/pipeline.py
import re
import time
import logging
import pandas as pd
from ppi_analyser.exporters.excel import export_excel, export_excel_simple
from dataclasses import dataclass
from ppi_analyser.analysis.sentence import process_sentence
from ppi_analyser.config import PipelineConfig, AnalysisMode
from ppi_analyser.state import SessionState
from ppi_analyser.exceptions import PPITagMissingError
from ppi_analyser.preprocessing.conversation import load_sentences, clean_conv, fix_speaker_turns
from ppi_analyser.preprocessing.segmentation import detect_segments_ia, detect_segments, detect_segments_ia_batch
from ppi_analyser.preprocessing.speakers import detect_speakers
from ppi_analyser.analysis.results import create_df
from ppi_analyser.exporters.pdf import export_pdf
from ppi_analyser.preprocessing.detect_narration import get_dialogue, clean_dialogue, get_dialogue_ecrit

logger = logging.getLogger(__name__)

from dataclasses import dataclass as _dc

@_dc
class OutputPaths:
    excel: str
    excel_simple: str
    pdf: str

@dataclass
class PreprocessedSentence:
    raw: str
    cleaned: str
    locuteur: str
    interlocuteurs: list[str]
    forme_relevee: str


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
from ppi_analyser.preprocessing.speakers import detect_speakers, get_loc_full_turn

def _fill_nlp_preprocessed(
    fixed: str,
    mode: AnalysisMode,
    state: SessionState,
    index:int
) -> None:
    """Compute NLP objects for a single sentence and append to state.nlp_preprocessed_turn."""
    from ppi_analyser.analysis.modifiers import get_ppi_sent
    # fix: double occurrence of same ppi in the same conv, we take the first one as the relevant one for the sentence
    
    ppi_matches = re.findall(r'<PPI>(.*?)</PPI>', fixed, re.IGNORECASE)
    ppi_text = ppi_matches[0] if ppi_matches else ""
    total_occurrences = len(re.findall(re.escape(ppi_text), fixed.replace('<PPI>','').replace('</PPI>',''), re.IGNORECASE)) if ppi_text else 1
    pre_ppi = fixed[:re.search(r'<PPI>', fixed, re.IGNORECASE).start()] if ppi_text else ""
    occurrence_index = len(re.findall(re.escape(ppi_text), pre_ppi, re.IGNORECASE))
    
    full_turn, surface_sent = get_loc_full_turn(fixed, AnalysisMode.ORAL)
    full_turn = full_turn.replace("/", "")
    full_turn = re.sub(r'(<.*?>)', '', full_turn)
    surface_sent = re.sub(r'(<.*?>)', '', surface_sent)

    surface_sent_nlp = state.nlp(surface_sent)
    full_turn_nlp_doc = state.nlp(full_turn)

    segments = re.split(r'[,;]', full_turn)
    full_turn_stripped = next(
        (seg for seg in segments if surface_sent.lower() in seg.lower()),
        full_turn,
    )
    full_turn_stripped_nlp_doc = state.nlp(full_turn_stripped)

    sent, _ = get_ppi_sent(surface_sent_nlp, full_turn_stripped_nlp_doc, state.nlp)
    expression_nlp_doc = state.nlp(state.expression)
    logger.warning("ppi index is %s", occurrence_index)
    state.nlp_preprocessed_turn.append({
        "full_turn_nlp_doc": full_turn_nlp_doc,
        "full_turn_stripped_nlp_doc": full_turn_stripped_nlp_doc,
        "expression_nlp_doc": expression_nlp_doc,
        "forme_nlp_doc": sent,
        "surface_sent_nlp": surface_sent_nlp,
        "index":index,
        "ppi_occurrence": occurrence_index #stores the occurrence index of the PPI in the conversation, to help disambiguate cases with multiple occurrences of the same PPI
    })
    logger.debug(
    "nlp_preprocessed_turn[%s]: full_turn=%s | full_turn_stripped=%s | surface_sent=%s | forme=%s",
    index,
    [w.text for s in full_turn_nlp_doc.sentences for w in s.words],
    [w.text for s in full_turn_stripped_nlp_doc.sentences for w in s.words],
    [w.text for s in surface_sent_nlp.sentences for w in s.words],
    [w.text for w in sent.words] if sent else None,
)
def _build_output_paths(config: PipelineConfig) -> OutputPaths:
    from pathlib import Path
    base = config.output_dir
    expression = config.expression.replace(" ", "_").replace("'", "")
    stem = Path(config.sentence_file).stem
    tag = str(config.sent_list) if config.sent_list else f"{config.start_sent}_{config.max_sentences}"
    return OutputPaths(
        excel=f"{base}/df_output_{expression}_{stem}_{tag}.xlsx",
        excel_simple=f"{base}/df_output_{expression}_{stem}_{tag}_simple.xlsx",
        pdf=f"{base}/Output_{expression}_{stem}_{tag}.pdf",
    )


def _extract_forme(conv: str, expression: str) -> str:
    match = re.findall(r'<PPI>(.*?)</PPI>', conv)
    return match[0] if match else expression


def _compute_sleep(config: PipelineConfig) -> float:
    if config.max_reqs == -1:
        return 0.0
    return 60 / (config.max_reqs / config.n_threads)


def _chunk(lst: list, size: int) -> list[list]:
    return [lst[i:i + size] for i in range(0, len(lst), size)]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _preprocess_one(raw: str, config: PipelineConfig, state: SessionState,sentid:int) -> PreprocessedSentence:
    from ppi_analyser.preprocessing.speakers import get_loc_full_turn, detect_speakers
    match config.mode:
        case AnalysisMode.ORAL:
            cleaned = fix_speaker_turns(raw)
            cleaned = clean_conv(cleaned, config.mode)
            cleaned = cleaned.replace('\\n', ' ')
            cleaned = cleaned.replace('[', '\n[')[1:]
            fixed = fix_speaker_turns(raw, config.mode)
            fixed = re.sub(r'(\[.*?\])', '', fixed)
            logger.info("Prétraitement des tours de parole avec Stanza:(%s) %s ... ", sentid, raw[:100])
            _fill_nlp_preprocessed(fixed, config.mode, state,sentid)
            state.conversation.append(cleaned)

        case AnalysisMode.ECRIT_IA:
            if not config.speaker_detection_model:
            	raise ValueError("speaker_detection_model must be set for ECRIT_IA mode")
            segmented = detect_segments_ia(raw, config.speaker_detection_model)
            state.full_ecrit_sentence.append(fix_speaker_turns(segmented, config.mode))
            dialogue = get_dialogue(segmented)
            cleaned = fix_speaker_turns(clean_dialogue(dialogue), config.mode)
            logger.info("Prétraitement des tours de parole avec Stanza:(%s) %s ... ", sentid, raw[:100])
            _fill_nlp_preprocessed(cleaned, config.mode, state,sentid)
            
            
        case AnalysisMode.ECRIT:
            segmented = detect_segments(raw, nlp=None)
            state.full_ecrit_sentence.append(fix_speaker_turns(segmented))
            cleaned = get_dialogue_ecrit(remove_incises(segmented))
        case AnalysisMode.ECRIT_TEST:
            cleaned = clean_conv("[locuteur 1] " + raw, "oral")
            state.full_ecrit_sentence.append(cleaned)

    locuteur, interlocuteurs = detect_speakers(cleaned, config.mode)
    forme_relevee = _extract_forme(cleaned, config.expression)
    return PreprocessedSentence(
        raw=raw, cleaned=cleaned, locuteur=locuteur,
        interlocuteurs=interlocuteurs, forme_relevee=forme_relevee,
    )


def _preprocess_chunk_batch(
    chunk: list[str],
    config: PipelineConfig,
    state: SessionState,
) -> list[PreprocessedSentence]:
    """Segment one chunk with a single model call, then finish per-sentence work."""
    if not config.speaker_detection_model:
        raise ValueError("speaker_detection_model must be set for ECRIT_IA mode")

    segmented_list = detect_segments_ia_batch(chunk, config.speaker_detection_model)

    i = 0
    results = []
    for raw, segmented in zip(chunk, segmented_list):
        fixed = fix_speaker_turns(segmented, config.mode)
        state.full_ecrit_sentence.append(fixed)
        dialogue = get_dialogue(segmented)
        cleaned = fix_speaker_turns(clean_dialogue(dialogue), config.mode)
        cleaned = cleaned.replace('\\n', ' ')
        locuteur, interlocuteurs = detect_speakers(cleaned, config.mode)
        forme_relevee = _extract_forme(cleaned, config.expression)
        sent_offset = len(state.nlp_preprocessed_turn)
        logger.warning("Prétraitement des tours de parole avec Stanza:(%s) %s ... ", sent_offset, raw[:100])
        
        _fill_nlp_preprocessed(cleaned, config.mode, state,i+sent_offset) 
        results.append(PreprocessedSentence(
            raw=raw, cleaned=cleaned, locuteur=locuteur,
            interlocuteurs=interlocuteurs, forme_relevee=forme_relevee,
        ))
        i +=1
    return results


def _preprocess_all_batch(
    sentences: list[str],
    config: PipelineConfig,
    state: SessionState,
) -> list[PreprocessedSentence]:
    """Preprocess all sentences in chunks of config.batch_size."""
    if config.mode != AnalysisMode.ECRIT_IA:
        return [_preprocess_one(raw, config, state,sent_id) for sent_id,raw in enumerate(sentences)]

    chunks = _chunk(sentences, config.batch_size)
    preprocessed = []

    for idx, chunk in enumerate(chunks):
        logger.info("Segmentation batch %d/%d (%d sentences)", idx + 1, len(chunks), len(chunk))
        preprocessed.extend(_preprocess_chunk_batch(chunk, config, state))

    return preprocessed


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _preprocess_and_analyse(
    sentences: list[str],
    lemmes: list[str] | None,
    config: PipelineConfig,
    state: SessionState,
) -> tuple[list[PreprocessedSentence], list[list[str]]]:
    """Non-batch mode: segment + analyse each sentence before moving to the next."""
    sleep_time = _compute_sleep(config)
    preprocessed = []
    all_results = []

    for i, raw in enumerate(sentences):
        sent = _preprocess_one(raw, config, state,i)
        preprocessed.append(sent)

        expression = lemmes[i] if lemmes else config.expression
        if lemmes:
            state.expression_list.append(expression)

        if "<PPI>" not in sent.cleaned:
            raise PPITagMissingError(f"Sentence {i} missing <PPI> tags: {sent.cleaned[:80]}")

        logger.info("Processing sentence %d/%d", i + 1, len(sentences))
        results = process_sentence(
            expression=expression,
            forme_relevee=sent.forme_relevee,
            conversation=sent.cleaned,
            locuteur=sent.locuteur,
            interlocuteurs=sent.interlocuteurs,
            sent_index=i,
            state=state,
            models=config.models,
            mode=config.mode,
        )
        all_results.append(results)

        if i < len(sentences) - 1:
            logger.info("Sleeping %.2fs for rate limit", sleep_time)
            time.sleep(sleep_time)
            logger.debug(sleep_time)

    return preprocessed, all_results


def _analyse_batch(
    preprocessed: list[PreprocessedSentence],
    lemmes: list[str] | None,
    config: PipelineConfig,
    state: SessionState,
) -> tuple[list[PreprocessedSentence], list[list[str]]]:
    """Batch analysis: chunks of batch_size, one batch prompt per property per chunk."""
    from ppi_analyser.analysis.sentence import process_sentences_batch
    chunks = _chunk(preprocessed, config.batch_size)
    lemme_chunks = _chunk(lemmes, config.batch_size) if lemmes else [None] * len(chunks)
    all_results = []

    for idx, (chunk, lemme_chunk) in enumerate(zip(chunks, lemme_chunks)):
        logger.info("Analysis batch %d/%d (%d sentences)", idx + 1, len(chunks), len(chunk))

        conversations = [s.cleaned for s in chunk]
        forme_relevee_list = [s.forme_relevee for s in chunk]
        locuteurs = [s.locuteur for s in chunk]
        interlocuteurs = [s.interlocuteurs for s in chunk]

        if lemme_chunk:
            state.expression_list.extend(lemme_chunk)
            expression = lemme_chunk[0]
        else:
            expression = config.expression

        # capturing the sentence offset
        
        start_offset = start_offset = idx * config.batch_size
        chunk_results = process_sentences_batch(
            expression=expression,
            forme_relevee=forme_relevee_list,
            conversations=conversations,
            locuteurs=locuteurs,
            interlocuteurs=interlocuteurs,
            state=state,
            models=config.models,
            mode=config.mode,
            start_offset=start_offset
        )
        all_results.extend(chunk_results)

    return preprocessed, all_results


# ---------------------------------------------------------------------------
# DataFrame + export
# ---------------------------------------------------------------------------

def _build_dataframe(
    all_results: list[list[str]],
    preprocessed: list[PreprocessedSentence],
    config: PipelineConfig,
    state: SessionState,
) -> pd.DataFrame:
    dfs = []
    for i, (results, sent) in enumerate(zip(all_results, preprocessed)):
        df = create_df(results, i, config.expression, sent.cleaned, state)
        df["Conversation"] = sent.cleaned
        df["Locuteur"] = sent.locuteur
        df["Interlocuteur(s)"] = ", ".join(sent.interlocuteurs)
        dfs.append(df)
    return pd.concat(dfs).reset_index(drop=True)


def _export(df: pd.DataFrame, config: PipelineConfig, state: SessionState) -> None:
    paths = _build_output_paths(config)
    state.total_time = time.perf_counter() - state.start_time
    export_excel(df, paths.excel)
    export_excel_simple(df, paths.excel_simple, sentence_file=config.sentence_file, config=config) # added session state to account for sent id and fix indices problem
    try:
        export_pdf(df, state, paths.pdf, config.mode)
    except Exception as e:
        logger.warning("PDF export failed: %s", e)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_sentences(config: PipelineConfig) -> tuple[list[str], list[str] | None]:
    sentences = load_sentences(config.sentence_file, config.sent_list)
    lemmes = None
    if config.expressions_from_file:
        df = pd.read_excel(config.sentence_file)
        rows = df.iloc[config.sent_list] if config.sent_list else df
        lemmes = rows["Lemme"].tolist()
    return sentences, lemmes


def _validate_range(
    sentences: list[str],
    lemmes: list[str] | None,
    config: PipelineConfig,
) -> tuple[list[str], list[str] | None]:
    n = len(sentences)
    max_s = n if config.max_sentences == "all" else min(config.max_sentences, n)
    if config.sent_list:
        if max(config.sent_list) >= n:
            raise IndexError(
                f"sent_list index {max(config.sent_list)} exceeds file length {n}"
            )
        return sentences, lemmes
    sentences = sentences[config.start_sent:max_s]
    lemmes = lemmes[config.start_sent:max_s] if lemmes else None
    return sentences, lemmes
