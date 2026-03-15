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
from ppi_analyser.preprocessing.segmentation import detect_segments_ia, detect_segments
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

def _build_output_paths(config: PipelineConfig) -> OutputPaths:
    from pathlib import Path
    base = config.output_dir
    expression = config.expression.replace(" ", "_").replace("'", "")
    stem = Path(config.sentence_file).stem

    if config.sent_list:
        tag = str(config.sent_list)
    else:
        tag = f"{config.start_sent}_{config.max_sentences}"

    return OutputPaths(
        excel=f"{base}/df_output_{expression}_{stem}_{tag}.xlsx",
        excel_simple=f"{base}/df_output_{expression}_{stem}_{tag}_simple.xlsx",
        pdf=f"{base}/Output_{expression}_{stem}_{tag}.pdf",
    )


def _extract_forme(conv: str, expression: str) -> str:
    match = re.findall(r'<PPI>(.*?)</PPI>', conv)
    return match[0] if match else expression
# original — scattered inline in the loop
def _compute_sleep(config: PipelineConfig) -> float:
    if config.max_reqs == -1:
        return 0.0
    return 60 / (config.max_reqs / config.n_threads)
@dataclass
class PreprocessedSentence:
    raw: str
    cleaned: str
    locuteur: str
    interlocuteurs: list[str]
    forme_relevee: str




def process_sentences(
    self,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, SessionState]:

    self.state.fichier = config.sentence_file
    self.state.expression = config.expression
    self.state.custom_properties_list = config.custom_properties

    sentences, lemmes = _load_sentences(config)
    sentences, lemmes = _validate_range(sentences, lemmes, config)
    
    preprocessed = _preprocess_all(sentences, config, self.state)
    results = _analyse_all(preprocessed, lemmes, config, self.state)
    df = _build_dataframe(results, config, self.state)
    _export(df, config, self.state)

    return df, self.state

def _load_sentences(
    config: PipelineConfig,
) -> tuple[list[str], list[str] | None]:

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
        return sentences, lemmes  # already sliced by load_sentences

    sentences = sentences[config.start_sent:max_s]
    lemmes = lemmes[config.start_sent:max_s] if lemmes else None
    return sentences, lemmes
def _preprocess_all(
    sentences: list[str],
    config: PipelineConfig,
    state: SessionState,
) -> list[PreprocessedSentence]:  # a small dataclass, see below

    results = []

    for raw in sentences:
        match config.mode:
            case AnalysisMode.ORAL:
                cleaned = clean_conv(raw, config.mode)

            case AnalysisMode.ECRIT_IA:
                if not config.speaker_detection_model:
                    raise ValueError(
                        "speaker_detection_model must be set for ECRIT_IA mode (e.g. 'ollama_mistral:7b')."
                    )
                segmented = detect_segments_ia(raw, config.speaker_detection_model)
                state.full_ecrit_sentence.append(fix_speaker_turns(segmented, config.mode))
                dialogue = get_dialogue(segmented)
                cleaned = fix_speaker_turns(clean_dialogue(dialogue), config.mode)

            case AnalysisMode.ECRIT:
                segmented = detect_segments(raw, nlp=None)
                state.full_ecrit_sentence.append(fix_speaker_turns(segmented))
                cleaned = get_dialogue_ecrit(remove_incises(segmented))

            case AnalysisMode.ECRIT_TEST:
                cleaned = clean_conv("[locuteur 1] " + raw, "oral")
                state.full_ecrit_sentence.append(cleaned)

        locuteur, interlocuteurs = detect_speakers(cleaned, config.mode)
        forme_relevee = _extract_forme(cleaned, config.expression)

        results.append(PreprocessedSentence(
            raw=raw,
            cleaned=cleaned,
            locuteur=locuteur,
            interlocuteurs=interlocuteurs,
            forme_relevee=forme_relevee,
        ))

    return results

    
def _analyse_all(
    preprocessed: list,
    lemmes: list[str] | None,
    config: PipelineConfig,
    state: SessionState,
) -> list[list[str]]:

    from ppi_analyser.analysis.sentence import process_sentence, process_sentences_batch
    from ppi_analyser.exceptions import PPITagMissingError

    sleep_time = _compute_sleep(config)

    if config.batch_mode:
        logger.info("Running in batch mode for %d sentences", len(preprocessed))

        conversations = [s.cleaned for s in preprocessed]
        forme_relevee_list = [s.forme_relevee for s in preprocessed]
        locuteurs = [s.locuteur for s in preprocessed]
        interlocuteurs = [s.interlocuteurs for s in preprocessed]

        if lemmes:
            for lemme in lemmes:
                state.expression_list.append(lemme)
            expression = lemmes[0]
        else:
            expression = config.expression

        return process_sentences_batch(
            expression=expression,
            forme_relevee=forme_relevee_list,
            conversations=conversations,
            locuteurs=locuteurs,
            interlocuteurs=interlocuteurs,
            state=state,
            models=config.models,
            mode=config.mode,
        )

    # normal mode
    all_results = []

    for i, sent in enumerate(preprocessed):
        expression = lemmes[i] if lemmes else config.expression

        if lemmes:
            state.expression_list.append(expression)

        logger.info("Processing sentence %d/%d", i + 1, len(preprocessed))

        if "<PPI>" not in sent.cleaned:
            raise PPITagMissingError(f"Sentence {i} missing <PPI> tags: {sent.cleaned[:80]}")

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

        if i < len(preprocessed) - 1:
            logger.info("Sleeping %.2fs for rate limit", sleep_time)
            time.sleep(sleep_time)

    return all_results
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
    export_excel_simple(df, paths.excel_simple, sentence_file=config.sentence_file)

    try:
        export_pdf(df, state, paths.pdf, config.mode)
    except Exception as e:
        logger.warning("PDF export failed: %s", e)
