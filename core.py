# ppi_analyser/core.py
import time
from pathlib import Path
import logging
from ppi_analyser.logger import setup_logging
from ppi_analyser.state import SessionState
from ppi_analyser.config import PipelineConfig
from ppi_analyser.analysis.pipeline import (
    _load_sentences,
    _validate_range,
    _preprocess_all_batch,
    _preprocess_and_analyse,
    _analyse_batch,
    _build_dataframe,
    _export,
)

logger = logging.getLogger(__name__)


def _is_mistral_batch(config: PipelineConfig) -> bool:
    return config.batch_mode and any(m.startswith("mistral_batch") for m in config.models)


class PPIAnalyser:
    def __init__(self, tokenization_mode: str = "simple", stanza_url: str = "http://localhost:5000"):
        from datetime import datetime
        self.state = SessionState()
        self.state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        setup_logging(session_id=self.state.session_id)
        self.state.tokenization_mode = tokenization_mode
        self.stanza_url = stanza_url
        if tokenization_mode == "nlp":
            self.state.nlp = self._load_nlp()

    def _load_nlp(self):
        try:
            from ppi_analyser.stanza.stanza_api_proxy import Pipeline
            return Pipeline('fr', api_url="http://localhost:5000")
        except Exception:
            logging.getLogger(__name__).warning(
                "Stanza server not available, falling back to local stanza"
            )
            import stanza
            return stanza.Pipeline('fr', processors='tokenize,pos,lemma,depparse')

    def process_sentences(
        self,
        config: PipelineConfig,
    ) -> tuple:
        self.state.fichier               = config.sentence_file
        self.state.expression            = config.expression
        self.state.custom_properties_list = config.custom_properties
        self.state.n_threads             = config.n_threads
        self.state.ollama_host           = config.ollama_host
        self.state.start_time            = time.perf_counter()

        self.state.use_analysis_cache = config.use_analysis_cache
        if config.use_analysis_cache:
            from ppi_analyser.analysis.analysis_cache import init as cache_init
            cache_path = config.analysis_cache_path or str(
                Path.home() / ".ppi_analyser" / "analysis_cache.json"
            )
            cache_init(cache_path)
            logger.info("Analysis cache enabled: %s", cache_path)

        logger.info("Chargement et préparation des concordances")
        sentences, lemmes = _load_sentences(config)
        sentences, lemmes = _validate_range(sentences, lemmes, config)

        if _is_mistral_batch(config):
            from ppi_analyser.analysis.mistral_batch_pipeline import analyse_batch_mistral_async
            logger.info("Prétraitement des conversations (batch)")
            preprocessed = _preprocess_all_batch(sentences, config, self.state)
            logger.info("Traitement des conversations (Mistral async batch)")
            preprocessed, results = analyse_batch_mistral_async(
                preprocessed, lemmes, config, self.state
            )

        elif config.batch_mode:
            logger.info("Prétraitement des conversations (batch)")
            preprocessed = _preprocess_all_batch(sentences, config, self.state)
            logger.info("Traitement des conversations (batch)")
            preprocessed, results = _analyse_batch(preprocessed, lemmes, config, self.state)

        else:
            logger.info("Prétraitement et traitement séquentiels des conversations")
            preprocessed, results = _preprocess_and_analyse(sentences, lemmes, config, self.state)

        logger.info("Exportation des résultats")
        df = _build_dataframe(results, preprocessed, config, self.state)
        _export(df, config, self.state)
        return df, self.state
