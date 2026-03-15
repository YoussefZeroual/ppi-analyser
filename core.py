# ppi_analyser/core.py

import time
import logging
from ppi_analyser.logger import setup_logging
from ppi_analyser.state import SessionState
from ppi_analyser.config import PipelineConfig
from ppi_analyser.analysis.pipeline import (
    _load_sentences,
    _validate_range,
    _preprocess_all,
    _analyse_all,
    _build_dataframe,
    _export,
)

logger = logging.getLogger(__name__)


class PPIAnalyser:

    def __init__(self,tokenization_mode:str = "simple", stanza_url: str = "http://localhost:5000"):
        from datetime import datetime
        self.state = SessionState()
        self.state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        setup_logging(session_id=self.state.session_id)
        self.state.tokenization_mode = tokenization_mode
        self.stanza_url = stanza_url

        if tokenization_mode == "nlp":
            self.state.nlp = self._load_nlp()

        setup_logging(session_id=self.state.session_id)

    def _load_nlp(self):
        try:
            from ppi_analyser.stanza.stanza_api_proxy import Pipeline
            return Pipeline('fr', api_url="http://localhost:5000")
        except Exception:
            import logging
            logging.getLogger(__name__).warning(
            "Stanza server not available, falling back to local stanza"
        )
            import stanza
            return stanza.Pipeline('fr', processors='tokenize,pos,lemma,depparse')
    def process_sentences(
        self,
        config: PipelineConfig,
    ) -> tuple:

        # initialise state for this run
        self.state.fichier = config.sentence_file
        self.state.expression = config.expression
        self.state.custom_properties_list = config.custom_properties
        self.state.n_threads = config.n_threads
        self.state.ollama_host = config.ollama_host
        self.state.start_time = time.perf_counter()

        # run the pipeline
        logger.info("Chargement et préparation des concordances")
        sentences, lemmes = _load_sentences(config)
        sentences, lemmes = _validate_range(sentences, lemmes, config)
        logger.info("Prétraitement des conversations")
        preprocessed = _preprocess_all(sentences, config, self.state)
        logger.info("Traitement des conversations")
        results = _analyse_all(preprocessed, lemmes, config, self.state)
        logger.info("Exportation des résultats")
        df = _build_dataframe(results, preprocessed, config, self.state)
        _export(df, config, self.state)

        return df, self.state
