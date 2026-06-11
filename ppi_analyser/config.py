from dotenv import load_dotenv
import os
load_dotenv()



from pathlib import Path
PACKAGE_DIR = Path(__file__).parent
SYSTEM_PROMPTS_FILE = PACKAGE_DIR / "system_prompts.txt"
PROMPTS_FILE = PACKAGE_DIR / "prompts.txt"

MISTRAL_API_KEY  = os.getenv("MISTRAL_API_KEY")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")
OLLAMA_HOST      = os.getenv("OLLAMA_HOST", "http://localhost:11434")
STANZA_API_URL   = "http://localhost:5000"

DEFAULT_SUBMODELS = {
    "ollama":   "mistral:7b",
    "mistral":  "mistral-large-latest",
    "groq":     "moonshotai/kimi-k2-instruct",
    "gemini":   "gemini-3-flash-preview",
    "no_model": "no_model",
}

RESULT_COLUMNS = [
    "Forme", "Forme Justification",
    "Lemme", "Lemme Justification",
    "Acception", "Acception Justification",
    "Type de phrase", "Type de phrase Justification",
    "Modalité d'énonciation", "Modalité d'énonciation Justification",
    "Position", "Position Justification",
    "Propriétés syntaxiques", "Propriétés syntaxiques Justification",
    "Expansion", "Expansion Justification",
    "Modifieurs", "Modifieurs Justification",
    "Cooccurrents", "Cooccurrents Justification",
    "Portée", "Portée Justification",
    "Déclenchement", "Déclenchement Justification",
    "Fonction globale", "Fonction globale Justification",
    "Fonctions spécifiques", "Fonctions spécifiques Justification",
    "Remarques diverses", "Remarques diverses Justification",
]

from dataclasses import dataclass, field
from enum import Enum
class predefined_responses():
	IGNORED_PROPRETY_RESPONSE = '```json\n{"Propriété": "Ignoré par l\'utilisateur", "Justification": "Ignoré par l\'utilisateur"}\n```'
class AnalysisMode(str, Enum):
    ORAL      = "oral"
    ECRIT     = "écrit"
    ECRIT_IA  = "écrit_ia"
    ECRIT_TEST = "écrit_test"



@dataclass
class PipelineConfig:
    models: list[str]
    expression: str
    sentence_file: str
    mode: AnalysisMode
    output_dir: str
    # optional with sensible defaults
    expressions_from_file: bool = False
    start_sent: int = 0
    max_sentences: int | str = "all"
    n_threads: int = 4
    ollama_host: str = "localhost"
    max_reqs: int = -1
    sent_list: list[int] | None = None
    speaker_detection_model: str | None = None
    custom_properties: list[str] | None = None
    batch_mode: bool = False
    batch_size: int = 5
    preprocessed_json: str | None = None
    properties: list[str] | None = None  # if set, only these properties are executed
    use_analysis_cache: bool = False
    analysis_cache_path: str | None = None  # e.g. "/home/joe/.ppi_analyser/analysis_cache.json"
    non_ia:list[int]  =  None
