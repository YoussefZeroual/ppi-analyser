from dataclasses import dataclass, field
from typing import Optional,Any

@dataclass
class SessionState:
    nlp: Any = None  # stanza pipeline if loaded
    nlp: Any = None  # stanza pipeline if loaded
    session_id: str = ""
    fichier: str = ""
    expression: str = ""
    expression_list: list[str] = field(default_factory=list)
    model_list: list[str] = field(default_factory=list)
    submodel_list: list[str] = field(default_factory=list)
    err_list: list[int] = field(default_factory=list)
    raw_responses: list[str] = field(default_factory=list)
    ollama_host: str = "http://localhost:11434"
    n_threads: int = 4
    total_time: float = 0.0
    individual_conv_time: list[float] = field(default_factory=list)
    sent_index: list[int] = field(default_factory=list)
    html: str = ""
    full_ecrit_sentence: list[str] = field(default_factory=list)
    conversation: list[str] = field(default_factory=list)
    prompts: list = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    dfs: list = field(default_factory=list)
    custom_properties_list: Optional[list[str]] = None
