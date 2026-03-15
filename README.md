# PPI Analyser

Outil d'analyse automatique de phrases préfabriquées d'interaction (PPI) françaises.

## Installation
```bash
python -m venv myenv
source myenv/bin/activate
pip install -e .
```

## Configuration

Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```

## Usage
```python
from ppi_analyser.core import PPIAnalyser
from ppi_analyser.config import PipelineConfig, AnalysisMode

analyser = PPIAnalyser()
config = PipelineConfig(
    models=["mistral_mistral-medium-latest"],
    expression="je t'en prie",
    sentence_file="path/to/corpus.xlsx",
    mode=AnalysisMode.ORAL,
    output_dir="path/to/output",
)
df, state = analyser.process_sentences(config)
```

## Modes

- `ORAL` — corpus oral
- `ECRIT` — corpus écrit (segmentation automatique)
- `ECRIT_IA` — corpus écrit avec détection des tours de parole par LLM
- `ECRIT_TEST` — mode test rapide

## Models supported

- Mistral (`mistral_mistral-medium-latest`)
- Ollama (`ollama_mistral:7b`)
- Groq (`groq_moonshotai/kimi-k2-instruct`)
- DeepSeek (`deepseek_deepseek-chat`)
- Gemini (`gemini_gemini-3-flash-preview`)
# ppi_analyser
