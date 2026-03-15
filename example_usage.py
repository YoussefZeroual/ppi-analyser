# example_usage.py
from ppi_analyser.core import PPIAnalyser
from ppi_analyser.config import PipelineConfig, AnalysisMode

expression = "Comment ça se fait"
models = ["deepseek_deepseek"]
out_dir = f"path/to/output/dir"  # replace with your path
os.makedirs(out_dir,exist_ok=True)

analyser = PPIAnalyser(tokenization_mode="nlp")
config = PipelineConfig(
    models=models,
    expression=expression,
    sentence_file="path/to/excel/file",  # replace with your path
    mode=AnalysisMode.ECRIT_IA,
    output_dir=out_dir,              
    n_threads=8,
    max_reqs=8,
    start_sent=10,
    max_sentences=11,
    speaker_detection_model = "deepseek_deepseek"
    
)
df, state = analyser.process_sentences(config)

