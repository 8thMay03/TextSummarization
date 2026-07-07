from dataclasses import dataclass


@dataclass(frozen=True)
class SummarizationConfig:
    model_name: str = "VietAI/vit5-base"
    dataset_name: str = "ithieund/VietNews-Abs-Sum"
    dataset_config: str | None = None
    source_column: str = "article"
    target_column: str = "abstract"
    prefix: str = ""
    max_source_length: int = 512
    max_target_length: int = 128
    output_dir: str = "models/vit5-vietnews-summarization"
    seed: int = 42
    generation_num_beams: int = 4


DEFAULT_CONFIG = SummarizationConfig()
