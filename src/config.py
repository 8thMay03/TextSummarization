from dataclasses import dataclass


@dataclass(frozen=True)
class SummarizationConfig:
    model_name: str = "t5-small"
    dataset_name: str = "cnn_dailymail"
    dataset_config: str = "3.0.0"
    source_column: str = "article"
    target_column: str = "highlights"
    prefix: str = "summarize: "
    max_source_length: int = 512
    max_target_length: int = 128
    output_dir: str = "models/t5-small-cnn-dailymail"


DEFAULT_CONFIG = SummarizationConfig()
