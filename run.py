from pydantic_settings import BaseSettings, SettingsConfigDict
from unitraj import ExperimentConfig


class CLIExperimentConfig(BaseSettings, ExperimentConfig):
    model_config = SettingsConfigDict(
        arbitrary_types_allowed=True,
        validate_default=True,
        validate_assignment=True,
        cli_parse_args=True,
    )


if __name__ == "__main__":
    config = CLIExperimentConfig()
    config.run()
