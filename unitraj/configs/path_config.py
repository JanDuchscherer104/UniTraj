from pathlib import Path
from typing import Annotated, List, Optional, Union

from pydantic import Field, ValidationInfo, field_validator

from ..datasets.types import Stage
from ..utils.base_config import SingletonConfig
from ..utils.console import Console


class PathConfig(SingletonConfig):
    """
    Singleton configuration class for managing all paths in the UniTraj framework.

    This class defines standard paths relative to the project root and handles
    path validation and creation when needed.
    """

    root: Path = Field(default_factory=lambda: Path(__file__).parents[5].resolve())
    checkpoints: Annotated[Path, Field(default=".logs/checkpoints")]
    """Directory for model checkpoints."""
    wandb: Annotated[Path, Field(default=".logs/wandb")]
    configs: Annotated[Path, Field(default=".configs")]
    """Directory for configuration files."""

    # Data paths
    data_root: Annotated[Path, Field(default="/work/share/traj-pred-data")]
    """Base directory for all data."""
    scenarionet_dirs: Annotated[
        List[Path],
        Field(
            default_factory=lambda: [
                "argoverse2_scenarionet/av2_scenarionet",
            ]
        ),
    ]
    argoverse2_dir: Annotated[Path, Field(default="argoverse2/motion-forecasting")]
    """List of directories containing ScenarioNet datasets. Relative to data_root."""
    dataset_dest_dir: Annotated[Path, Field(default="av2_sn_processed")]
    """Directory to store the processed dataset. Relative to data_root."""

    @field_validator(
        "data_root",
        "checkpoints",
        "configs",
        "wandb",
        mode="before",
    )
    @classmethod
    def convert_to_path(cls, v: Union[str, Path], info: ValidationInfo) -> Path:
        """
        Convert string paths to Path objects and ensure they exist.

        Args:
            v (Union[str, Path]): The path value to convert.
            info (ValidationInfo): Additional validation context.

        Returns:
            Path: The validated Path object.
        """
        CONSOLE = Console.with_prefix(cls.__name__, "convert_to_path", info.field_name)

        if isinstance(v, str):
            root = info.data.get("root", Path.cwd())
            v = root / v if not Path(v).is_absolute() else Path(v)
        v = v.resolve()

        if not v.exists():
            try:
                v.mkdir(parents=True, exist_ok=True)
                CONSOLE.log(f"Created directory: {v}")
            except Exception as e:
                CONSOLE.error(f"Failed to create directory {v}: {e}")
                raise

        assert v.exists(), f"Path does not exist: {v}"

        return v

    @field_validator("scenarionet_dirs", mode="before")
    @classmethod
    def convert_scenarionet_dirs(cls, v: List[str], info: ValidationInfo) -> List[Path]:
        """
        Convert scenario net directories to Path objects.

        Args:
            v (Union[str, Path]): The path value to convert.
            info (ValidationInfo): Additional validation context.

        Returns:
            List[Path]: The validated list of Path objects.
        """
        assert len(v) > 0, "`path_config.scenarionet_dirs` cannot be empty."

        paths: List[Path] = []
        for raw_path in v:
            path = Path(raw_path)
            if not path.is_absolute():
                path = info.data["data_root"] / path
            assert path.exists(), f"Scenarionet path does not exist: {path}"
            paths.append(path.resolve())

        return paths

    @field_validator("dataset_dest_dir", mode="before")
    @classmethod
    def convert_dataset_dest_dir(cls, v: str, info: ValidationInfo) -> Path:
        root = info.data["data_root"]
        path = (root / v if not Path(v).is_absolute() else Path(v)).resolve()

        path.mkdir(parents=True, exist_ok=True)
        return path

    @field_validator("argoverse2_dir", mode="before")
    @classmethod
    def convert_dir_rel_data(cls, v: str, info: ValidationInfo) -> Path:
        root = info.data["data_root"]
        path = (root / v if not Path(v).is_absolute() else Path(v)).resolve()

        assert path.exists(), f"Argoverse2 path does not exist: {path}"
        return path

    def find_av2_split(self, scenario_id: str) -> Optional[Stage]:
        for stage in Stage:
            split_dir = self.argoverse2_dir / str(stage)
            assert split_dir.exists(), f"Split directory does not exist: {split_dir}"

            if (split_dir / scenario_id).exists():
                return stage

        return None

    def get_sample_metadata_path(self, stage: Stage) -> Path:
        """
        Get the path to the sample metadata file for a given stage.

        Args:
            stage (Stage): The stage of the dataset.

        Returns:
            Path: The path to the sample metadata pickle file.
        """
        return self.dataset_dest_dir / f"sampe_metadata{stage.name.lower()}.pkl"
