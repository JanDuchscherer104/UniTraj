import shutil
from pathlib import Path
from typing import Annotated, List, Optional, Type, Union

from pydantic import Field, ValidationInfo, field_validator

from ..utils.base_config import CONSOLE, SingletonConfig


class PathConfig(SingletonConfig):
    """
    Singleton configuration class for managing all paths in the UniTraj framework.

    This class defines standard paths relative to the project root and handles
    path validation and creation when needed.
    """

    root: Path = Field(default_factory=lambda: Path(__file__).parents[4].resolve())

    # Standard paths
    data: Annotated[
        Path, Field(default="/work/share/argoverse2_scenarionet/av2_scenarionet")
    ]
    """Base directory for all data."""

    checkpoints: Annotated[Path, Field(default=".logs/checkpoints")]
    """Directory for model checkpoints."""

    configs: Annotated[Path, Field(default=".configs")]
    """Directory for configuration files."""

    cache: Annotated[Path, Field(default=".cache")]
    """Directory for cached dataset files."""

    temp_dir: Annotated[Path, Field(default=".temp_dir")]
    """Directory for temporary files during data processing."""

    split_data_output_dir: Annotated[Path, Field(default="split_data_output")]
    """Directory to copy data to when using the split_data function."""

    # External datasets paths
    # nuscenes_root: Optional[Annotated[Path, Field(default=None)]]
    # """Path to the nuScenes dataset root directory."""

    @field_validator(
        "data",
        "checkpoints",
        "configs",
        "cache",
        "split_data_output_dir",
        "temp_dir",
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
        if isinstance(v, str):
            root = info.data.get("root", Path.cwd())
            v = root / v if not Path(v).is_absolute() else Path(v)
        v = v.resolve()
        # Use ensure_dir method
        cls.ensure_dir(v)
        return v

    # @field_validator(
    #     "nuscenes_root",
    #     mode="before",
    # )
    # @classmethod
    # def validate_optional_path(
    #     cls, v: Optional[Union[str, Path]], info: ValidationInfo
    # ) -> Optional[Path]:
    #     """
    #     Validate optional path fields.

    #     Args:
    #         v (Optional[Union[str, Path]]): The optional path to validate.
    #         info (ValidationInfo): Additional validation context.

    #     Returns:
    #         Optional[Path]: The validated Path object or None.
    #     """
    #     if v is None:
    #         return None

    #     if isinstance(v, str):
    #         root = info.data.get("root", Path.cwd())
    #         v = root / v if not Path(v).is_absolute() else Path(v)

    #     return v.resolve()

    @staticmethod
    def ensure_dir(path: Path):
        """Ensure a directory exists, creating it if necessary."""
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            CONSOLE.error(f"Failed to create directory {path}: {e}")
            raise

    @staticmethod
    def remove_dir(path: Path):
        """Remove a directory and its contents."""
        if path.exists():
            try:
                shutil.rmtree(path)
                CONSOLE.log(f"Removed directory: {path}")
            except Exception as e:
                CONSOLE.error(f"Failed to remove directory {path}: {e}")
                raise
        else:
            CONSOLE.warn(f"Attempted to remove non-existent directory: {path}")

    @staticmethod
    def check_exists(path: Path) -> bool:
        """Check if a path exists."""
        return path.exists()

    def get_cache_path(self, dataset_name: str, phase: str) -> Path:
        """Get the cache path for a specific dataset and phase."""
        path = self.cache / dataset_name / phase
        self.ensure_dir(path)
        return path

    def get_temp_path(self, dataset_name: str, phase: str) -> Path:
        """Get the temporary path for a specific dataset and phase."""
        path = self.temp_dir / dataset_name / phase
        self.ensure_dir(path)
        return path

    def get_temp_split_path(self, dataset_name: str, phase: str, index: int) -> Path:
        """Get the path for a temporary data split file."""
        temp_path = self.get_temp_path(dataset_name, phase)
        return temp_path / f"{index}.pkl"

    def get_cache_chunk_path(self, dataset_name: str, phase: str, index: int) -> Path:
        """Get the path for a cached data chunk (HDF5 file)."""
        cache_path = self.get_cache_path(dataset_name, phase)
        return cache_path / f"{index}.h5"
