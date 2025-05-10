from enum import Enum
from pathlib import Path
from threading import Lock
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    ForwardRef,
    Generic,
    Optional,
    Type,
    TypeVar,
)

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rich.text import Text
from rich.tree import Tree

from .console import Console

TargetType = TypeVar("TargetType")


class NoTarget:
    @staticmethod
    def setup_target(config: "BaseConfig", **kwargs: Any) -> None:
        return None


class BaseConfig(BaseModel, Generic[TargetType]):
    target: Callable[["BaseConfig[TargetType]"], TargetType] = Field(
        default_factory=lambda: NoTarget,
        exclude=True,
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_default=True,
        validate_assignment=True,
        protected_namespaces=(),
    )

    propagated_fields: Dict[str, Any] = Field(
        default_factory=dict,
        exclude=True,
        description="Tracks fields propagated from parent configs",
    )

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> "BaseConfig":
        """
        Convert a raw Hydra DictConfig into our strongly-typed Pydantic AppConfig.
        """
        # turn into plain python dict, resolving interpolations
        data = OmegaConf.to_container(cfg, resolve=True)
        # feed into Pydantic constructor (will validate/convert types)
        return cls.model_validate(data, strict=False)  # type: ignore

    @classmethod
    def from_toml(cls, toml_path: Path) -> "BaseConfig":
        """
        Load a configuration from a TOML file into this BaseConfig.
        """
        try:
            import tomllib  # type: ignore[import]
        except ImportError:
            import tomli as tomllib
        with toml_path.open("rb") as f:
            data = tomllib.load(f)
        # validate and convert nested configs
        return cls.model_validate(data, strict=False)

    def to_toml(self, toml_path: Path, exclude_none: bool = True) -> None:
        """
        Dump this config to a TOML file, serializing only Pydantic fields.
        """
        try:
            import toml
        except ImportError:
            raise RuntimeError("Install 'toml' package to use BaseConfig.to_toml()")
        data = self.model_dump(exclude_none=exclude_none)
        toml_str = toml.dumps(data)
        path = Path(toml_path)
        path.write_text(toml_str, encoding="utf-8")

    def dump_toml(self, toml_path: Path):
        try:
            import tomllib  # type: ignore[import]
        except ImportError:
            import tomli_w as tomllib
        try:
            with toml_path.open("wb") as f:
                tomllib.dump(self.model_dump(), f)
        except Exception as e:
            Console().error(f"Failed to write TOML file {toml_path}: {e}")
            raise e

    def setup_target(self, **kwargs: Any) -> TargetType:
        if not callable(factory := getattr(self.target, "setup_target", self.target)):
            Console().print(
                f"Target '[bold yellow]{self.target}[/bold yellow]' of type [bold yellow]{factory.__class__.__name__}[/bold yellow] is not callable."
            )
            raise ValueError(
                f"Target '{self.target}' of type {factory.__class__.__name__} is not callable / does not have a 'setup_target' or '__init__' method."
            )

        return factory(self, **kwargs)

    def inspect(self, show_docs: bool = False) -> None:
        """Pretty print config structure using rich.

        Args:
            indent: Base indentation level
            show_docs: Whether to show field descriptions and docstrings
        """
        tree = self._build_tree(show_docs=show_docs)
        Console().print(tree, soft_wrap=False, highlight=True, markup=True, emoji=False)

    def _build_tree(self, show_docs: bool = False) -> Tree:
        """Build rich Tree representation of config."""
        tree = Tree(Text(self.__class__.__name__, style="config.name"))

        if show_docs and self.__class__.__doc__:
            tree.add(Text(self.__class__.__doc__, style="config.doc"))

        for field_name, field in self.model_fields.items():
            value = getattr(self, field_name)
            field_style = (
                "config.propagated"
                if field_name in self.propagated_fields
                else "config.field"
            )

            # Create field node text
            field_text = Text()
            field_text.append(f"{field_name}: ", style=field_style)

            # Handle nested configs
            if isinstance(value, BaseConfig):
                subtree = tree.add(field_text)
                nested_tree = value._build_tree(show_docs=show_docs)
                subtree.add(nested_tree)
                continue

            # Handle lists/tuples of configs
            if (
                isinstance(value, (list, tuple))
                and value
                and isinstance(value[0], BaseConfig)
            ):
                subtree = tree.add(field_text)
                for i, item in enumerate(value):
                    item_tree = item._build_tree(show_docs=show_docs)
                    subtree.add(Text(f"[{i}]", style="config.field")).add(item_tree)
                continue

            # Format value
            value_str = self._format_value(value)
            field_text.append(value_str, style="config.value")

            # Add type info
            type_name = self._get_type_name(field.annotation)
            field_text.append(f" ({type_name})", style="config.type")

            # Add field and documentation
            field_node = tree.add(field_text)
            if show_docs and field.description:
                field_node.add(Text(field.description, style="config.doc"))

        return tree

    def _format_value(self, value: Any) -> str:
        """Format a value for display."""
        try:
            if isinstance(value, str):
                return f'"{value}"'
            if isinstance(value, (int, float, bool)):
                return str(value)
            if isinstance(value, Enum):
                return str(value.value if hasattr(value, "value") else value)
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                if not value:
                    return "{}"
                items = [f"{k}: {repr(v)}" for k, v in value.items()]
                return "{" + ", ".join(items) + "}"
            if value is None:
                return "None"
            if isinstance(value, type):
                return value.__name__
            return repr(value)
        except Exception:
            return "<unprintable>"

    def _get_type_name(self, annotation: Any) -> str:
        """Get type name from annotation."""
        try:
            if hasattr(annotation, "__origin__"):
                origin = annotation.__origin__.__name__
                args = []
                for arg in annotation.__args__:
                    if isinstance(arg, ForwardRef):
                        args.append(arg.__forward_arg__)
                    elif hasattr(arg, "__name__"):
                        args.append(arg.__name__)
                    else:
                        args.append(str(arg))
                return f"{origin}[{', '.join(args)}]"
            return str(annotation).replace("typing.", "")
        except Exception:
            return "Any"

    # @model_validator(mode="after")
    # def _propagate_shared_fields(self) -> "BaseConfig":
    #     """Propagate shared fields to nested BaseConfig instances"""
    #     for field_name, field_value in self:

    #         # If field is another BaseConfig
    #         if isinstance(field_value, BaseConfig):
    #             self._propagate_to_child(field_name, field_value)

    #         # Handle lists/tuples of BaseConfigs
    #         elif isinstance(field_value, (list, tuple)):
    #             for item in field_value:
    #                 if isinstance(item, BaseConfig):
    #                     self._propagate_to_child(field_name, item)

    #     return self

    def _propagate_to_child(
        self, parent_field: str, child_config: "BaseConfig"
    ) -> None:
        """Propagate matching fields from parent to child config"""
        shared_fields = {
            name: value
            for name, value in self
            if name in child_config.model_fields
            and name != parent_field
            and name not in ("propagated_fields", "target")
        }

        for name, value in shared_fields.items():
            current_value = getattr(child_config, name, None)
            if current_value != value:
                setattr(child_config, name, value)
                child_config.propagated_fields[name] = value

                Console().log(
                    f"Propagated {name}={value} from "
                    f"{self.__class__.__name__} to {child_config.__class__.__name__}"
                )


class SingletonConfig(BaseConfig):
    """Base class for singleton configurations."""

    _instances: ClassVar[Dict[Type, Any]] = {}
    _lock: ClassVar[Lock] = Lock()

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, validate_default=True
    )

    def __new__(cls):
        with cls._lock:
            if cls not in cls._instances:
                instance = super(BaseConfig, cls).__new__(cls)
                instance.__dict__["_initialized"] = False
                cls._instances[cls] = instance
            return cls._instances[cls]

    def __init__(self, **kwargs):
        if not getattr(self, "_initialized", False):
            super().__init__(**kwargs)
            self.__dict__["_initialized"] = True
        else:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    current = getattr(self, key)
                    if current != value:
                        Console().warn(
                            f"Updating singleton {self.__class__.__name__} "
                            f"field '{key}' from {current} to {value}"
                        )
                    setattr(self, key, value)

    def __copy__(self) -> "SingletonConfig":
        """Return self since this is a singleton."""
        return self

    def __deepcopy__(self, memo: Optional[Dict[int, Any]] = None) -> "SingletonConfig":
        """Return self since this is a singleton. Implements proper deepcopy protocol."""
        if memo is not None:
            memo[id(self)] = self
        return self
