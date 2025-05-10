import traceback
from pathlib import Path
from typing import Any, Optional

from devtools import pformat
from rich.console import Console as RichConsole
from rich.theme import Theme


class Console(RichConsole):
    # TODO: get prefix automatically from caller (via caller stack?)
    is_debug: bool
    prefix: Optional[str] = None

    default_sttings = {
        "theme": Theme(
            {
                "config.name": "bold blue",  # Config class names
                "config.field": "green",  # Regular fields
                "config.propagated": "yellow",  # Propagated fields
                "config.value": "white",  # Field values
                "config.type": "dim",  # Type annotations
                "config.doc": "italic dim",  # Documentation
            }
        ),
        "width": 120,
        "force_terminal": True,
        "color_system": "auto",
        "markup": True,
        "highlight": True,
    }

    def __init__(self, **kwargs):
        settings = self.default_sttings.copy()
        settings.update(kwargs)
        super().__init__(**settings)
        self.is_debug = False
        self.verbose = True
        self.show_timestamps = False
        self.prefix = None

    @classmethod
    def with_prefix(cls, *parts: str) -> "Console":
        """
        Create a new Console instance with a custom prefix for all log messages.
        Enables builder-style chaining.

        Usage:
        ```python
        console = Console.with_prefix(
            self.__class__.__name__,
            <name_of_the_current_method>
            <further_parts>, # eg. stage, worker_idx...
        )
        ```
        """
        instance = cls()
        instance.set_prefix(*parts)
        return instance

    def set_prefix(self, *parts: str) -> "Console":
        """
        Set a custom prefix for all log messages (e.g., class name + stage).
        Enables builder-style chaining.
        """
        if not parts:
            self.prefix = None
        else:
            self.prefix = "[/bold cyan][grey]::[/grey][bold cyan]".join(
                filter(None, parts)
            )

        return self

    def unset_prefix(self) -> "Console":
        """Unset the prefix for all log messages."""
        self.prefix = None
        return self

    def log(self, message: str) -> None:
        if self.verbose:
            self.print(self._format_message(message))

    def warn(self, message: str) -> None:
        if self.verbose:
            self.print(
                f"[bright_yellow]Warning:[/bright_yellow] {self._format_message(message)}\n"
                f"[dim]{self._get_caller_stack()}[/dim]"
            )

    def error(self, message: str) -> None:
        self.print(
            f"[bright_red]Error:[/bright_red] {self._format_message(message)}\n"
            f"[dim]{self._get_caller_stack()}[/dim]"
        )

    def plog(self, obj: Any, **kwargs) -> None:
        """Pretty print an object using rich."""
        if self.verbose:
            self.print(pformat(obj, **kwargs))

    def dbg(self, message: str) -> None:
        if self.is_debug:
            self.print(
                f"[bold magenta]Debug:[/bold magenta] {self._format_message(message)}"
            )

    def set_verbose(self, verbose: bool) -> "Console":
        self.verbose = verbose
        return self

    def set_debug(self, is_debug: bool) -> "Console":
        self.is_debug = is_debug
        self.is_verbose = self.verbose or is_debug
        return self

    def set_timestamp_display(self, show_timestamps: bool) -> "Console":
        self.show_timestamps = show_timestamps
        return self

    def _format_message(self, message: str) -> str:
        """Format message with optional timestamp and prefix."""
        prefix = f"\[[bold cyan]{self.prefix}[/bold cyan]]: " if self.prefix else ""
        if self.show_timestamps:
            return f"[{self._get_timestamp()}] {prefix}{message}"
        return f"{prefix}{message}"

    def _get_caller_stack(self) -> str:
        """Get formatted stack trace excluding Console internals"""
        stack = traceback.extract_stack()
        # Filter out frames from this file
        current_file = Path(__file__).resolve()
        relevant_frames = [
            frame
            for frame in stack[:-1]  # Exclude current frame
            if Path(frame.filename).resolve() != current_file
        ]
        # Format remaining frames
        return "".join(
            traceback.format_list(relevant_frames[-2:])
        )  # Show last 2 relevant frames
