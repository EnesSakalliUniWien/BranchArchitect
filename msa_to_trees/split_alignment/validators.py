"""Custom validators for argument parsing."""

import argparse
from typing import Any, Sequence


class PositiveIntegerAction(argparse.Action):
    """Argparse action that validates the value is >= 1."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        """
        Validate and set the value.

        Args:
            parser: The argument parser.
            namespace: The namespace object.
            values: The input value to validate.
            option_string: The option string that triggered this action.

        Raises:
            ArgumentError: If the value is less than 1.
        """
        # Since this is used with type=int in the argument parser,
        # values will already be converted to int by argparse
        if not isinstance(values, int):
            parser.error(f"{option_string} must be an integer")
            return

        if values < 1:
            parser.error(f"Minimum value for {option_string} is 1")
        setattr(namespace, self.dest, values)
