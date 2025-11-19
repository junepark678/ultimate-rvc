"""
Compatibility module for Python 3.10 support.

This module provides backports of features introduced in later Python
versions to ensure compatibility with Python 3.10.13.
"""

from __future__ import annotations

import sys
from enum import Enum

# Import Self from typing or typing_extensions
if sys.version_info >= (3, 11):
    from typing import Self

    from enum import IntEnum, StrEnum
else:
    from typing_extensions import Self

    class StrEnum(str, Enum):
        """
        Enum where members are also (and must be) strings.

        Backport of Python 3.11's StrEnum for Python 3.10 compatibility.
        """

        def __new__(cls, value: str) -> Self:
            """
            Create a new StrEnum member.

            Parameters
            ----------
            value : str
                The string value for the enum member.

            Returns
            -------
            StrEnum
                The new enum member.

            Raises
            ------
            TypeError
                If the value is not a string.

            """
            if not isinstance(value, str):
                msg = f"{value!r} is not a string"
                raise TypeError(msg)
            member = str.__new__(cls, value)
            member._value_ = value
            return member

        def __str__(self) -> str:
            """
            Return the string representation of the enum member.

            Returns
            -------
            str
                The string value of the enum member.

            """
            return str(self.value)

        @staticmethod
        def _generate_next_value_(
            name: str,
            _start: int,
            _count: int,
            _last_values: list[str],
        ) -> str:
            """
            Generate the next value when not given.

            Parameters
            ----------
            name : str
                The name of the member.
            start : int
                The initial start value or None.
            count : int
                The number of existing members.
            last_values : list[str]
                A list of the last defined values.

            Returns
            -------
            str
                The generated value (lowercase name).

            """
            return name.lower()

    # IntEnum is available in Python 3.10, so just re-export it
    from enum import IntEnum

__all__ = ["IntEnum", "Self", "StrEnum"]
