"""MetaForge error taxonomy (E1xx..E5xx).

Every failure carries a stable code so the pipeline / CLI can branch on it
deterministically instead of parsing message text.
"""

from __future__ import annotations


class MetaForgeError(Exception):
    """Base class for all MetaForge errors."""

    code = "E000"

    def __init__(self, message: str = "", *, code: str | None = None):
        self.message = message
        if code is not None:
            self.code = code
        super().__init__(f"[{self.code}] {message}")


class ConfigError(MetaForgeError):
    code = "E100"


class DataError(MetaForgeError):
    code = "E200"


class BackboneError(MetaForgeError):
    code = "E300"


class TrainError(MetaForgeError):
    code = "E400"


class EvalError(MetaForgeError):
    code = "E500"
