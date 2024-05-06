from __future__ import annotations

from typing import Any

from .base import Language
from .character import RecursiveCharacterTextSplitter


class LatexTextSplitter(RecursiveCharacterTextSplitter):
    """Attempts to split the text along Latex-formatted layout elements."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a LatexTextSplitter."""
        separators = self.get_separators_for_language(Language.LATEX)
        super().__init__(separators=separators, **kwargs)
