"""
This package contains the modules for text chunking.
It is copied and refactored from the LangChain project.
As the langchain package is quite heavy, the necessary modules
have been copied into this directory for performance.
"""

from .base import (
    Language,
    TextSplitter,
    Tokenizer,
    TokenTextSplitter,
    split_text_on_tokens,
)
from .character import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from .html import (
    ElementType,
    HTMLHeaderTextSplitter,
    HTMLSectionSplitter,
)
from .json import RecursiveJsonSplitter
from .latex import LatexTextSplitter
from .markdown import (
    HeaderType,
    LineType,
    MarkdownHeaderTextSplitter,
    MarkdownTextSplitter,
)
from .nltk import NLTKTextSplitter
from .python import PythonCodeTextSplitter
from .sentence_transformers import (
    SentenceTransformersTokenTextSplitter,
)
from .spacy import SpacyTextSplitter

__all__ = [
    "TokenTextSplitter",
    "TextSplitter",
    "Tokenizer",
    "Language",
    "RecursiveCharacterTextSplitter",
    "RecursiveJsonSplitter",
    "LatexTextSplitter",
    "PythonCodeTextSplitter",
    "KonlpyTextSplitter",
    "SpacyTextSplitter",
    "NLTKTextSplitter",
    "split_text_on_tokens",
    "SentenceTransformersTokenTextSplitter",
    "ElementType",
    "HeaderType",
    "LineType",
    "HTMLHeaderTextSplitter",
    "HTMLSectionSplitter",
    "MarkdownHeaderTextSplitter",
    "MarkdownTextSplitter",
    "CharacterTextSplitter",
]