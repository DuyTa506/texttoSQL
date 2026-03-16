"""
Unified dataset parser package.

Provides a registry of dataset parsers and a factory function to instantiate
them by name.  Each parser converts a dataset's native format into the standard
:class:`~src.schema.models.Database` / :class:`~src.schema.models.Example`
representation.

Usage::

    from src.data_parser import get_parser

    parser = get_parser("bird")
    databases, examples = parser.load("./datasets/data/bird/dev_20240627")

    parser = get_parser("spider_v1")
    databases, examples = parser.load("./data/spider")
"""

from __future__ import annotations

from .base import BaseParser
from .bird_parser import BirdParser
from .spider_parser import SpiderParser

# ---- Parser registry ---------------------------------------------------------

PARSERS: dict[str, type[BaseParser]] = {
    "spider_v1": SpiderParser,
    "spider": SpiderParser,      # convenience alias
    "bird": BirdParser,
}


def get_parser(name: str) -> BaseParser:
    """Instantiate a parser by registered name.

    Parameters
    ----------
    name:
        Parser name — one of ``"spider_v1"``, ``"spider"``, ``"bird"``.

    Raises
    ------
    KeyError:
        If *name* is not in the registry.
    """
    key = name.lower().strip()
    if key not in PARSERS:
        available = ", ".join(sorted(PARSERS))
        raise KeyError(
            f"Unknown parser '{name}'. Available parsers: {available}"
        )
    return PARSERS[key]()


__all__ = [
    "BaseParser",
    "BirdParser",
    "SpiderParser",
    "PARSERS",
    "get_parser",
]
