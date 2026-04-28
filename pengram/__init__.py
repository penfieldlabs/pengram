# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""PENgram — Parse. Extract. Normalize.

An open-source tool that takes raw content and extracts entities and typed
relationships, producing a structured knowledge graph.
"""

from .errors import ConfigError as ConfigError
from .errors import ExtractionError as ExtractionError
from .errors import PengramError as PengramError

__version__ = "0.1.0"
