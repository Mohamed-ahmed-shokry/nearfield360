"""Global pytest configuration and environment setup."""

from __future__ import annotations

import os

# Disable ANSI color escape codes in test CLI runners to prevent Rich from splitting option flags
os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"
