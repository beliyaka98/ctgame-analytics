"""Allow ``python -m ctgame``."""

import sys

from .cli import main

sys.exit(main())
