"""让 `python -m toklens` 也能用。"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
