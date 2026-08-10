"""Run the demo web server: ``python -m chemistree.demo``."""

from __future__ import annotations

import uvicorn


def main() -> None:
    """Serve the demo app on localhost."""
    uvicorn.run("chemistree.demo.app:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
