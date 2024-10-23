import logging
import os

from yohane_cli.app import app

if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())
    app()
