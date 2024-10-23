import logging
import os

try:
    from yohane_cli import app  # type: ignore
except ImportError:
    app = None


def main():
    if not app:
        raise RuntimeError('To use the yohane command, please install "yohane[cli]"')
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())
    app()


if __name__ == "__main__":
    main()
