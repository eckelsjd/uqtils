from pathlib import Path
import os
import shutil
import logging


def pytest_sessionfinish(session, exitstatus):
    """Delete all stray test build products (for example)."""
    files = os.listdir('.')
    logging.shutdown()

    for f in files:
        if Path(f).is_dir() and f.startswith('something_to_delete'):
            # shutil.rmtree(f)
            pass
