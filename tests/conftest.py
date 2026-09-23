"""Test temporary files inside the writable workspace on restricted Windows hosts."""

from pathlib import Path
import shutil
from uuid import uuid4

import pytest


@pytest.fixture
def tmp_path():
    folder = Path.cwd() / f"codex-test-{uuid4().hex}"
    folder.mkdir()
    try:
        yield folder
    finally:
        shutil.rmtree(folder)
