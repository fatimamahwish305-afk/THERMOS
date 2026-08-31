import sys
from pathlib import Path
import pytest

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent / "core-backend"))
    exit_code = pytest.main(["test_backend_updates.py", "-v"])
    sys.exit(exit_code)
