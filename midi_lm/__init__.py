import logging
import sys
from pathlib import Path

__version__ = "0.1"
__all__ = ["__version__", "logger", "ROOT_DIR"]

ROOT_DIR = Path(__file__).parent

logger = logging.getLogger(__name__)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
