__version__ = "0.0.1"
version = __version__

from mingpt.model import GPT
from mingpt.trainer import Trainer

__all__ = ["GPT", "Trainer", "__version__", "version"]
