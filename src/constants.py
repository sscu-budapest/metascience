import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

RAW_ROOT = Path(os.environ["RAW_SCIENCE_DATA_PATH"])
PARSED_ROOT = Path(os.environ["PARSED_SCIENCE_DATA_PATH"])
