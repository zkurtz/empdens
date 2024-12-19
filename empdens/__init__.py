from importlib.metadata import version

__version__ = version("empdens")

from pathlib import Path

import pandas as pd

data_path = Path(__file__).parent / "resources" / "data" / "japanese_vowels.csv"
_ = pd.read_csv(data_path)
