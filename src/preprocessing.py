"""Gedeelde code voor het inladen en opschonen van de stroke-data."""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TARGET = "stroke"
NUMERIC_COLS = ["age", "avg_glucose_level", "bmi"]
DROP_COLS = [
    "id",
    "gender_Other",
    "gender_Female",
    "ever_married_No",
    "work_type_Never_worked",
    "Residence_type_Rural",
]


class StrokeData:
    """Laad en schoon de train- en testset van de stroke-competitie op."""

    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        """Lees de data in en voer direct de opschoonstappen uit."""
        self.data_dir = Path(data_dir)
        self.train = pd.read_csv(self.data_dir / "train.csv")
        self.test = pd.read_csv(self.data_dir / "test.csv")
        self.test_ids = self.test["id"].copy()
        self._clean()

    @staticmethod
    def _bool_to_int(df: pd.DataFrame) -> pd.DataFrame:
        """Zet alle boolean kolommen om naar integers (0/1)."""
        df = df.copy()
        bool_cols = df.select_dtypes(include="bool").columns
        df[bool_cols] = df[bool_cols].astype(int)
        return df

    def _clean(self) -> None:
        """Verwijder irrelevante rijen en kolommen (zie EDA, paragraaf 1.6)."""
        self.train = self._bool_to_int(self.train)
        self.test = self._bool_to_int(self.test)
        self.train = self.train[self.train["gender_Other"] == 0]
        self.train = self.train.drop(columns=DROP_COLS)
        self.test = self.test.drop(columns=DROP_COLS)

    def get_xy(self) -> tuple[pd.DataFrame, pd.Series]:
        """Geef de features (X) en de target (y) van de trainset."""
        X = self.train.drop(columns=TARGET)
        y = self.train[TARGET]
        return X, y

    def get_test(self) -> pd.DataFrame:
        """Geef de opgeschoonde features van de testset."""
        return self.test