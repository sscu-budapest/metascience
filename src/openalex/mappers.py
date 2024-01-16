from functools import cached_property

import polars as pl

from . import meta as oam
from .constants import ID_MAP_ROOT

ID_PREFIX = "https://openalex.org/"

k_col = "__key"
v_col = oam.idc
NULL_IND_VAL = 0


class IdMapper:
    def __init__(self, name: str, allow_nulls=True) -> None:
        ID_MAP_ROOT.mkdir(parents=True, exist_ok=True)
        self.fp = ID_MAP_ROOT / f"{name}.parquet"
        self.allow_nulls = allow_nulls

    def __call__(
        self, df: pl.DataFrame, col: str = oam.idc, can_miss=False, warn=False
    ):
        out_df = df.pipe(_setup, self.map_df, col)
        mism = out_df.shape[0] != df.shape[0]
        if warn and mism:
            print(f"missed {self.fp.name}: {out_df.shape[0]} vs {df.shape[0]}")
        assert can_miss or (not mism)
        return out_df

    def set_df(self, df: pl.DataFrame, col=oam.idc):
        map_df = self.set_map(df[col].unique())
        out_df = df.pipe(_setup, map_df, col)
        assert out_df.shape[0] == df.shape[0]
        return out_df

    def set_map(self, ids):
        old_map = self.map_df
        ext_map = (
            pl.DataFrame({k_col: ids})
            .select(clean_id(k_col))
            .filter(~pl.col(k_col).is_in(old_map[k_col]))
            .with_row_count(name=v_col, offset=old_map.shape[0])
            .select(k_col, v_col)
        )
        new_map = pl.concat([old_map, ext_map])
        new_map.write_parquet(self.fp)
        return new_map

    @cached_property
    def map_df(self) -> pl.DataFrame:
        if self.fp.exists():
            return pl.read_parquet(self.fp)
        initer = {k_col: None, v_col: NULL_IND_VAL} if self.allow_nulls else []
        return pl.DataFrame(initer, schema={k_col: pl.Int64, v_col: pl.UInt32})


def clean_id(col: str):
    return pl.col(col).str.slice(len(ID_PREFIX) + 1, None).cast(pl.Int64)


def _setup(df: pl.DataFrame, map_df: pl.DataFrame, col: str = oam.idc) -> pl.DataFrame:
    return (
        df.rename({col: k_col})
        .filter(pl.col(k_col).str.starts_with(ID_PREFIX))
        .with_columns(clean_id(k_col))
        .join(map_df.rename({v_col: col}), on=k_col)
        .drop(k_col)
    )


work_type_mapper = {
    "journal-article": 1,
    "book-chapter": 2,
    "proceedings-article": 3,
    "dissertation": 4,
    "posted-content": 5,
    "book": 6,
    "dataset": 7,
    "journal-issue": 8,
    "report": 9,
    "other": 10,
    "reference-entry": 11,
    "monograph": 12,
    "reference-book": 13,
    "peer-review": 14,
    "component": 15,
    "standard": 16,
    "proceedings": 17,
    "journal": 18,
    "report-series": 19,
    "book-part": 20,
    "grant": 21,
    "journal-volume": 22,
    "book-series": 23,
    "proceedings-series": 24,
    "book-set": 25,
}

author_position_mapper = {"middle": 0, "first": 1, "last": 2}

journal_area_mapper = {
    "Agricultural and Biological Sciences": 1,
    "Arts and Humanities": 2,
    "Biochemistry": 3,
    "Business": 4,
    "Chemical Engineering": 5,
    "Chemistry": 6,
    "Computer Science": 7,
    "Decision Sciences": 8,
    "Dentistry": 9,
    "Earth and Planetary Sciences": 10,
    "Econometrics and Finance": 11,
    "Economics": 12,
    "Energy": 13,
    "Engineering": 14,
    "Environmental Science": 15,
    "Genetics and Molecular Biology": 16,
    "Health Professions": 17,
    "Immunology and Microbiology": 18,
    "Management and Accounting": 19,
    "Materials Science": 20,
    "Mathematics": 21,
    "Medicine": 22,
    "Multidisciplinary": 23,
    "Neuroscience": 24,
    "Nursing": 25,
    "Pharmacology": 26,
    "Physics and Astronomy": 27,
    "Psychology": 28,
    "Social Sciences": 29,
    "Toxicology and Pharmaceutics": 30,
    "Veterinary": 31,
}

inst_type_mapper = {
    "government": 1,
    "other": 2,
    "archive": 3,
    "nonprofit": 4,
    "healthcare": 5,
    "education": 6,
    "facility": 7,
    "company": 8,
}
