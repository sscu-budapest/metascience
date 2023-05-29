import re
from functools import partial

import datazimmer as dz
import pandas as pd

url_base = dz.SourceUrl("https://www.scimagojr.com/journalrank.php")


class Journal(dz.AbstractEntity):
    sourceid = dz.Index & int

    title = str
    type = str
    issn = str
    country = str
    region = str
    publisher = str
    coverage = str


class JournalRecord(dz.AbstractEntity):
    journal = dz.Index & Journal
    year = dz.Index & int

    categories = str
    rank = int
    journal_rating = float
    h_index = int
    total_docs = int
    ref_per_doc = float
    sjr_best_quartile = str
    total_docs_3years = int
    total_refs = int
    total_cites_3years = int
    citable_docs_3years = int


class JournalArea(dz.AbstractEntity):
    journal = Journal
    area = str


journal_table = dz.ScruTable(Journal)
journal_record_table = dz.ScruTable(JournalRecord)
area_table = dz.ScruTable(JournalArea)


@dz.register_data_loader
def proc():
    start_year = 1999
    end_year = 2021

    df = pd.concat(
        pd.read_csv(f"{url_base}?out=xls&year={y}", sep=";")
        .assign(year=y)
        .rename(columns=partial(renamer, y=y))
        for y in range(start_year, end_year + 1)
    ).assign(
        journal_rating=lambda df: df["sjr"].pipe(_f2str),
        ref_per_doc=lambda df: df["ref_/_doc"].pipe(_f2str),
    )
    journal_table.replace_all(
        df.groupby(journal_table.index_cols)[journal_table.feature_cols].first()
    )

    journal_record_table.replace_all(
        df.rename(columns={Journal.sourceid: JournalRecord.journal.sourceid})
    )

    area_table.replace_all(
        df.set_index(Journal.sourceid)["areas"]
        .rename(JournalArea.area)
        .str.split(re.compile(",|; "))
        .explode()
        .str.strip()
        .reset_index()
        .rename(columns={Journal.sourceid: JournalArea.journal.sourceid})
        .drop_duplicates()
    )


def get_complete_area_pivot() -> pd.DataFrame:
    return (
        journal_table.get_full_df("complete")[Journal.issn]
        .str.split(", ")
        .explode()
        .loc[lambda s: s.str.len() == 8]
        .apply(lambda s: f"{s[:4]}-{s[4:]}")
        .reset_index()
        .merge(
            area_table.get_full_df("complete"),
            left_on=Journal.sourceid,
            right_on=JournalArea.journal.sourceid,
        )
        .dropna(subset=JournalArea.area)
        .pivot_table(
            index=Journal.issn,
            columns=JournalArea.area,
            values=Journal.sourceid,
            aggfunc="count",
        )
        .fillna(0)
        .drop("nan", axis=1, errors="ignore")
    )


def _f2str(s):
    return s.str.replace(",", ".").astype(float)


def renamer(s: str, y: int):
    return (
        s.lower()
        .replace(f"({y})", "")
        .strip()
        .replace(" ", "_")
        .replace(".", "")
        .replace("(", "")
        .replace(")", "")
        .strip()
    )
