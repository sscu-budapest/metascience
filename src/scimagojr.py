import datetime as dt
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


class JournalCategory(dz.AbstractEntity):
    journal = Journal
    year = int
    category = str
    q = float


journal_table = dz.ScruTable(Journal)
journal_record_table = dz.ScruTable(JournalRecord)
area_table = dz.ScruTable(JournalArea)
category_table = dz.ScruTable(JournalCategory)


@dz.register_data_loader
def proc():
    start_year = 1999
    end_year = 2022

    df = pd.concat(
        pd.read_csv(f"{url_base}?out=xls&year={y}", sep=";", low_memory=False)
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
        .str.split("; ")
        .explode()
        .str.strip()
        .reset_index()
        .rename(columns={Journal.sourceid: JournalArea.journal.sourceid})
        .drop_duplicates()
    )
    category_table.replace_all(
        df.set_index([Journal.sourceid, JournalRecord.year])[JournalRecord.categories]
        .dropna()
        .str.split("; ")
        .explode()
        .reset_index()
        .assign(
            q_base=lambda df: df.loc[:, JournalRecord.categories]
            .str.extract(r"(\(Q\d\))")
            .values[:, 0]
        )
        .assign(
            **{
                JournalCategory.category: lambda df: [
                    r[JournalRecord.categories].replace(r["q_base"], "").strip()
                    for _, r in df.fillna("").iterrows()
                ],
                JournalCategory.q: lambda df: df["q_base"]
                .str.extract(r"Q(\d)")
                .astype(float)
                .values,
            }
        )
        .rename(columns={Journal.sourceid: JournalCategory.journal.sourceid})
    )


def get_issn_joiner() -> pd.DataFrame:
    return (
        journal_table.get_full_df("complete")[Journal.issn]
        .str.split(", ")
        .explode()
        .loc[lambda s: s != "-"]
        .apply(lambda s: f"{s[:4]}-{s[4:]}")
        .reset_index()
    )


def get_issn_area_base() -> pd.DataFrame:
    return (
        get_issn_joiner()
        .merge(
            area_table.get_full_df("complete"),
            left_on=Journal.sourceid,
            right_on=JournalArea.journal.sourceid,
        )
        .dropna(subset=JournalArea.area)
        .loc[lambda df: df[JournalArea.area] != "nan"]
    )


def get_complete_area_pivot() -> pd.DataFrame:
    return (
        get_issn_area_base()
        .pivot_table(
            index=Journal.issn,
            columns=JournalArea.area,
            values=Journal.sourceid,
            aggfunc="count",
        )
        .fillna(0)
    )


def get_best_q_by_year() -> pd.DataFrame:
    return (
        journal_record_table.get_full_df("complete")[JournalRecord.sjr_best_quartile]
        .loc[lambda s: s != "-"]
        .reset_index()
        .pivot_table(
            index=JournalRecord.journal.sourceid,
            columns=JournalRecord.year,
            values=JournalRecord.sjr_best_quartile,
            aggfunc="first",
        )
        .reindex(range(1950, dt.date.today().year + 1), axis=1)
        .fillna(method="ffill", axis=1)
        .fillna(method="bfill", axis=1)
        .merge(get_issn_joiner(), left_index=True, right_on=Journal.sourceid)
        .drop(Journal.sourceid, axis=1)
        .melt(id_vars=[Journal.issn])
        .rename(columns={"variable": "year", "value": "best_q"})
    )


def _f2str(s: pd.Series):
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
