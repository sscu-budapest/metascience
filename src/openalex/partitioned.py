import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from parquetranger import HashPartitioner, TableRepo

from ..scimagojr import get_best_q_by_year, get_complete_area_pivot
from .constants import N_GROUPS, PARQUET_BLOB_ROOT, PARTITIONED_CSV_PATH

INST_GROUPS = 16

wid = "work_id"
iid = "institution_id"
sid = "source_id"
cid = "concept_id"
aid = "author_id"
idc = "id"

ind_hasher = HashPartitioner(num_groups=N_GROUPS)

w_hasher = HashPartitioner(wid, N_GROUPS)
i_hasher = HashPartitioner(iid, INST_GROUPS)
a_hasher = HashPartitioner(aid, N_GROUPS)

authorship_by_inst = TableRepo(
    PARQUET_BLOB_ROOT / "authorship-by-institution", group_cols=i_hasher
)
authorship_by_author = TableRepo(
    PARQUET_BLOB_ROOT / "authorship-by-author", group_cols=a_hasher
)

authorship_by_work = TableRepo(
    PARQUET_BLOB_ROOT / "authorship-by-work", group_cols=w_hasher
)


work_basics = TableRepo(PARQUET_BLOB_ROOT / "work-stats", group_cols=ind_hasher)
work_concepts = TableRepo(PARQUET_BLOB_ROOT / "work-concepts", group_cols=ind_hasher)
work_categories = TableRepo(
    PARQUET_BLOB_ROOT / "work-categories", group_cols=ind_hasher
)

work_impacts = TableRepo(PARQUET_BLOB_ROOT / "work-impact", group_cols=w_hasher)
work_impacted = TableRepo(PARQUET_BLOB_ROOT / "work-impacted", group_cols=w_hasher)


# citation_by_author = ...
# citation_by_institution = ...

important_works = TableRepo(PARQUET_BLOB_ROOT / "important-works")  # 500+ seems good

journal_categories = TableRepo(PARQUET_BLOB_ROOT / "journal-categories")
journal_qs = TableRepo(PARQUET_BLOB_ROOT / "journal-qs")
root_mapper_table = TableRepo(PARQUET_BLOB_ROOT / "concept-root-mapper")

conc_cutoff = 0.65
cited_levels = {
    5: "acknowledged",
    10: "noted",
    50: "influential",
    100: "successful",
    500: "outstanding",
    1000: "major",
    5_000: "essential",
    10_000: "generational",
}

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


@dataclass
class WorkPartition:
    pdir: Path

    def get_authorships(self):
        return self._get("authorships")

    def get_biblio(self):
        return self._get("biblio")

    def get_concepts(self):
        return self._get("concepts")

    def get_ids(self):
        return self._get("ids")

    def get_locations(self):
        return self._get("locations")

    def get_mesh(self):
        return self._get("mesh")

    def get_open_access(self):
        return self._get("open_access")

    def get_referenced_works(self):
        return self._get("referenced_works")

    def get_related_works(self):
        return self._get("related_works")

    def get_works(self):
        return self._get("works")

    @property
    def partition_name(self):
        return self.pdir.name

    def _get(self, n):
        return pd.read_csv(self.pdir / f"{n}.csv")


def _load_non_parted(k):
    return pd.read_csv(
        PARTITIONED_CSV_PATH / k / f"{k}.csv", low_memory=False
    ).set_index("id")


def load_sources():
    return _load_non_parted("sources")


def load_institutions():
    return _load_non_parted("institutions")


def load_concepts():
    return _load_non_parted("concepts")


def clean_id(s):
    return s.str.replace("https://openalex.org/", "").str[1:].astype(np.uint64)


def dump_oal_source_extension():
    sodf = load_sources()

    _issns = pd.concat(
        [
            sodf["issn"].dropna().apply(json.loads).explode().reset_index(),
            sodf["issn_l"].dropna().rename("issn").reset_index(),
        ]
    ).drop_duplicates()

    journal_qs.replace_all(
        get_best_q_by_year()
        .merge(_issns)
        .drop("issn", axis=1)
        .groupby([idc, "year"])
        .min()
    )

    return journal_categories.replace_all(
        get_complete_area_pivot()
        .reset_index()
        .merge(_issns)
        .drop("issn", axis=1)
        .groupby("id")
        .max()
        .astype(bool)
    )


def dump_root_mapper():
    hier_df = pd.read_csv(
        PARTITIONED_CSV_PATH / "concepts" / f"ancestors.csv", low_memory=False
    )

    full_conc_df = load_concepts()

    root_concepts = full_conc_df.loc[lambda df: df["level"] <= 1]

    root_mapper_table.replace_all(
        hier_df.loc[lambda df: df["ancestor_id"].isin(root_concepts.index)]
        .pipe(
            lambda df: pd.concat(
                [df, pd.DataFrame(dict(zip(df.columns, [root_concepts.index] * 2)))],
                ignore_index=True,
            )
        )
        .drop_duplicates()
    )


def iter_workparts():
    return map(WorkPartition, (PARTITIONED_CSV_PATH / "works").iterdir())


refid = f"referenced_{wid}"
puby = "publication_year"
ccount = "cited_by_count"
base_work_cols = [puby]
ext_work_cols = ["doi", "title", "mapped_type", ccount]


def proc_wp_init(wp: WorkPartition):
    wconc_df = wp.get_concepts().loc[lambda df: df["score"] >= conc_cutoff]
    wloc_df = wp.get_locations()

    work_df = (
        wp.get_works()
        .loc[lambda df: ~df["is_retracted"], :ccount]
        .set_index(idc)
        .dropna(subset=[puby])
        .astype({puby: np.uint16, ccount: np.uint32})
        .assign(
            mapped_type=lambda df: df["type"]
            .map(lambda e: work_type_mapper.get(e, 0))
            .astype(np.uint8)
        )
        .loc[:, base_work_cols + ext_work_cols]
    )

    q_by_work = (
        wloc_df[[wid, sid]]
        .merge(work_df[[puby]], left_on=wid, right_index=True)
        .merge(
            journal_qs.get_full_df().reset_index(),
            left_on=[sid, puby],
            right_on=[idc, "year"],
        )
        .groupby(wid)["best_q"]
        .min()
    )

    work_basics.extend(
        work_df.assign(
            best_q=lambda df: q_by_work.reindex(df.index)
            .fillna("Q5")
            .str[1:]
            .astype(np.uint8)
        )
    )

    work_concepts.extend(
        wconc_df.merge(root_mapper_table.get_full_df())
        .loc[:, [wid, "ancestor_id"]]
        .drop_duplicates()
        .assign(b=True)
        .pivot_table(index=wid, columns="ancestor_id", values="b", aggfunc="any")
        .fillna(False)
        .astype(bool)
        .loc[lambda df: df.index.intersection(work_df.index), :]
    )

    work_categories.extend(
        wloc_df.loc[:, [wid, sid]]
        .merge(journal_categories.get_full_df(), left_on=sid, right_index=True)
        .drop(sid, axis=1)
        .groupby(wid)
        .max()
        # .reindex(work_df.index)
        .fillna(False)
        .loc[lambda df: df.index.intersection(work_df.index), :]
        # .assign(Uncategorized=lambda df: ~df.all(axis=1))
        # .astype(bool)
    )

    waff_df = (
        wp.get_authorships()
        .loc[lambda df: df[wid].isin(work_df.index)]
        .assign(
            n_authors=lambda df: df.groupby(wid)[aid]
            .transform("nunique")
            .astype(np.uint16),
            aff_rate=lambda df: df.groupby([wid, aid])["n_authors"]
            .transform("count")
            .astype(np.uint8),
            mapped_position=lambda df: df["author_position"]
            .replace(author_position_mapper)
            .astype(np.uint8),
        )
    )

    authorship_by_author.extend(
        waff_df.loc[:, [wid, aid, "mapped_position", "n_authors"]].drop_duplicates()
    )

    authorship_by_inst.extend(
        waff_df.dropna(subset=iid)
        .loc[:, [wid, iid, aid, "mapped_position", "n_authors", "aff_rate"]]
        .drop_duplicates()
    )
    authorship_by_work.extend(
        waff_df.fillna({iid: ""})
        .loc[:, [wid, iid, aid, "mapped_position", "n_authors", "aff_rate"]]
        .drop_duplicates()
    )


def proc_w_round2(wp: WorkPartition):
    refs_df = wp.get_referenced_works().dropna(subset=refid)
    work_impacts.extend(
        summ_body(wp.partition_name, refs_df)
        .drop([wid], axis=1)
        .rename({refid: wid})
        .groupby([wid, puby])
        .sum()
        .reset_index()
    )
    i_by_year = []
    for gid, gdf in refs_df.groupby(
        HashPartitioner(refid, num_groups=N_GROUPS)(refs_df)
    ):
        if gid != wp.partition_name:
            continue
        i_by_year.append(summ_body(gid, gdf))
    work_impacted.extend(
        pd.concat(i_by_year).drop(refid, axis=1).groupby([wid, puby]).sum()
    )


def summ_body(partition: str, merge_onto: pd.DataFrame, left_col=wid):
    m_kws = dict(
        how="left",
        left_on=left_col,
        right_index=True,
    )
    cat_base = work_categories.get_partition_df(partition)
    return (
        merge_onto.merge(work_basics.get_partition_df(partition)[[puby]], **m_kws)
        .merge(work_concepts.get_partition_df(partition), **m_kws)
        .fillna(False)
        .merge(cat_base, **m_kws)
        .fillna(False)
        .assign(UnCategorized=lambda df: ~df[cat_base.columns].any(axis=1))
    )
