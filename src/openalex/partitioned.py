import json
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path

import pandas as pd
import polars as pl
from shackleton import TableShack

from ..scimagojr import get_best_q_by_year, get_issn_area_base
from .constants import N_GROUPS, PARQUET_BLOB_ROOT, PARTITIONED_CSV_PATH
from .mappers import author_position_mapper, journal_area_mapper, work_type_mapper

INST_GROUPS = 16

wid = "work_id"
iid = "institution_id"
sid = "source_id"
cid = "concept_id"
aid = "author_id"
idc = "id"

idg = "id_group"
w_idg = "work_id_group"

refid = f"referenced_{wid}"
citeid = f"citing_{wid}"
puby = "publication_year"
ccount = "cited_by_count"


def get_shack(name, id_col=None, **kwargs):
    return TableShack(
        PARQUET_BLOB_ROOT / name,
        ipc=True,
        compression="uncompressed",
        id_col=id_col,
        **kwargs,
    )


def get_parted_shack(name, id_col):
    return get_shack(name, id_col, partition_cols=[idg])


authorship_by_inst = get_parted_shack("authorship-by-institution", id_col=iid)
authorship_by_author = get_parted_shack("authorship-by-author", id_col=aid)
authorship_by_work = get_parted_shack("authorship-by-work", id_col=wid)
authorship_with_institution_by_work = get_parted_shack(
    "authorship-w-inst-by-work", id_col=wid
)


work_basics, work_concepts, work_areas, work_unstacked_concepts = [
    get_parted_shack(f"work-{k}", id_col=idc)
    for k in ["basics", "concepts", "areas", "unstacked-concepts"]
]

work_impacts = get_parted_shack("work-impact", id_col=wid)
work_impacted = get_parted_shack("work-impacted", id_col=wid)

# citation_by_author = ...
# citation_by_institution = ...

important_works = get_shack("important-works")  # 500+ seems good
journal_areas = get_shack("journal-areas", id_col=idc)
journal_qs = get_shack("journal-qs", id_col=idc)
root_mapper_table = get_shack("concept-root-mapper", id_col=cid)

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


def old_hash_group(c, n=N_GROUPS):
    # this is based on full (non-clean) id which is stupid
    return (
        pl.col(c)
        .cast(pl.Binary)
        .apply(lambda e: int(md5(e).hexdigest(), base=16) % n)
        .cast(pl.UInt8)
        .alias(idg)
    )


def hash_group(c, n=N_GROUPS):
    return pl.col(c).hash().mod(n).alias(idg).cast(pl.UInt8)


def add_hash_group(df: pl.DataFrame, c=idc, n=N_GROUPS):
    return df.select([pl.all(), hash_group(c, n)])


def clean_id(col: pl.Expr):
    return col.str.slice(22, None).cast(pl.Int64)


def clean_k(k: str = idc):
    return clean_id(pl.col(k))


def piv_pref(df: pl.DataFrame, col, ind=wid, ind_as=idc):
    return (
        df.select([pl.all(), pl.lit(True).alias("__isthat")])
        .pivot(index=ind, columns=col, values="__isthat", aggregate_function="first")
        .fill_null(False)
        .rename({ind: ind_as})
        .pipe(lambda _df: _df.select(sorted(_df.columns)))
    )


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
        return pl.read_csv(self.pdir / f"{n}.csv.gz", low_memory=False)


def _load_non_parted(k):
    return pl.read_csv(PARTITIONED_CSV_PATH / k / f"{k}.csv.gz", low_memory=False)


def load_sources():
    return _load_non_parted("sources")


def load_institutions():
    return _load_non_parted("institutions")


def load_concepts():
    return _load_non_parted("concepts")


def dump_oal_source_extension():
    sodf = load_sources().to_pandas().set_index(idc)

    _isc = "issn"
    _issns = pd.concat(
        [
            sodf[_isc].dropna().apply(json.loads).explode().reset_index(),
            sodf["issn_l"].dropna().rename(_isc).reset_index(),
        ]
    ).drop_duplicates()

    journal_qs.replace_all(
        pl.from_pandas(get_best_q_by_year())
        .select(
            [
                pl.col(_isc),
                pl.col("year").cast(pl.UInt16),
                pl.col("best_q").str.slice(1, None).cast(pl.UInt8),
            ]
        )
        .join(pl.from_pandas(_issns).select([clean_k(), pl.col(_isc)]), on=_isc)
        .drop(_isc)
    )

    return journal_areas.replace_all(
        pl.from_pandas(get_issn_area_base().merge(_issns)).select(
            [clean_k(), pl.col("area").map_dict(journal_area_mapper).cast(pl.UInt8)]
        )
    )


def dump_root_mapper():
    hier_df = pl.read_csv(
        PARTITIONED_CSV_PATH / "concepts" / "ancestors.csv.gz", low_memory=False
    )

    full_conc_df = load_concepts()
    root_concepts = full_conc_df.filter(pl.col("level") <= 1)

    ancid = "ancestor_id"
    root_mapper_table.replace_all(
        hier_df.filter(pl.col(ancid).is_in(root_concepts[idc]))
        .pipe(
            lambda df: pl.concat(
                [df, pl.DataFrame(dict(zip(df.columns, [root_concepts["id"]] * 2)))]
            )
        )
        .select([clean_k(cid), clean_k(ancid)])
        .unique()
    )


def dump_statics():
    dump_oal_source_extension()
    dump_root_mapper()


def iter_workparts():
    return map(WorkPartition, (PARTITIONED_CSV_PATH / "works").iterdir())


def proc_wp_init(wp: WorkPartition):
    dump_work_level(wp)
    dump_ships(wp)


def dump_work_level(wp: WorkPartition):
    wloc_df = wp.get_locations().select([clean_k(wid), clean_k(sid)]).unique()

    work_base = (
        wp.get_works()
        .filter(~pl.col("is_retracted") & pl.col(puby).is_not_null())
        .select(
            clean_k(idc),
            pl.col(puby).cast(pl.UInt16),
            pl.col(ccount).cast(pl.Int32),
            pl.col("type").map_dict(work_type_mapper).cast(pl.UInt8).fill_null(0),
        )
        .unique(subset=idc, keep="first")
        .sort(idc)
        .pipe(add_hash_group)
    )

    work_best_q = (
        wloc_df.join(work_base, left_on=wid, right_on=idc, how="inner")
        .join(
            journal_qs.get_full_df(),
            left_on=[puby, sid],
            right_on=["year", idc],
            how="inner",
        )
        .groupby(wid)
        .agg(pl.col("best_q").min())
        .pipe(piv_pref, col="best_q")
        .rename({str(c): f"Q{c}" for c in range(1, 5)})
    )

    work_basics.extend(work_base.join(work_best_q, on=idc, how="left").fill_null(False))

    concept_base = (
        wp.get_concepts()
        .filter(pl.col("score") >= conc_cutoff)
        .select([clean_k(wid), clean_k(cid)])
        .join(root_mapper_table.get_full_df(), on=cid)
        .select([wid, pl.col("ancestor_id").alias(cid)])
        .unique()
    )

    work_concepts.extend(concept_base.pipe(piv_pref, col=cid).pipe(add_hash_group))
    work_unstacked_concepts.extend(concept_base.rename({wid: idc}).pipe(add_hash_group))

    work_areas.extend(
        wloc_df.join(
            journal_areas.get_full_df(), left_on=sid, right_on=idc, how="inner"
        )
        .pipe(piv_pref, col="area")
        .pipe(add_hash_group)
    )


def dump_ships(wp: WorkPartition):
    waff_base = (
        wp.get_authorships()
        .select(
            clean_k(aid),
            clean_k(wid),
            clean_k(iid),
            pl.col(iid).n_unique().over([wid, aid]).alias("inst_count").cast(pl.UInt8),
            pl.col(aid).n_unique().over(wid).alias("author_count").cast(pl.UInt16),
            pl.col("author_position").map_dict(author_position_mapper).cast(pl.UInt8),
        )
        .unique()
    )
    inst_ship_base = waff_base.select([iid, wid]).drop_nulls().unique()

    authorship_by_author.extend(waff_base.select([aid, wid, hash_group(aid)]).unique())
    authorship_by_inst.extend(
        inst_ship_base.select(
            [pl.all(), hash_group(iid, INST_GROUPS), hash_group(wid).alias(w_idg)]
        )
    )
    authorship_with_institution_by_work.extend(
        inst_ship_base.pipe(add_hash_group, c=wid)
    )
    authorship_by_work.extend(waff_base.pipe(add_hash_group, c=wid))

    ref_df = wp.get_referenced_works().unique().select([clean_k(wid), clean_k(refid)])
    for shack, renamer in [
        (work_impacts, {refid: wid, wid: citeid}),
        (work_impacted, {}),
    ]:
        shack.extend(ref_df.rename(renamer).pipe(add_hash_group, c=wid))


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
    cat_base = work_areas.get_partition_df(partition)
    return (
        merge_onto.merge(work_basics.get_partition_df(partition)[[puby]], **m_kws)
        .merge(work_concepts.get_partition_df(partition), **m_kws)
        .fillna(False)
        .merge(cat_base, **m_kws)
        .fillna(False)
        .assign(UnCategorized=lambda df: ~df[cat_base.columns].any(axis=1))
    )
