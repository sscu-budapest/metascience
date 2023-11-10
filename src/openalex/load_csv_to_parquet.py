import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import polars as pl
import requests
from tqdm import tqdm

from ..scimagojr import get_best_q_by_year, get_issn_area_base
from . import mappers
from . import meta as oam
from .constants import N_GROUPS, PARTITIONED_CSV_PATH

SOURCE_K = "sources"
CONC_K = "concepts"
INST_K = "institutions"
WORK_K = "works"


def load_country_df():
    url = "https://countries.trevorblades.com/"
    data = '{"query":"query { countries {name, code, continent {name}}}"}'

    return pd.DataFrame(
        requests.post(url, data=data).json()["data"]["countries"]
    ).assign(continent=lambda df: df["continent"].apply(lambda d: d["name"]))


# def hash_group(c, n=N_GROUPS):
#     return pl.col(c).hash().mod(n).alias(idg).cast(pl.UInt8)


# def add_hash_group(df: pl.DataFrame, c=idc, n=N_GROUPS):
#     return df.with_columns(hash_group(c, n))


# def clean_id(col: pl.Expr):
#     return col.str.slice(22, None).cast(pl.Int64)


# def clean_k(k: str = idc):
#     return clean_id(pl.col(k))


def limit_group(c, n=oam.MAX_WORK_IN_PART):
    return (pl.col(c) // n).alias(oam.idg).cast(pl.UInt32)


def add_limit_group(df: pl.DataFrame, c=oam.idc, n=oam.MAX_WORK_IN_PART):
    return df.with_columns(limit_group(c, n))


def piv_pref(df: pl.DataFrame, col, ind=oam.wid, ind_as=oam.idc):
    return (
        df.with_columns(pl.lit(True).alias("__isthat"))
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
        # pubmed thing
        return self._get("mesh")

    def get_open_access(self):
        return self._get("open_access")

    def get_referenced_works(self):
        return self._get("referenced_works")

    def get_related_works(self):
        return self._get("related_works")

    def get_works(self):
        return mappers.IdMapper(WORK_K).set_df(self._get(WORK_K))

    @property
    def partition_name(self):
        return self.pdir.name

    def _get(self, n):
        return pl.read_csv(self.pdir / f"{n}.csv.gz", low_memory=False, n_rows=10)


def load_non_parted(k, sub=None):
    return pl.read_csv(
        PARTITIONED_CSV_PATH / k / f"{sub or k}.csv.gz", low_memory=False
    )


def _load_id_mapped(k, id_col=oam.idc):
    return mappers.IdMapper(k).set_df(load_non_parted(k).rename({id_col: oam.idc}))


def load_sources():
    return _load_id_mapped(SOURCE_K)


def load_institutions():
    return _load_id_mapped(INST_K)


def load_institution_geo():
    return load_non_parted(INST_K, "geo")


def dump_statics():
    dump_oa_source_extension()
    dump_concepts()
    dump_institutions()


def dump_work_statics():
    for wp in tqdm(iter_workparts()):
        dump_work_partition_statics(wp)


def post_dump_work_extend():
    for wp in tqdm(iter_workparts()):
        dump_work_relationships(wp)


def dump_oa_source_extension():
    source_base = load_sources()
    sodf = source_base.to_pandas().set_index(oam.idc)

    _isc = "issn"
    _issns = pd.concat(
        [
            sodf[_isc].dropna().apply(json.loads).explode().reset_index(),
            sodf["issn_l"].dropna().rename(_isc).reset_index(),
        ]
    ).drop_duplicates()

    oam.journal_qs.replace_all(
        pl.from_pandas(get_best_q_by_year())
        .select(
            [
                pl.col(_isc),
                pl.col("year").cast(pl.UInt16),
                pl.col("best_q").str.slice(1, None).cast(pl.UInt8),
            ]
        )
        .join(pl.from_pandas(_issns).select([oam.idc, pl.col(_isc)]), on=_isc)
        .drop(_isc)
    )
    area_mapped = pl.col("area").map_dict(mappers.journal_area_mapper).cast(pl.UInt8)
    oam.journal_areas.replace_all(
        pl.from_pandas(get_issn_area_base().merge(_issns))
        .select([oam.idc, area_mapped])
        .unique()
    )
    oam.sources_table.replace_all(
        source_base.select(
            oam.idc, "display_name", "publisher", "is_oa", "homepage_url"
        )
    )


def dump_concepts():
    root_concepts = _load_id_mapped(CONC_K).filter(pl.col("level") <= 1)
    id_map_df = mappers.IdMapper(CONC_K).map_df
    hierarchy_df = (
        load_non_parted(CONC_K, "ancestors")
        .select(
            mappers.clean_id(oam.cid), mappers.clean_id(oam.ancid).alias(mappers.k_col)
        )
        .join(id_map_df.rename({mappers.v_col: oam.ancid}), on=mappers.k_col)
        .drop(mappers.k_col)
        .rename({oam.cid: mappers.k_col})
        .join(id_map_df.rename({mappers.v_col: oam.cid}), on=mappers.k_col)
        .select(oam.cid, oam.ancid)
        .pipe(_add_self_links)
        .unique()
    )

    oam.root_mapper_table.replace_all(hierarchy_df)

    oam.concepts_table.replace_all(
        root_concepts.select(
            oam.idc,
            pl.col("level").cast(pl.UInt8),
            "display_name",
            "description",
            "image_url",
        ).join(hierarchy_df, left_on=oam.idc, right_on=oam.cid, how="left")
    )


def dump_institutions():
    # country_df = pl.from_dataframe(
    #    load_country_df().rename(columns={"code": oam.ccode})
    # ).drop("name")
    country_df = oam.country_shack.get_full_df()
    oam.institution_table.replace_all(
        load_institutions()
        .drop(oam.ccode)
        .join(
            load_institution_geo()
            .rename({oam.iid: oam.idc})
            .pipe(mappers.IdMapper(INST_K).set_df)
            .join(
                country_df.rename(
                    {
                        oam.Country.code: oam.ccode,
                        oam.Country.id: oam.Institution.country.id,
                    }
                ),
                on=oam.ccode,
                how="left",
            ),
            on=oam.idc,
            how="left",
        )
        .with_columns(
            pl.col(oam.Institution.type)
            .map_dict(mappers.inst_type_mapper)
            .fill_null(0)
            .cast(pl.UInt8)
        )
        .select(oam.dz.EntityClass.from_cls(oam.Institution).table_all_columns)
        .unique(subset=oam.idc)
        .pipe(add_limit_group, oam.idc, oam.MAX_INST_IN_PART)
        # .pipe(
        #    lambda df: pl.concat(
        #        [get_inst_group_ext(gid, gdf) for gid, gdf in df.groupby(oam.idg)]
        #    )
        # )
    )


def get_inst_group_ext(gid: int, gdf: pl.DataFrame):
    part_df = oam.authorship_by_inst.get_partition_df({oam.idg: gid})
    return gdf.with_columns(
        pl.col(oam.idc)
        .alias("l_ind")
        .apply(lambda inst_id: part_df[oam.iid].search_sorted(inst_id, side="left")),
    ).with_columns(
        pl.col(oam.idc)
        .alias("paper_count")
        .apply(lambda inst_id: part_df[oam.iid].search_sorted(inst_id, side="right"))
        - pl.col("l_ind"),
    )


def iter_workparts():
    return map(WorkPartition, (PARTITIONED_CSV_PATH / "works").iterdir())


def dump_work_partition_statics(wp: WorkPartition):
    work_base = (
        wp.get_works()
        .filter(
            ~pl.col("is_retracted") & pl.col(oam.Work.publication_year).is_not_null()
        )
        .select(
            oam.idc,
            oam.Work.doi,
            oam.Work.title,
            pl.col(oam.Work.publication_year).cast(pl.UInt16),
            pl.col(oam.Work.cited_by_count).cast(pl.Int32),
            pl.col("type")
            .map_dict(mappers.work_type_mapper)
            .cast(pl.UInt8)
            .fill_null(0),
        )
        .unique(subset=oam.idc, keep="first")
        .sort(oam.idc)
        .pipe(add_limit_group)
    )
    # sexp, cexp, iexp, wexp = _get_partial_exps([SOURCE_K, CONC_K, INST_K, WORK_K])
    smap, cmap, imap, wmap = map(mappers.IdMapper, [SOURCE_K, CONC_K, INST_K, WORK_K])

    wloc_df = (
        wp.get_locations()
        .pipe(smap, oam.sid, can_miss=True)
        .pipe(wmap, oam.wid)
        .select(oam.sid, oam.wid)
        .unique()
    )
    oam.work_sources.extend(wloc_df)

    oam.work_dois.extend(
        wp.get_ids().select(oam.wid, "doi").drop_nulls("doi").pipe(wmap, oam.wid)
    )

    work_best_q = (
        wloc_df.join(work_base, left_on=oam.wid, right_on=oam.idc, how="inner")
        .join(
            oam.journal_qs.get_full_df(),
            left_on=[oam.Work.publication_year, oam.sid],
            right_on=["year", oam.idc],
            how="inner",
        )
        .groupby(oam.wid)
        .agg(pl.col("best_q").min())
        .pipe(piv_pref, col="best_q")
        .rename({str(c): f"Q{c}" for c in range(1, 5)})
    )

    oam.work_basics.extend(
        work_base.join(work_best_q, on=oam.idc, how="left").fill_null(False)
    )

    # oam.work_concepts.extend(concept_base.pipe(piv_pref, col=cid).pipe(add_hash_group))
    oam.work_unstacked_concepts.extend(
        wp.get_concepts()
        .filter(pl.col("score") >= oam.conc_cutoff)
        .pipe(cmap, oam.cid, can_miss=True)
        .pipe(wmap, oam.wid)
        .select(oam.wid, oam.cid)
        .join(oam.root_mapper_table.get_full_df(), on=oam.cid)
        .select([oam.wid, pl.col(oam.ancid).alias(oam.cid)])
        .unique()
        .rename({oam.wid: oam.idc})
        .pipe(add_limit_group)
    )

    oam.work_areas.extend(
        wloc_df.join(
            oam.journal_areas.get_full_df(),
            left_on=oam.sid,
            right_on=oam.idc,
            how="inner",
        )
        .pipe(piv_pref, col="area")
        .pipe(add_limit_group)
    )
    # TODO: author things later
    waff_base = (
        wp.get_authorships()
        .pipe(imap, oam.iid, can_miss=True)
        .pipe(wmap, oam.wid)
        .select(
            oam.wid,
            oam.iid,
            pl.col(oam.iid)
            .n_unique()
            .over([oam.wid, oam.aid])
            .alias("inst_count")
            .cast(pl.UInt8),
            pl.col(oam.aid)
            .n_unique()
            .over(oam.wid)
            .alias("author_count")
            .cast(pl.UInt16),
            pl.col("author_position")
            .map_dict(mappers.author_position_mapper)
            .cast(pl.UInt8),
        )
        .unique()
    )
    inst_ship_base = waff_base.select([oam.iid, oam.wid]).drop_nulls().unique()

    # authorship_by_author.extend(waff_base.select([aid, wid, hash_group(aid)]).unique())
    oam.authorship_by_inst.extend(
        inst_ship_base.with_columns(
            limit_group(oam.iid, oam.MAX_INST_IN_PART),
            limit_group(oam.wid).alias(oam.w_idg),
        )
    )
    oam.authorship_with_institution_by_work.extend(
        inst_ship_base.pipe(add_limit_group, c=oam.wid)
    )
    # authorship_by_work.extend(waff_base.pipe(add_hash_group, c=wid))


def dump_work_relationships(wp: WorkPartition):
    wmap = mappers.IdMapper(WORK_K)
    ref_df = (
        wp.get_referenced_works()
        .unique()
        .pipe(wmap, oam.refid, can_miss=True)
        .pipe(wmap, oam.wid, can_miss=True)
        .select(oam.wid, oam.refid)
    )
    for shack, renamer in [
        (oam.work_impacts, {oam.refid: oam.wid, oam.wid: oam.citeid}),
        (oam.work_impacted, {}),
    ]:
        shack.extend(ref_df.rename(renamer).pipe(add_limit_group, c=oam.wid))


def _add_self_links(df: pl.DataFrame) -> pl.DataFrame:
    unstacked = pl.concat([df[c] for c in df.columns])
    return pl.concat(
        [df, pl.DataFrame(dict(zip(df.columns, [unstacked.unique().sort()] * 2)))]
    )
