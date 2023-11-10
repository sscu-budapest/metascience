import datazimmer as dz
import numpy as np
import polars as pl
from shackleton import TableShack

from .constants import PARQUET_BLOB_ROOT

IND_TYPE = pl.UInt32


class Continent(dz.AbstractEntity):
    id = dz.Index & int
    display_name = object


class Country(dz.AbstractEntity):
    id = dz.Index & int
    code = object
    display_name = object
    continent = Continent


class Work(dz.AbstractEntity):
    id = int
    doi = str
    title = str
    publication_year = int
    cited_by_count = int
    type = int
    Q1 = bool
    Q2 = bool
    Q3 = bool
    Q4 = bool


class Institution(dz.AbstractEntity):
    id = dz.Index & int
    ror = object
    display_name = object
    homepage_url = object
    image_url = object
    city = object
    region = object
    country = Country
    latitude = float
    longitude = float
    type = int


class Concept(dz.AbstractEntity):
    concept_id = np.int64
    child_concept_code = np.uint8
    root_concept_code = np.uint8
    level = np.uint8
    display_name = object
    description = object
    image_url = object


# Country, Big Paper, Journal

MAX_INST_IN_PART = 2**14
MAX_WORK_IN_PART = 2**21

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
ancid = "ancestor_id"
topc_id = "root_concept_code"
subc_id = "child_concept_code"
ccode = "country_code"

max_sub = 45


def get_shack(name, id_col=None, **kwargs):
    return TableShack(
        PARQUET_BLOB_ROOT / name,
        # ipc=True,
        # compression="uncompressed",
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


(
    work_basics,
    work_concepts,
    work_areas,
    work_sources,
    work_dois,
    work_unstacked_concepts,
    # work_half_stacked_concepts,
    # work_hierarchical_concepts,
) = [
    get_parted_shack(f"work-{k}", id_col=idc)
    for k in [
        "basics",
        "concepts",
        "areas",
        "sources",
        "dois",
        "unstacked-concepts",
        # "half-stacked-concepts",
        # "hierarchical-concepts",
    ]
]

work_impacts = get_parted_shack("work-impact", id_col=wid)
work_impacted = get_parted_shack("work-impacted", id_col=wid)

# citation_by_author = ...
# citation_by_institution = ...

# important_works = get_shack("important-works")  # 500+ seems good
journal_areas = get_shack("journal-areas", id_col=idc)
journal_qs = get_shack("journal-qs", id_col=idc)
root_mapper_table = get_shack("concept-root-mapper", id_col=cid)

institution_table = get_shack("institutions", id_col=idc)
concepts_table = get_shack("concepts", id_col=idc)
sources_table = get_shack("sources", id_col=idc)

continent_shack = get_shack("continent")
country_shack = get_shack("country")

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
