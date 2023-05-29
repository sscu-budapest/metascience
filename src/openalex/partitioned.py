import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from parquetranger import TableRepo

from ..scimagojr import get_complete_area_pivot
from .constants import PARQUET_BLOB_ROOT, PARTITIONED_CSV_PATH

# ship_by_inst = TableRepo()
# work_stats_init = TableRepo()
# journal_cats = TableRepo()


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

    def _get(self, n):
        return pd.read_csv(self.pdir / f"{n}.csv")


def get_oal_souce_cats():
    sodf = pd.read_csv(
        PARTITIONED_CSV_PATH / "sources" / "sources.csv", low_memory=False
    ).set_index("id")

    _issns = pd.concat(
        [
            sodf["issn"].dropna().apply(json.loads).explode().reset_index(),
            sodf["issn_l"].dropna().rename("issn").reset_index(),
        ]
    ).drop_duplicates()

    return (
        get_complete_area_pivot()
        .reset_index()
        .merge(_issns)
        .drop("issn", axis=1)
        .groupby("id")
        .max()
        .astype(bool)
    )
