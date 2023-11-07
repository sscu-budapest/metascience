from ..constants import PARSED_ROOT, RAW_ROOT

SNAPSHOT_PATH = RAW_ROOT / "openalex-snapshot"

PARTITIONED_CSV_PATH = RAW_ROOT / "openalex-partitioned-csv"
PARQUET_BLOB_ROOT = PARSED_ROOT / "openalex-parquet"
ID_MAP_ROOT = PARSED_ROOT / "id-maps"

N_GROUPS = 128
