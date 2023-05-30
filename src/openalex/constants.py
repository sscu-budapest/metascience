from ..constants import PARSED_ROOT, RAW_ROOT

SNAPSHOT_PATH = RAW_ROOT / "openalex-snapshot"

PARTITIONED_CSV_PATH = PARSED_ROOT / "openalex-partitioned-csv"
PARQUET_BLOB_ROOT = PARSED_ROOT / "openalex-parquet"

N_GROUPS = 128
