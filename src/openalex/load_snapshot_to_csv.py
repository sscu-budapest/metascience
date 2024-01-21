import csv
import glob
import gzip
import json
import multiprocessing as mp
import os
from dataclasses import dataclass, field
from functools import partial
from io import TextIOWrapper
from itertools import islice
from pathlib import Path
from typing import Iterable

from partcsv import partition_dicts
from tqdm import tqdm

from .constants import N_GROUPS, PARTITIONED_CSV_PATH, SNAPSHOT_PATH


@dataclass
class MWriter:
    cols: list
    _file: TextIOWrapper = field(init=False)
    _writer: csv.DictWriter = field(init=False)

    def init(self, parent_dir: Path, sub_name: str, partition: str):
        partition_dir = parent_dir / partition  
        partition_dir.mkdir(exist_ok=True, parents=True)
        path = partition_dir / f"{sub_name}.csv.gz"
        self._file = gzip.open(path, "wt", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file, fieldnames=self.cols, extrasaction="ignore"
        )
        self._writer.writeheader()

    def close(self):
        self._file.close()

    def writerow(self, d: dict, jsonify_keys=()):
        for k in jsonify_keys:
            d[k] = json.dumps(d.get(k), ensure_ascii=False)
        self._writer.writerow(d)


@dataclass
class PWriter:
    partition: str
    parent_dir: Path

    def __enter__(self):
        for name, v in self._iterwriters():
            v.init(self.parent_dir, name, self.partition)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for _, v in self._iterwriters():
            v.close()

    def write(self, dic: dict):
        pass

    def _iterwriters(self):
        for name, v in self.__class__.__dict__.items():
            if isinstance(v, MWriter):
                yield name, v


def sub_write(key: str, writer: MWriter, dic: dict, sub_key: str, subval_key=None):
    if subd := dic.get(key):
        subd_l = [subd] if isinstance(subd, dict) else subd
        for _sd in subd_l:
            if not isinstance(_sd, dict):
                _sd = {subval_key: _sd}
            _sd[sub_key] = dic["id"]
            writer.writerow(_sd)


class AuthorsWriter(PWriter):
    authors = MWriter(
        [
            "id",
            "orcid",
            "display_name",
            "display_name_alternatives",
            "works_count",
            "cited_by_count",
            "last_known_institution",
            # "works_api_url",
            "updated_date",
        ]
    )
    ids = MWriter(
        ["author_id", "openalex", "orcid", "scopus", "twitter", "wikipedia", "mag"]
    )
    counts_by_year = MWriter(["author_id", "year", "works_count", "cited_by_count"])

    def write(self, dic: dict):
        dic["last_known_institution"] = (dic.get("last_known_institution") or {}).get(
            "id"
        )
        self.authors.writerow(dic, ["display_name_alternatives"])

        _swrite = partial(sub_write, dic=dic, sub_key="author_id")
        _swrite("ids", self.ids)
        _swrite("counts_by_year", self.counts_by_year)


class ConceptsWriter(PWriter):
    concepts = MWriter(
        [
            "id",
            "wikidata",
            "display_name",
            "level",
            "description",
            "works_count",
            "cited_by_count",
            "image_url",
            "image_thumbnail_url",
            # "works_api_url",
            "updated_date",
        ]
    )
    ancestors = MWriter(["concept_id", "ancestor_id"])
    counts_by_year = MWriter(["concept_id", "year", "works_count", "cited_by_count"])
    ids = MWriter(
        [
            "concept_id",
            "openalex",
            "wikidata",
            "wikipedia",
            "umls_aui",
            "umls_cui",
            "mag",
        ]
    )
    related_concepts = MWriter(["concept_id", "related_concept_id", "score"])

    def write(self, dic: dict):
        concept_id = dic["id"]

        self.concepts.writerow(dic)

        if concept_ids := dic.get("ids"):
            concept_ids["concept_id"] = concept_id
            self.ids.writerow(concept_ids, ["umls_aui", "umls_cui"])

        if ancestors := dic.get("ancestors"):
            for ancestor in ancestors:
                if ancestor_id := ancestor.get("id"):
                    self.ancestors.writerow(
                        {
                            "concept_id": concept_id,
                            "ancestor_id": ancestor_id,
                        }
                    )

        sub_write("counts_by_year", self.counts_by_year, dic, "concept_id")

        if related_concepts := dic.get("related_concepts"):
            for related_concept in related_concepts:
                if related_concept_id := related_concept.get("id"):
                    self.related_concepts.writerow(
                        {
                            "concept_id": concept_id,
                            "related_concept_id": related_concept_id,
                            "score": related_concept.get("score"),
                        }
                    )


class InstitutionsWriter(PWriter):
    institutions = MWriter(
        [
            "id",
            "ror",
            "display_name",
            "country_code",
            "type",
            "homepage_url",
            "image_url",
            "image_thumbnail_url",
            "display_name_acroynyms",
            "display_name_alternatives",
            "works_count",
            "cited_by_count",
            # "works_api_url",
            "updated_date",
        ]
    )
    ids = MWriter(
        ["institution_id", "openalex", "ror", "grid", "wikipedia", "wikidata", "mag"]
    )
    geo = MWriter(
        [
            "institution_id",
            "city",
            "geonames_city_id",
            "region",
            "country_code",
            "country",
            "latitude",
            "longitude",
        ]
    )
    associated_institutions = MWriter(
        ["institution_id", "associated_institution_id", "relationship"]
    )
    counts_by_year = MWriter(
        ["institution_id", "year", "works_count", "cited_by_count"]
    )

    def write(self, dic: dict):
        institution_id = dic["id"]
        self.institutions.writerow(
            dic, ["display_name_acroynyms", "display_name_alternatives"]
        )

        _swrite = partial(sub_write, dic=dic, sub_key="institution_id")
        _swrite("ids", self.ids)
        _swrite("geo", self.geo)
        _swrite("counts_by_year", self.counts_by_year)

        # associated_institutions
        if associated_institutions := dic.get(
            "associated_institutions",
            dic.get("associated_insitutions"),  # typo in api
        ):
            for associated_institution in associated_institutions:
                if associated_institution_id := associated_institution.get("id"):
                    self.associated_institutions.writerow(
                        {
                            "institution_id": institution_id,
                            "associated_institution_id": associated_institution_id,
                            "relationship": associated_institution.get("relationship"),
                        }
                    )


class PublishersWriter(PWriter):
    publishers = MWriter(
        [
            "id",
            "display_name",
            "alternate_titles",
            "country_codes",
            "hierarchy_level",
            "parent_publisher",
            "works_count",
            "cited_by_count",
            # "sources_api_url",
            "updated_date",
        ]
    )
    counts_by_year = MWriter(["publisher_id", "year", "works_count", "cited_by_count"])
    ids = MWriter(["publisher_id", "openalex", "ror", "wikidata"])

    def write(self, dic: dict):
        self.publishers.writerow(dic, ["alternate_titles", "country_codes"])
        _swrite = partial(sub_write, dic=dic, sub_key="publisher_id")
        _swrite("ids", self.ids)
        _swrite("counts_by_year", self.counts_by_year)


class SourcesWriter(PWriter):
    sources = MWriter(
        [
            "id",
            "issn_l",
            "issn",
            "display_name",
            "publisher",
            "works_count",
            "cited_by_count",
            "is_oa",
            "is_in_doaj",
            "homepage_url",
            # "works_api_url",
            "updated_date",
        ]
    )
    ids = MWriter(
        ["source_id", "openalex", "issn_l", "issn", "mag", "wikidata", "fatcat"]
    )
    counts_by_year = MWriter(["source_id", "year", "works_count", "cited_by_count"])

    def write(self, dic: dict):
        source_id = dic["id"]
        self.sources.writerow(dic, ["issn"])

        if source_ids := dic.get("ids"):
            source_ids["source_id"] = source_id
            self.ids.writerow(source_ids, ["issn"])

        sub_write("counts_by_year", self.counts_by_year, dic, "source_id")


class WorksWriter(PWriter):
    works = MWriter(
        [
            "id",
            "doi",
            "title",
            "display_name",
            "publication_year",
            "publication_date",
            "type",
            "cited_by_count",
            "is_retracted",
            "is_paratext",
            # "cited_by_api_url",
            # "abstract_inverted_index",
        ]
    )
    locations = MWriter(
        [
            "work_id",
            "source_id",
            "landing_page_url",
            "pdf_url",
            "is_oa",
            "version",
            "license",
            "tag",
        ]
    )
    authorships = MWriter(
        [
            "work_id",
            "author_position",
            "author_id",
            "institution_id",
            "raw_affiliation_string",
        ]
    )
    biblio = MWriter(["work_id", "volume", "issue", "first_page", "last_page"])
    concepts = MWriter(["work_id", "concept_id", "score"])
    ids = MWriter(["work_id", "openalex", "doi", "mag", "pmid", "pmcid"])
    mesh = MWriter(
        [
            "work_id",
            "descriptor_ui",
            "descriptor_name",
            "qualifier_ui",
            "qualifier_name",
            "is_major_topic",
        ]
    )
    open_access = MWriter(["work_id", "is_oa", "oa_status", "oa_url"])
    referenced_works = MWriter(["work_id", "referenced_work_id"])
    related_works = MWriter(["work_id", "related_work_id"])

    def write(self, dic: dict):
        work_id = dic["id"]
        self.works.writerow(dic, ["abstract_inverted_index"])

        # locations
        locations: list = dic.get("locations") or []
        for k in ["primary_location", "best_oa_location"]:
            if extra_loc := dic.get(k):
                locations.append(extra_loc | {"tag": k})
        for location in locations:
            if (location.get("source") or {}).get("id"):
                self.locations.writerow(
                    {
                        "work_id": work_id,
                        "source_id": location["source"]["id"],
                    }
                    | location
                )

        # authorships
        if authorships := dic.get("authorships"):
            for authorship in authorships:
                if author_id := authorship.get("author", {}).get("id"):
                    institutions = authorship.get("institutions")
                    institution_ids = [
                        i.get("id") for i in institutions if i.get("id")
                    ] or [None]

                    for institution_id in institution_ids:
                        self.authorships.writerow(
                            {
                                "work_id": work_id,
                                "author_position": authorship.get("author_position"),
                                "author_id": author_id,
                                "institution_id": institution_id,
                                "raw_affiliation_string": authorship.get(
                                    "raw_affiliation_string"
                                ),
                            }
                        )

        _swrite = partial(sub_write, dic=dic, sub_key="work_id")
        _swrite("biblio", self.biblio)
        _swrite("ids", self.ids)
        _swrite("mesh", self.mesh)
        _swrite("open_access", self.open_access)
        _swrite(
            "referenced_works", self.referenced_works, subval_key="referenced_work_id"
        )
        _swrite("related_works", self.related_works, subval_key="related_work_id")

        # concepts
        for concept in dic.get("concepts", []):
            if concept_id := concept.get("id"):
                self.concepts.writerow(
                    {
                        "work_id": work_id,
                        "concept_id": concept_id,
                        "score": concept.get("score"),
                    }
                )
        return super().write(dic)


def flatten_concepts():
    single_uid_run("concepts", ConceptsWriter)


def flatten_publishers():
    single_uid_run("publishers", PublishersWriter)


def flatten_sources():
    single_uid_run("sources", SourcesWriter)


def flatten_institutions():
    single_uid_run("institutions", InstitutionsWriter)


def flatten_minors():
    flatten_concepts()
    flatten_publishers()
    flatten_institutions()
    flatten_sources()


def flatten_authors():
    big_run("authors", AuthorsWriter)


def flatten_works():
    big_run("works", WorksWriter)


POISON_PILL = None


def meta_partition_writer(
    partition_name,
    parent_dir,
    queue: mp.Queue,
    append: bool,
    batch_size: int,
    force_keys: list | None,
    kls=WorksWriter,
):
    with kls(partition_name, parent_dir) as writer:
        while True:
            row = queue.get()
            if row is POISON_PILL:
                return
            writer.write(row)


def para_queue_filler(it: Iterable, main_queue: mp.Queue, batch_size: int):
    proces = []
    open_files = mp.Queue(maxsize=4)
    for filename in it:
        open_files.put(1)
        p = mp.Process(
            target=file_consumer, args=(filename, main_queue, batch_size, open_files)
        )
        p.start()
        proces.append(p)
    for pc in proces:
        pc.join()


def file_consumer(filename, main_queue, batch_size, open_files: mp.Queue):
    dicit = filename_extr(filename)
    while True:
        o = list(islice(dicit, batch_size))
        if not o:
            break
        main_queue.put(o)
    open_files.get()


def single_uid_run(k, kls):
    with kls("", PARTITIONED_CSV_PATH / k) as writer:
        for dic in tqdm(unique_iter(k), desc=k):
            writer.write(dic)


def big_run(key: str, kls: type):
    partition_dicts(
        tqdm(dic_iter(key), desc=key),
        # tqdm(file_iter(key)),
        partition_key="id",
        num_partitions=N_GROUPS,
        director_count=20,
        parent_dir=PARTITIONED_CSV_PATH / key,
        slot_per_partition=1_500,
        batch_size=500,
        writer_function=partial(meta_partition_writer, kls=kls),
        # main_queue_filler=para_queue_filler,
    )


def file_iter(k):
    return sorted(
        glob.glob(os.path.join(SNAPSHOT_PATH.as_posix(), "data", k, "*", "*.gz")),
        reverse=True,
    )


def filename_extr(filename):
    with gzip.open(filename, "r") as jsonl:
        for single_json in jsonl:
            if not single_json.strip():
                continue
            dic = json.loads(single_json)
            if not dic.get("id"):
                continue
            yield dic


def dic_iter(k):
    for filename in file_iter(k):
        for d in filename_extr(filename):
            yield d


def unique_iter(k):
    seen_ids = set()
    for dic in dic_iter(k):
        _id = dic["id"]
        if _id in seen_ids:
            continue
        seen_ids.add(_id)
        yield dic
