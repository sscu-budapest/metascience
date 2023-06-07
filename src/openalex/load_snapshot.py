import csv
import glob
import gzip
import json
import multiprocessing as mp
import os
from dataclasses import dataclass, field
from functools import partial
from io import FileIO
from itertools import islice
from pathlib import Path
from typing import Iterable

from partcsv import partition_dicts
from tqdm import tqdm

from .constants import N_GROUPS, PARTITIONED_CSV_PATH, SNAPSHOT_PATH


@dataclass
class MWriter:
    cols: list
    _file: FileIO = field(init=False, default=None)
    _writer: csv.DictWriter = field(init=False, default=None)

    def init(self, parent_dir: Path, sub_name: str, partition: str):
        path = parent_dir / partition / f"{sub_name}.csv"
        path.parent.mkdir(exist_ok=True, parents=True)
        self._file = gzip.open(
            parent_dir / partition / f"{sub_name}.csv.gz", "wt", encoding="utf-8"
        )
        self._writer = csv.DictWriter(
            self._file, fieldnames=self.cols, extrasaction="ignore"
        )
        self._writer.writeheader()

    def close(self):
        self._file.close()

    def writerow(self, d: dict):
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
            "works_api_url",
            "updated_date",
        ]
    )
    ids = MWriter(
        ["author_id", "openalex", "orcid", "scopus", "twitter", "wikipedia", "mag"]
    )
    counts_by_year = MWriter(["author_id", "year", "works_count", "cited_by_count"])

    def write(self, dic: dict):
        author_id = dic["id"]
        dic["display_name_alternatives"] = json.dumps(
            dic.get("display_name_alternatives"), ensure_ascii=False
        )
        dic["last_known_institution"] = (dic.get("last_known_institution") or {}).get(
            "id"
        )
        self.authors.writerow(dic)

        # ids
        if author_ids := dic.get("ids"):
            author_ids["author_id"] = author_id
            self.ids.writerow(author_ids)

        # counts_by_year
        if counts_by_year := dic.get("counts_by_year"):
            for count_by_year in counts_by_year:
                count_by_year["author_id"] = author_id
                self.counts_by_year.writerow(count_by_year)


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
            "works_api_url",
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
            concept_ids["umls_aui"] = json.dumps(
                concept_ids.get("umls_aui"), ensure_ascii=False
            )
            concept_ids["umls_cui"] = json.dumps(
                concept_ids.get("umls_cui"), ensure_ascii=False
            )
            self.ids.writerow(concept_ids)

        if ancestors := dic.get("ancestors"):
            for ancestor in ancestors:
                if ancestor_id := ancestor.get("id"):
                    self.ancestors.writerow(
                        {
                            "concept_id": concept_id,
                            "ancestor_id": ancestor_id,
                        }
                    )

        if counts_by_year := dic.get("counts_by_year"):
            for count_by_year in counts_by_year:
                count_by_year["concept_id"] = concept_id
                self.counts_by_year.writerow(count_by_year)

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
            "works_api_url",
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
        # institutions
        dic["display_name_acroynyms"] = json.dumps(
            dic.get("display_name_acroynyms"), ensure_ascii=False
        )
        dic["display_name_alternatives"] = json.dumps(
            dic.get("display_name_alternatives"), ensure_ascii=False
        )
        self.institutions.writerow(dic)

        # ids
        if institution_ids := dic.get("ids"):
            institution_ids["institution_id"] = institution_id
            self.ids.writerow(institution_ids)

        # geo
        if institution_geo := dic.get("geo"):
            institution_geo["institution_id"] = institution_id
            self.geo.writerow(institution_geo)

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

        # counts_by_year
        if counts_by_year := dic.get("counts_by_year"):
            for count_by_year in counts_by_year:
                count_by_year["institution_id"] = institution_id
                self.counts_by_year.writerow(count_by_year)


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
            "sources_api_url",
            "updated_date",
        ]
    )
    counts_by_year = MWriter(["publisher_id", "year", "works_count", "cited_by_count"])
    ids = MWriter(["publisher_id", "openalex", "ror", "wikidata"])

    def write(self, dic: dict):
        publisher_id = dic["id"]
        dic["alternate_titles"] = json.dumps(
            dic.get("alternate_titles"), ensure_ascii=False
        )
        dic["country_codes"] = json.dumps(dic.get("country_codes"), ensure_ascii=False)
        self.publishers.writerow(dic)

        if publisher_ids := dic.get("ids"):
            publisher_ids["publisher_id"] = publisher_id
            self.ids.writerow(publisher_ids)

        if counts_by_year := dic.get("counts_by_year"):
            for count_by_year in counts_by_year:
                count_by_year["publisher_id"] = publisher_id
                self.counts_by_year.writerow(count_by_year)


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
            "works_api_url",
            "updated_date",
        ]
    )
    ids = MWriter(
        ["source_id", "openalex", "issn_l", "issn", "mag", "wikidata", "fatcat"]
    )
    counts_by_year = MWriter(["source_id", "year", "works_count", "cited_by_count"])

    def write(self, dic: dict):
        source_id = dic["id"]
        dic["issn"] = json.dumps(dic.get("issn"))
        self.sources.writerow(dic)

        if source_ids := dic.get("ids"):
            source_ids["source_id"] = source_id
            source_ids["issn"] = json.dumps(source_ids.get("issn"))
            self.ids.writerow(source_ids)

        if counts_by_year := dic.get("counts_by_year"):
            for count_by_year in counts_by_year:
                count_by_year["source_id"] = source_id
                self.counts_by_year.writerow(count_by_year)


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
            "cited_by_api_url",
            "abstract_inverted_index",
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

        if (abstract := dic.get("abstract_inverted_index")) is not None:
            dic["abstract_inverted_index"] = json.dumps(abstract, ensure_ascii=False)

        self.works.writerow(dic)

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
                    institution_ids = [i.get("id") for i in institutions]
                    institution_ids = [i for i in institution_ids if i]
                    institution_ids = institution_ids or [None]

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

        # biblio
        if biblio := dic.get("biblio"):
            biblio["work_id"] = work_id
            self.biblio.writerow(biblio)

        # concepts
        for concept in dic.get("concepts"):
            if concept_id := concept.get("id"):
                self.concepts.writerow(
                    {
                        "work_id": work_id,
                        "concept_id": concept_id,
                        "score": concept.get("score"),
                    }
                )

        # ids
        if ids := dic.get("ids"):
            ids["work_id"] = work_id
            self.ids.writerow(ids)

        # mesh
        for mesh in dic.get("mesh"):
            mesh["work_id"] = work_id
            self.mesh.writerow(mesh)

        # open_access
        if open_access := dic.get("open_access"):
            open_access["work_id"] = work_id
            self.open_access.writerow(open_access)

        # referenced_works
        for referenced_work in dic.get("referenced_works", []):
            if referenced_work:
                self.referenced_works.writerow(
                    {"work_id": work_id, "referenced_work_id": referenced_work}
                )

        # related_works
        for related_work in dic.get("related_works", []):
            if related_work:
                self.related_works.writerow(
                    {"work_id": work_id, "related_work_id": related_work}
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
    big_run("work", WorksWriter)


POISON_PILL = None


def meta_partition_writer(
    partition_name, parent_dir, queue: mp.Queue, mode: str = "wt", kls=WorksWriter
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
        for dic in tqdm(uiter(k)):
            writer.write(dic)


def big_run(key: str, kls: type):
    partition_dicts(
        tqdm(file_iter(key)),
        partition_key="id",
        num_partitions=N_GROUPS,
        director_count=10,
        parent_dir=PARTITIONED_CSV_PATH / key,
        slot_per_partition=4_500,
        batch_size=1500,
        writer_function=partial(meta_partition_writer, kls=kls),
        main_queue_filler=para_queue_filler,
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


def uiter(k):
    seen_ids = set()
    for dic in dic_iter(k):
        _id = dic["id"]
        if _id in seen_ids:
            continue
        seen_ids.add(_id)
        yield dic
