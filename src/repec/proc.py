import re
from itertools import islice
from multiprocessing import cpu_count

import aswan
import datazimmer as dz
import pandas as pd
from atqo import parallel_map
from bs4 import BeautifulSoup

from .collect import HistoryHandler, NepBase, PaperHandler, RepecProject
from .meta import (
    Author,
    Authorship,
    NepInclusion,
    NepIssue,
    Paper,
    author_table,
    authorship_table,
    nep_inclusion_table,
    nep_issue_table,
    nep_table,
    paper_table,
    stat_base,
)

N_WORKERS = cpu_count() // 3 + 1


@dz.register_data_loader(extra_deps=[RepecProject])
def load():

    proj = RepecProject()

    for nep_out in proj.get_unprocessed_events(NepBase):
        nep_df = nep_out.content
        break

    nep_table.replace_records(nep_df)

    cev_iter = proj.get_unprocessed_events(HistoryHandler)

    paper_rel_df = pd.concat(
        parallel_map(get_paper_rel_df, cev_iter, pbar=True, workers=N_WORKERS)
    )

    nep_issue_table.replace_records(
        paper_rel_df.rename(
            columns={"nepis": NepIssue.neid, "nep": NepIssue.nep.nid}
        ).drop_duplicates(subset=[NepIssue.neid])
    )
    nep_inclusion_table.replace_records(
        paper_rel_df.rename(
            columns={
                "nepis": NepInclusion.issue.neid,
                "pid": NepInclusion.paper.pid,
            }
        ).drop_duplicates(subset=nep_inclusion_table.index_cols)
    )
    dump_paper_meta(proj)


@dz.register_env_creator
def make_envs(abstract_chars, min_papers_per_author):
    au_df = (
        authorship_table.get_full_df()
        .assign(c=1)
        .groupby(Authorship.author.aid)
        .transform("sum")
        .loc[lambda df: df["c"] >= min_papers_per_author]
    )
    pids = au_df.index.get_level_values(Authorship.paper.pid).unique().to_numpy()
    aids = au_df.index.get_level_values(Authorship.author.aid).unique().to_numpy()
    paper_df = (
        paper_table.get_full_df()
        .loc[pids, :]
        .assign(**{Paper.abstract: lambda df: df[Paper.abstract].str[:abstract_chars]})
    )
    neinc_df = nep_inclusion_table.get_full_df().loc[
        lambda df: df[NepInclusion.paper.pid].isin(set(pids)), :
    ]
    dz.dump_dfs_to_tables(
        [
            (au_df, authorship_table),
            (paper_df, paper_table),
            (author_table.get_full_df().loc[aids, :], author_table),
            (nep_table.get_full_df(), nep_table),
            (nep_issue_table.get_full_df(), nep_issue_table),
            (neinc_df, nep_inclusion_table),
        ]
    )


def get_paper_rel_df(cev: aswan.ParsedCollectionEvent):
    soup = BeautifulSoup(cev.content, "xml")
    paper_as = soup.find_all("a", href=re.compile("/paper/.*/.*"))
    if not paper_as:
        return pd.DataFrame()
    return (
        pd.DataFrame({"p_link": [a["href"] for a in paper_as]})
        .assign(
            ind=lambda df: range(df.shape[0]),
            h_link=cev.url,
            pid=lambda df: df["p_link"].pipe(paper_link_to_id),
        )
        .pipe(_extract_from_h_link)
        .assign(
            nepis=lambda df: df["nep"] + "-" + df["published"],
            ind=lambda df: df["ind"] + df["page"],
        )
    )


def _extract_from_h_link(df):
    ext1 = r"search.pf\?neplist=(?P<nep>.*)(?P<published>\d\d\d\d\-\d\d-\d\d)"
    ext2 = r";pg=(?P<page>\d*)"
    dfs = [
        df,
        df["h_link"].str.extract(ext1),
        df["h_link"].str.extract(ext2).fillna("0").astype(int),
    ]
    return pd.concat(dfs, axis=1)


def dump_paper_meta(project: dz.DzAswan):
    cev_iter = project.get_unprocessed_events(PaperHandler)
    paper_dics = parallel_map(get_paper_dic, cev_iter, pbar=True, workers=N_WORKERS)
    paper_meta = (
        pd.DataFrame(paper_dics)
        .assign(**{Paper.pid: lambda df: paper_link_to_id(df["paper_link"])})
        .rename(columns={"paper_link": Paper.link})
        .set_index(Paper.pid)
        .rename(
            columns=lambda s: s.replace("citation_", "").replace(
                "technical_report_", ""
            )
        )
    )

    paper_table.replace_records(paper_meta.reindex(paper_table.feature_cols, axis=1))
    usc = "authors"
    if usc not in paper_meta.columns:
        return
    proc_authors(
        paper_meta[usc]
        .dropna()
        .str.split("; ", expand=True)
        .unstack()
        .dropna()
        .reset_index(level=0, drop=True)
        .reset_index()
        .drop_duplicates()
    )


def get_paper_dic(cev: aswan.ParsedCollectionEvent):
    soup = BeautifulSoup(cev.content, "xml")
    stat_link = soup.find("a", href=re.compile(f"{stat_base}/scripts/paperstat.pf.*"))
    dic = {m.get("name"): m["content"] for m in soup.find_all("meta") if m.get("name")}
    return {"stat_link": (stat_link or {}).get("href"), "paper_link": cev.url} | dic


def proc_authors(rels: pd.DataFrame):
    base_df = rels.assign(
        **{Author.aid: lambda df: df.loc[:, 0].str.lower().str.replace(", ", ":")}
    )
    author_table.replace_records(
        base_df.drop_duplicates(subset=[Author.aid]).rename(columns={0: Author.name})
    )
    authorship_table.replace_records(
        base_df.reset_index().rename(
            columns={
                Author.aid: Authorship.author.aid,
                Paper.pid: Authorship.paper.pid,
            }
        )
    )


def paper_link_to_id(s):
    return s.str.extract(r"/paper/(.*)\.htm")
