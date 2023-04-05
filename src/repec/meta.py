import datetime as dt

import datazimmer as dz


class Nep(dz.AbstractEntity):
    nid = dz.Index & str
    title = str
    info = str


class Paper(dz.AbstractEntity):
    pid = dz.Index & str

    link = str
    year = dz.Nullable(float)
    abstract = dz.Nullable(str)
    title = str
    institution = dz.Nullable(str)
    keywords = dz.Nullable(str)


class Author(dz.AbstractEntity):
    aid = dz.Index & str
    name = str


class NepIssue(dz.AbstractEntity):
    neid = dz.Index & str

    nep = Nep
    published = dt.datetime


class NepInclusion(dz.AbstractEntity):
    ind = dz.Index & int
    issue = dz.Index & NepIssue

    paper = Paper


class Authorship(dz.AbstractEntity):
    paper = dz.Index & Paper
    author = dz.Index & Author


class KeywordCategorization(dz.AbstractEntity):
    paper = Paper
    keyword = str


nep_table = dz.ScruTable(Nep)
paper_table = dz.ScruTable(Paper)
author_table = dz.ScruTable(Author)
authorship_table = dz.ScruTable(Authorship)
nep_issue_table = dz.ScruTable(NepIssue)
nep_inclusion_table = dz.ScruTable(NepInclusion)

econpaper_base = dz.SourceUrl("https://econpapers.repec.org")
nep_base = dz.SourceUrl("http://nep.repec.org/")
stat_base = dz.SourceUrl("https://logec.repec.org")
