import re

import aswan
import datazimmer as dz
import pandas as pd
from bs4 import BeautifulSoup

from .meta import Nep, econpaper_base, nep_base


class NepBase(aswan.RequestHandler):
    def parse(self, blob):
        df = pd.DataFrame(parse_nep_blob(blob))
        urls = [
            f"{econpaper_base}/scripts/nep.pf?list={nep_id}" for nep_id in df[Nep.nid]
        ]
        self.register_links_to_handler(urls, ArchiveHandler)

        return df


class ArchiveHandler(aswan.RequestHandler):
    max_in_parallel = 30

    def parse(self, blob):

        soup = BeautifulSoup(blob, "xml")

        urls = []
        for a in soup.find_all("a", href=re.compile(r"/scripts/search.pf\?neplist=.*")):
            archive_link = a["href"]
            urls.append(f"{econpaper_base}{archive_link};iframes=no")
        self.register_links_to_handler(urls, HistoryHandler)
        return urls


class HistoryHandler(aswan.RequestHandler):
    process_indefinitely = True
    url_root = econpaper_base
    max_in_parallel = 20

    def parse(self, blob):
        soup = BeautifulSoup(blob, "xml")
        paper_a_list = soup.find_all("a", href=re.compile("/paper/.*/.*"))
        urls = [a["href"] for a in paper_a_list]
        self.register_links_to_handler(urls, PaperHandler)
        next_page_button = soup.find("img", class_="rightarrow", src="/right.png")
        if next_page_button is not None:
            self.register_links_to_handler([next_page_button.parent["href"]])
        return blob


class PaperHandler(aswan.RequestHandler):
    process_indefinitely = True
    max_in_parallel = 20


class RepecProject(dz.DzAswan):
    name: str = "repec-nep"
    cron = "0 8 * * 1"
    starters = {NepBase: [nep_base]}


def parse_nep_blob(blob: bytes):
    for _nep in BeautifulSoup(blob, "xml").find_all("div", class_="nitpo_antem"):
        info_txt = _nep.find_all("span")[2].text
        yield {
            Nep.nid: _nep.find_all("span")[1].text.strip(),
            Nep.title: info_txt.split(",")[0].strip(),
            Nep.info: ", ".join(info_txt.split(", ")[1:]).strip(),
        }
