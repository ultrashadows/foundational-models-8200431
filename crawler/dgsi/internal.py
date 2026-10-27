"""Crawl one DGSI (www.dgsi.pt) jurisprudence database into MongoDB (database ``files``).

Every DGSI database is a Lotus Notes view with the same layout, so a single crawler
covers all of them. Each court is stored in a collection named after its database.
"""

import argparse
import asyncio

import aiohttp
from bs4 import BeautifulSoup

from crawler.dgsi.helpers import fetch_page
from db import connect

BASE_URL = "https://www.dgsi.pt"
COURTS = [
    "atco1", "cajp", "jcon", "jdgpj", "jsta", "jstj", "jtca", "jtcampca",
    "jtcampct", "jtcn", "jtrc", "jtre", "jtrg", "jtrl", "jtrp",
]
# JSTA documents keep the full text in a collapsed section that must be expanded.
URL_SUFFIX = {"jsta": "&ExpandSection=1#_Section1"}
# Cells with this background are section headers, not field values.
HEADER_BGCOLOR = "#71B2CF"


def parse_document(html: str) -> dict[str, str]:
    data = {}
    for row in BeautifulSoup(html, "html.parser").find_all("tr"):
        cells = row.find_all("td")
        if len(cells) == 2:
            key = cells[0].get_text(strip=True).rstrip(":").strip()
            if key and cells[1].get("bgcolor") != HEADER_BGCOLOR:
                data[key] = cells[1].get_text().strip()
    return data


async def fetch_document(url: str, session: aiohttp.ClientSession) -> dict[str, str]:
    async with session.get(url) as response:
        return parse_document(await response.text(encoding="ISO-8859-1"))


async def crawl(court: str) -> None:
    collection = connect().files[court]
    url = f"{BASE_URL}/{court}.nsf?OpenDatabase"
    suffix = URL_SUFFIX.get(court, "")
    total = 0
    async with aiohttp.ClientSession() as session:
        while True:
            document_urls, next_url = await fetch_page(url, session)

            tasks = [fetch_document(BASE_URL + doc_url + suffix, session) for doc_url in document_urls]
            documents = await asyncio.gather(*tasks)

            if documents:
                collection.insert_many(documents)
            total += len(documents)
            print("Inserted", len(documents), "documents into the database. Total:", total)

            if next_url and BASE_URL + next_url != url:
                url = BASE_URL + next_url
            else:
                print("No more pages")
                break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("court", choices=COURTS, help="DGSI database to crawl")
    asyncio.run(crawl(parser.parse_args().court))


if __name__ == "__main__":
    main()
