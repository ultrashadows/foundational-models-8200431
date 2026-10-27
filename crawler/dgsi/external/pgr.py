"""Crawl the PGR legal opinions (ministeriopublico.pt/pareceres-pgr) into MongoDB ``files.pgr``."""

import argparse
import asyncio

import aiohttp
from bs4 import BeautifulSoup

from db import connect

BASE_URL = "https://ministeriopublico.pt"
SECTIONS = ["paragraph-administrativos", "paragraph-descritores", "paragraph-texto", "paragraph-anotacoes"]


def parse_document(html: str, url: str = "") -> dict[str, str]:
    data = {}
    soup = BeautifulSoup(html, "html.parser")
    for section_id in SECTIONS:
        section = soup.find(id=section_id)
        if not section:
            print(f"No {section_id} found on url {url}")
            continue
        for field in section.find_all("div", class_="field"):
            label = field.find("div", class_="field__label").get_text(strip=True)
            data[label] = field.find("div", class_="field__value").get_text(strip=True, separator=", ")
    return data


async def fetch_document(url: str) -> dict[str, str] | None:
    # TLS verification is disabled because the site's certificate chain did not validate.
    async with aiohttp.ClientSession() as session:
        async with session.get(url, ssl=False) as response:
            if response.status != 200:
                return None
            return parse_document(await response.text(), url)


async def fetch_documents(page: int) -> None:
    collection = connect().files.pgr

    while True:
        url = f"{BASE_URL}/pareceres-pgr?page={page}"
        print(url)
        async with aiohttp.ClientSession() as session:
            async with session.get(url, ssl=False) as response:
                soup = BeautifulSoup(await response.text(), "html.parser")

        rows = soup.find_all(class_=["odd", "even"])
        if not rows:
            print(f"No rows found on page {page}")
            break

        links = [a["href"] for row in rows if (a := row.find("a", href=True))]
        documents = [doc for doc in await asyncio.gather(*(fetch_document(BASE_URL + link) for link in links)) if doc]

        if not documents:
            print("No documents fetched from this page. Retrying...")
        else:
            collection.insert_many(documents)
            print(f"{page} - Inserted {len(documents)} documents into MongoDB.")
            page += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-page", type=int, default=0, help="first listing page (0-based)")
    asyncio.run(fetch_documents(parser.parse_args().start_page))


if __name__ == "__main__":
    main()
