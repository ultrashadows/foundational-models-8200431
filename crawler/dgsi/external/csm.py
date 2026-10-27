"""Crawl the CSM jurisprudence portal (jurisprudencia.csm.org.pt) into MongoDB ``files.csm``."""

import argparse
import asyncio

import aiohttp
import requests
from bs4 import BeautifulSoup

from db import connect

BASE_URL = "https://jurisprudencia.csm.org.pt"
PAGE_SIZE = 100


def _join_fonts(div) -> str:
    return " ".join(font.get_text(strip=True) for font in div.find_all("font"))


def parse_document(html: str) -> dict[str, str]:
    data = {}
    soup = BeautifulSoup(html, "html.parser")

    title = soup.find("h3", id="ecli-title")
    if title:
        data["document"] = title.get_text(strip=True)

    descriptors_div = soup.find("div", class_="descriptors")
    if descriptors_div:
        for h4 in descriptors_div.find_all("h4"):
            content_title = h4.find("strong", class_="content-title")
            content = h4.find("span", class_="content")
            if content_title and content:
                data[content_title.get_text(strip=True).replace(":", "")] = content.get_text(strip=True)

    for div_id, key in [("summary", "Sumário"), ("parcial-text", "Decisão Parcial"),
                        ("integral-text", "Decisão Integral")]:
        div = soup.find("div", id=div_id)
        if div:
            data[key] = _join_fonts(div)

    return data


async def fetch_document(url: str) -> dict[str, str] | None:
    # TLS verification is disabled because the portal's certificate chain did not validate.
    async with aiohttp.ClientSession() as session:
        async with session.get(url, ssl=False) as response:
            if response.status != 200:
                print(f"Failed to fetch document from {url}.")
                return None
            return parse_document(await response.text())


async def fetch_documents(page: int) -> None:
    collection = connect().files.csm

    while True:
        url = f"{BASE_URL}/items/loadItems?page={page}&perPage={PAGE_SIZE}&offset={(page - 1) * PAGE_SIZE}"
        response = requests.get(url, verify=False)
        if response.status_code != 200:
            print(f"Listing request failed with status {response.status_code}. Retrying...")
            await asyncio.sleep(1)
            continue

        records = response.json()["records"]
        if not records:
            print("No more records to fetch. Exiting...")
            break

        tasks = [fetch_document(f"{BASE_URL}/ecli/{record['ecli']}/") for record in records]
        documents = [doc for doc in await asyncio.gather(*tasks) if doc]

        if not documents:
            print("No documents fetched from this page. Retrying...")
        else:
            collection.insert_many(documents)
            print(f"{page} - Inserted {len(documents)} documents into MongoDB.")
            page += 1
        await asyncio.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-page", type=int, default=1, help="first listing page (1-based)")
    asyncio.run(fetch_documents(parser.parse_args().start_page))


if __name__ == "__main__":
    main()
