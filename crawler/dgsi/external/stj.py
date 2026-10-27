"""Crawl the STJ jurisprudence portal (juris.stj.pt) into MongoDB ``files.stj``."""

import argparse
import asyncio

import aiohttp
from bs4 import BeautifulSoup

from db import connect

BASE_URL = "https://juris.stj.pt"


def parse_document(html: str) -> dict[str, str]:
    container = BeautifulSoup(html, "html.parser").find(class_="container-fluid")
    data = {}

    for row in container.find_all(class_="row border-bottom"):
        key = row.find(class_="col-1").get_text(strip=True).replace(":", "")
        col_7 = row.find(class_="col-7")
        col_11 = row.find(class_="col-11")

        if col_7:
            data[key] = col_7.get_text(strip=True)
        elif col_11:
            a_tag = col_11.find("a")
            data[key] = (a_tag or col_11).get_text(strip=True)

    sections = container.find_all(class_="p-2")
    if len(sections) >= 2:
        data["Sumário"] = sections[0].get_text(separator=" ", strip=True)
        data["Texto Integral"] = sections[1].get_text(separator=" ", strip=True)

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
    collection = connect().files.stj

    while True:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/api/search?page={page}", ssl=False) as response:
                if response.status != 200:
                    print(f"Search request failed with status {response.status}. Retrying...")
                    await asyncio.sleep(1)
                    continue
                items = await response.json()

        if not items:
            print("No more results. Exiting...")
            break

        numbers = [item["_source"].get("Número de Processo") for item in items if item["_source"]]
        tasks = [fetch_document(f"{BASE_URL}/{number.replace('/', '%2F')}") for number in numbers if number]
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
    parser.add_argument("--start-page", type=int, default=0, help="first search page (0-based)")
    asyncio.run(fetch_documents(parser.parse_args().start_page))


if __name__ == "__main__":
    main()
