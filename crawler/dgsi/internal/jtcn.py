import asyncio
import aiohttp
from bs4 import BeautifulSoup
from crawler.dgsi.helpers import fetch_page, connect


async def fetch_document(url: str, session: aiohttp.ClientSession) -> dict[str, str]:
    async with session.get(url) as response:
        content = await response.text(encoding='ISO-8859-1')
        soup = BeautifulSoup(content, 'html.parser')

        rows = soup.find_all('tr')
        data_dict = {}

        for row in rows:
            cells = row.find_all('td')
            if len(cells) == 2:
                key = cells[0].get_text(strip=True).rstrip(':').strip()
                value = cells[1].get_text()

                if key and cells[1].get('bgcolor') != "#71B2CF":
                    data_dict[key] = value.strip()

    return data_dict


async def main():
    collection = connect().files.jtcn
    base_url = 'https://www.dgsi.pt/jtcn.nsf?OpenDatabase'
    url = base_url
    total = 0
    async with aiohttp.ClientSession() as session:
        while True:
            document_urls, next_url = await fetch_page(url)

            tasks = [fetch_document('https://www.dgsi.pt' + doc_url, session) for doc_url in document_urls]
            documents = await asyncio.gather(*tasks)

            collection.insert_many(documents)
            total += len(documents)
            print("Inserted", len(documents), "documents into the database. Total:", total)

            if next_url and ('https://www.dgsi.pt' + next_url) != url:
                url = 'https://www.dgsi.pt' + next_url
            else:
                print("No more pages")
                break


if __name__ == "__main__":
    asyncio.run(main())
