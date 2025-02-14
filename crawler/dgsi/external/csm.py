import asyncio
import json
import aiohttp
import requests
from bs4 import BeautifulSoup
from crawler.dgsi.helpers import connect


async def fetch_document(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url, ssl=False) as response:
            if response.status == 200:
                data = {}

                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')

                title = soup.find('h3', id='ecli-title')

                if title:
                    data['document'] = title.get_text(strip=True)

                descriptors_div = soup.find('div', class_='descriptors')

                if descriptors_div:
                    for h4 in descriptors_div.find_all('h4'):
                        content_title = h4.find('strong', class_='content-title')
                        content = h4.find('span', class_='content')

                        if content_title and content:
                            key = content_title.get_text(strip=True).replace(':', '')
                            value = content.get_text(strip=True)

                            data[key] = value

                summary_div = soup.find('div', id='summary')

                if summary_div:
                    summary_content = " ".join([font.get_text(strip=True) for font in summary_div.find_all('font')])

                    data['Sumário'] = summary_content if summary_content else ''

                partial_text_div = soup.find('div', id='parcial-text')

                if partial_text_div:
                    partial_text_content = " ".join([font.get_text(strip=True) for font in partial_text_div
                                                    .find_all('font')])

                    data['Decisão Parcial'] = partial_text_content if partial_text_content else ''

                integral_text_div = soup.find('div', id='integral-text')

                if integral_text_div:
                    integral_text_content = " ".join([font.get_text(strip=True) for font in integral_text_div
                                                      .find_all('font')])

                    data['Decisão Integral'] = integral_text_content if integral_text_content else ''

                json_data = json.dumps(data, indent=4, ensure_ascii=False)
                return json.loads(json_data)
            else:
                print(f"Failed to fetch document from {url}.")
                return await fetch_documents(url)


async def fetch_documents(page=2263):
    all_documents = []
    collection = connect().files.csm

    while True:
        url = f"https://jurisprudencia.csm.org.pt/items/loadItems?page={page}&perPage=100&offset={(page - 1) * 100}"
        response = requests.get(url, verify=False)

        if response.status_code == 200:
            data = response.json()
            records = data['records']

            if records:
                fetch_urls = []

                for record in records:
                    fetch_urls.append(f"https://jurisprudencia.csm.org.pt/ecli/{record['ecli']}/")

                tasks = [fetch_document(url) for url in fetch_urls]
                results = await asyncio.gather(*tasks)

                all_documents.extend([result for result in results if result])

                if not all_documents:
                    print(f"No more documents to fetch. Retrying...")
                else:
                    collection.insert_many(all_documents)
                    print(f"{page} - Inserted {len(all_documents)} documents into MongoDB.")
                    all_documents = []
                    page += 1
                await asyncio.sleep(1)
            else:
                print(f"No more records to fetch. Exiting...")
                break

    return


if __name__ == "__main__":
    asyncio.run(fetch_documents())
