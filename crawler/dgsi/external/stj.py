import asyncio
import json
import aiohttp
from bs4 import BeautifulSoup
from crawler.dgsi.helpers import connect


async def fetch_document(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url, ssl=False) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                container = soup.find(class_='container-fluid')

                data = {}

                for row in container.find_all(class_='row border-bottom'):
                    key = row.find(class_='col-1').get_text(strip=True).replace(':', '')
                    col_7 = row.find(class_='col-7')
                    col_11 = row.find(class_='col-11')

                    if col_7:
                        value = col_7.get_text(strip=True)
                    elif col_11:
                        a_tag = col_11.find('a')
                        if a_tag:
                            value = a_tag.get_text(strip=True)
                        else:
                            value = col_11.get_text(strip=True)

                    data[key] = value

                sections = container.find_all(class_='p-2')
                if len(sections) >= 2:
                    summary_text = sections[0].get_text(separator=" ", strip=True)
                    texto_integral_text = sections[1].get_text(separator=" ", strip=True)

                    data["Sumário"] = summary_text
                    data["Texto Integral"] = texto_integral_text

                json_data = json.dumps(data, indent=4, ensure_ascii=False)
                return json.loads(json_data)
            else:
                print(f"Failed to fetch document from {url}.")
                return await fetch_documents(url)


async def fetch_documents(page=0):
    all_documents = []
    collection = connect().files.stj

    while True:
        url = f"https://juris.stj.pt/api/search?page={page}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, ssl=False) as response:
                if response.status == 200:
                    data = await response.json()
                    url_list = []
                    for item in data:
                        if item['_source']:
                            link = item['_source'].get('Número de Processo')
                            if link:
                                url_list.append(link)

                    tasks = [fetch_document(f'https://juris.stj.pt/{stj.replace("/", "%2F")}') for stj in url_list]
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


if __name__ == "__main__":
    asyncio.run(fetch_documents(page=6889))
