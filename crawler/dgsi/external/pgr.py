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

                data = {}

                paragraph_administrativos = soup.find(id="paragraph-administrativos")
                if paragraph_administrativos:
                    fields = paragraph_administrativos.find_all('div', class_='field')

                    for field in fields:
                        label = field.find('div', class_='field__label').get_text(strip=True)
                        value = field.find('div', class_='field__value').get_text(strip=True, separator=', ')
                        data[label] = value
                else:
                    print(f"No paragraph-administrativos found on url {url}")

                paragraph_descritores = soup.find(id="paragraph-descritores")
                if paragraph_descritores:
                    fields = paragraph_descritores.find_all('div', class_='field')

                    for field in fields:
                        label = field.find('div', class_='field__label').get_text(strip=True)
                        value = field.find('div', class_='field__value').get_text(strip=True, separator=', ')
                        data[label] = value
                else:
                    print(f"No paragraph-descritores found on url {url}")

                paragraph_texto = soup.find(id="paragraph-texto")
                if paragraph_texto:
                    fields = paragraph_texto.find_all('div', class_='field')

                    for field in fields:
                        label = field.find('div', class_='field__label').get_text(strip=True)
                        value = field.find('div', class_='field__value').get_text(strip=True, separator=', ')
                        data[label] = value
                else:
                    print(f"No paragraph-texto found on url {url}")

                paragraph_anotacoes = soup.find(id="paragraph-anotacoes")
                if paragraph_anotacoes:
                    fields = paragraph_anotacoes.find_all('div', class_='field')

                    for field in fields:
                        label = field.find('div', class_='field__label').get_text(strip=True)
                        value = field.find('div', class_='field__value').get_text(strip=True, separator=', ')
                        data[label] = value
                else:
                    print(f"No paragraph-anotacoes found on url {url}")

                json_data = json.dumps(data, indent=4, ensure_ascii=False)
                return json.loads(json_data)


async def fetch_documents(page=0):
    all_documents = []
    collection = connect().files.pgr

    while True:
        url = f"https://ministeriopublico.pt/pareceres-pgr?page={page}"
        print(url)
        async with aiohttp.ClientSession() as session:
            async with session.get(url, ssl=False) as response:
                html_content = await response.text()
                soup = BeautifulSoup(html_content, 'html.parser')

                rows = soup.find_all(class_=['odd', 'even'])
                pgr_links = []
                if rows:
                    for row in rows:
                        a_tag = row.find('a', href=True)
                        if a_tag:
                            pgr_links.append(a_tag['href'])
                    tasks = [fetch_document(f'https://ministeriopublico.pt{pgr}') for pgr in pgr_links]
                    results = await asyncio.gather(*tasks)

                    all_documents.extend([result for result in results if result])

                    if not all_documents:
                        print(f"No more documents to fetch. Retrying...")
                    else:
                        collection.insert_many(all_documents)
                        print(f"{page} - Inserted {len(all_documents)} documents into MongoDB.")
                        all_documents = []
                        page += 1
                else:
                    print(f"No rows found on page {page}")
                    break


if __name__ == "__main__":
    asyncio.run(fetch_documents(page=1))
