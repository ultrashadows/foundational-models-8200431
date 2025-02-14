import aiohttp
from bs4 import BeautifulSoup
from pymongo import MongoClient


# Returns the active MongoDB connection on the given port
def connect():
    return MongoClient('localhost', 27017)


async def fetch_page(url: str) -> tuple[list[str], str]:
    urls = []
    next_url = None
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            content = await response.text()
            soup = BeautifulSoup(content, 'html.parser')
            links = soup.find_all('a')
            for link in links:
                if link.text == "Seguinte":
                    next_url = link.get('href')
                href = link.get('href')
                if href and href.endswith("OpenDocument"):
                    urls.append(href)
    return urls, next_url
