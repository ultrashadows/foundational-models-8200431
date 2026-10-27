import aiohttp
from bs4 import BeautifulSoup


async def fetch_page(url: str, session: aiohttp.ClientSession) -> tuple[list[str], str | None]:
    """Return the document links on a DGSI listing page and the link to the next page."""
    urls = []
    next_url = None
    async with session.get(url) as response:
        content = await response.text()
    soup = BeautifulSoup(content, "html.parser")
    for link in soup.find_all("a"):
        href = link.get("href")
        if link.text == "Seguinte":
            next_url = href
        if href and href.endswith("OpenDocument"):
            urls.append(href)
    return urls, next_url
