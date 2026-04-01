import argparse
import uuid
from collections import deque
from typing import Deque, List, Set
from urllib.parse import urljoin, urlparse

import chromadb
import requests
from bs4 import BeautifulSoup


def _same_site(url: str, base_netloc: str) -> bool:
    return urlparse(url).netloc == base_netloc


def _extract_text(soup: BeautifulSoup) -> str:
    for tag in soup.find_all(["header", "footer", "script", "style", "noscript"]):
        tag.decompose()

    parts: List[str] = []
    for tag in soup.find_all(["article", "p", "div"]):
        text = tag.get_text(" ", strip=True)
        if text and len(text) > 40:
            parts.append(text)
    return "\n".join(parts).strip()


def scrape_and_ingest(start_url: str, chroma_path: str, max_pages: int) -> int:
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(name="imex_knowledge")

    parsed = urlparse(start_url)
    base_netloc = parsed.netloc

    queue: Deque[str] = deque([start_url])
    visited: Set[str] = set()
    added = 0

    session = requests.Session()
    session.headers.update({"User-Agent": "imexassist-bot-scraper/1.0"})

    while queue and len(visited) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        try:
            response = session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
        except Exception as exc:
            print(f"Skipping {url}: {exc}")
            continue

        clean_text = _extract_text(soup)
        if clean_text:
            collection.add(
                ids=[str(uuid.uuid4())],
                documents=[clean_text],
                metadatas=[{"source": "website", "url": url}],
            )
            added += 1

        for anchor in soup.find_all("a", href=True):
            next_url = urljoin(url, anchor["href"]).split("#")[0]
            if _same_site(next_url, base_netloc) and next_url not in visited:
                queue.append(next_url)

    return added


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape IMEX site and ingest into ChromaDB.")
    parser.add_argument("--url", default="https://imex-mce-mbt.com/", help="Start URL.")
    parser.add_argument("--chroma-dir", default="data/chroma", help="ChromaDB persistence path.")
    parser.add_argument("--max-pages", type=int, default=30, help="Max pages to crawl.")
    args = parser.parse_args()

    total = scrape_and_ingest(args.url, args.chroma_dir, args.max_pages)
    print(f"Scrape complete. Added {total} website entries to imex_knowledge.")


if __name__ == "__main__":
    main()
