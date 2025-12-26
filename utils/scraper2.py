import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
from tqdm import tqdm
import time

BASE_URL = "https://talk.newagtalk.com/forums/"
FORUM_URL = "forum-view.asp?fid=12"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"
}

def get_soup(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def get_forum_pages():
    pages = []
    page = 1

    while True:
        url = urljoin(BASE_URL, f"{FORUM_URL}&p={page}")
        soup = get_soup(url)

        threads = soup.select("a.threadlink")
        if not threads:
            break

        pages.append(url)
        page += 1
        time.sleep(1)

    return pages

def extract_thread_links(forum_page_url):
    soup = get_soup(forum_page_url)
    links = []

    for a in soup.select("a.threadlink"):
        href = a.get("href")
        if href and "thread-view.asp" in href:
            full_url = urljoin(BASE_URL, href)
            links.append(full_url)

    return list(set(links))

def scrape_thread(thread_url):
    soup = get_soup(thread_url)

    title_tag = soup.find("span", id=lambda x: x and x.startswith("tmid"))
    title = title_tag.get_text(strip=True) if title_tag else "UNKNOWN"

    posts = []
    for td in soup.select("td.messagemiddle"):
        text = td.get_text(" ", strip=True)
        if text:
            posts.append(text)

    return title, posts

def main():
    data = []

    print("Finding forum pages...")
    forum_pages = get_forum_pages()

    print(f"Found {len(forum_pages)} forum pages")

    thread_urls = set()
    for page in forum_pages:
        thread_urls.update(extract_thread_links(page))

    print(f"Found {len(thread_urls)} unique threads")

    for thread_url in tqdm(thread_urls, desc="Scraping threads"):
        try:
            title, posts = scrape_thread(thread_url)
            for post in posts:
                data.append({
                    "thread_title": title,
                    "post_content": post,
                    "thread_url": thread_url
                })
            time.sleep(1)
        except Exception as e:
            print(f"Failed {thread_url}: {e}")

    df = pd.DataFrame(data)
    df.to_csv("newagtalk_forum_fid12.csv", index=False)
    print("Saved newagtalk_forum_fid12.csv")

if __name__ == "__main__":
    main()
