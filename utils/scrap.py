import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
import time
import logging
from tqdm import tqdm

# ---------------- CONFIG ---------------- #

BASE_URL = "https://talk.newagtalk.com/forums/"
FORUM_PATH = "forum-view.asp?fid=12"

MAX_FORUM_PAGES = 50      # hard safety limit
REQUEST_DELAY = 1.0       # seconds between requests
TIMEOUT = 30

HEADERS = {
    "User-Agent": "Mozilla/5.0 (research-bot; contact: you@example.com)"
}

OUTPUT_CSV = "newagtalk_fid12_postssss.csv"

# ---------------- LOGGING ---------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("scraper")

# ---------------- HELPERS ---------------- #

def get_soup(url):
    log.debug(f"Fetching URL: {url}")
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

# ---------------- FORUM PAGES ---------------- #

def get_forum_pages():
    pages = []
    page = 1

    log.info("Starting forum page discovery")

    while page <= MAX_FORUM_PAGES:
        url = urljoin(BASE_URL, f"{FORUM_PATH}&p={page}")
        log.info(f"Checking forum page {page}: {url}")

        soup = get_soup(url)
        threads = soup.select("a.threadlink")

        if not threads:
            log.info("No threads found — stopping page discovery")
            break

        pages.append(url)
        log.info(f"✓ Page {page} valid ({len(threads)} thread links found)")
        page += 1
        time.sleep(REQUEST_DELAY)

    log.info(f"Discovered {len(pages)} forum pages total")
    return pages

# ---------------- THREAD LINKS ---------------- #

def extract_thread_links(forum_page_url):
    soup = get_soup(forum_page_url)
    links = set()

    for a in soup.select("a.threadlink"):
        href = a.get("href")
        if href and "thread-view.asp" in href:
            full_url = urljoin(BASE_URL, href)
            links.add(full_url)

    log.info(f"Extracted {len(links)} unique thread URLs from page")
    return links

# ---------------- THREAD CONTENT ---------------- #

def scrape_thread(thread_url):
    soup = get_soup(thread_url)

    # Thread title
    title_tag = soup.find("span", id=lambda x: x and x.startswith("tmid"))
    title = title_tag.get_text(strip=True) if title_tag else "UNKNOWN"

    # Post bodies
    posts = []
    for td in soup.select("td.messagemiddle"):
        text = td.get_text(" ", strip=True)
        if text:
            posts.append(text)

    log.info(f"Thread '{title}' → {len(posts)} posts scraped")
    return title, posts

# ---------------- MAIN ---------------- #

def main():
    log.info("========== SCRAPER START ==========")

    all_rows = []

    forum_pages = get_forum_pages()

    thread_urls = set()
    for page_url in forum_pages:
        try:
            links = extract_thread_links(page_url)
            thread_urls.update(links)
        except Exception as e:
            log.error(f"Failed extracting threads from {page_url}: {e}")

    log.info(f"Total unique threads found: {len(thread_urls)}")

    for thread_url in tqdm(thread_urls, desc="Scraping threads"):
        try:
            title, posts = scrape_thread(thread_url)
            for post in posts:
                all_rows.append({
                    "thread_title": title,
                    "post_content": post,
                    "thread_url": thread_url
                })
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            log.error(f"Failed scraping thread {thread_url}: {e}")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    log.info(f"Saved CSV → {OUTPUT_CSV}")
    log.info(f"Total rows written: {len(df)}")
    log.info("========== SCRAPER END ==========")

# ---------------- RUN ---------------- #

if __name__ == "__main__":
    main()
