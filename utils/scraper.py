import requests
from bs4 import BeautifulSoup
import csv
import time

# Base URLs
base_forum_url = "https://talk.newagtalk.com/forums/forum-view.asp?fid=12"
base_thread_url = "https://talk.newagtalk.com/forums/"

# CSV output
output_file = "agtalk_stock_talk_posts.csv"

# Headers (optional, mimic a real browser)
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

# Function to get thread URLs from a single forum page
def get_thread_urls(html):
    soup = BeautifulSoup(html, "html.parser")
    threads = []
    for a in soup.select("a[href*='thread-view.asp?']"):
        href = a.get("href")
        if "tid=" in href:
            full_url = requests.compat.urljoin(base_thread_url, href)
            threads.append(full_url)
    return list(set(threads))  # Unique threads

# Function to scrape the first post from a thread
def scrape_thread(thread_url):
    res = requests.get(thread_url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")

    # Title of the thread
    title_tag = soup.find("title")
    title = title_tag.text.strip() if title_tag else ""

    # First post block (original post)
    post = soup.find("table", class_="forumBoard")  # main thread table
    if not post:
        return None

    post_content = ""
    author = ""
    post_date = ""

    # Get author & date if available
    author_tag = post.find("td", class_="forumName")
    date_tag = post.find("td", class_="forumDate")

    if author_tag:
        author = author_tag.text.strip()
    if date_tag:
        post_date = date_tag.text.strip()

    # Get the content
    content_div = post.find("td", class_="forumMessage")
    if content_div:
        post_content = content_div.get_text("\n", strip=True)

    return {
        "thread_url": thread_url,
        "title": title,
        "author": author,
        "post_date": post_date,
        "content": post_content
    }

# Main
if __name__ == "__main__":

    all_threads = set()
    page = 1

    # Grab multiple pages until no more threads found or a reasonable limit
    while True:
        print(f"Scraping forum page: {page}")
        params = {"Page": page}
        r = requests.get(base_forum_url, params=params, headers=headers)

        if r.status_code != 200:
            break

        threads = get_thread_urls(r.text)
        if not threads:
            break

        # Add to master set
        all_threads.update(threads)
        page += 1

        # stop if too many pages
        if page > 10:
            break

        time.sleep(1)

    print(f"Found {len(all_threads)} threads.")

    # CSV writing
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["thread_url", "title", "author", "post_date", "content"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, thread_url in enumerate(all_threads):
            print(f"Scraping thread [{i+1}/{len(all_threads)}]: {thread_url}")
            post_data = scrape_thread(thread_url)
            if post_data:
                writer.writerow(post_data)
            time.sleep(1)

    print("Done! Data saved to", output_file)
