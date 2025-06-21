import requests
from bs4 import BeautifulSoup
import os

with open("scraping site.txt", "r") as f:
    urls = [line.strip() for line in f if line.strip()]

for idx, url in enumerate(urls, 1):
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        main_content = soup.find("main") or soup.body  # fallback
        text = main_content.get_text(separator="\n", strip=True)

        filename = f"Data/page_{idx:02d}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"URL: {url}\n\n{text}")

        print(f"✅ Saved: {filename}")

    except Exception as e:
        print(f"❌ Error scraping {url}: {e}")
