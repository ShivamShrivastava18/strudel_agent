import requests
from bs4 import BeautifulSoup

# URL of the Strudel documentation
url = "https://strudel.cc/workshop/getting-started/"

# Send a GET request to fetch the page content
response = requests.get(url)
response.raise_for_status()  # Raises an error if the request fails

# Parse HTML content using BeautifulSoup
soup = BeautifulSoup(response.text, "html.parser")

# Optional: narrow down to only the main content (based on HTML inspection)
main_content = soup.find("main")
if not main_content:
    main_content = soup.body  # fallback

# Extract and clean all text
text = main_content.get_text(separator="\n", strip=True)

# Save to a local file
with open("Data/strudel_getting_started.txt", "w", encoding="utf-8") as file:
    file.write(text)

print("✅ Documentation scraped and saved to 'strudel_getting_started.txt'")
