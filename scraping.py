import requests
from bs4 import BeautifulSoup

url = "https://strudel.cc/workshop/getting-started/"

response = requests.get(url)
response.raise_for_status()  

soup = BeautifulSoup(response.text, "html.parser")

main_content = soup.find("main")
if not main_content:
    main_content = soup.body  

text = main_content.get_text(separator="\n", strip=True)

with open("Data/strudel_getting_started.txt", "w", encoding="utf-8") as file:
    file.write(text)

print("✅ Documentation scraped and saved to 'strudel_getting_started.txt'")
