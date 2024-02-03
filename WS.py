import requests
from bs4 import BeautifulSoup

url = "https://indianbirdsong.org/"
response = requests.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.content, "html.parser")
    image_tags = soup.find_all("img")

    image_urls = []
    for i, img in enumerate(image_tags[:2]):
        src = img.get("src")
        if src and src.startswith("https"):
            image_urls.append(src)
            print(f"Image {i + 1} URL: {src}")

    if not image_urls:
        print("No image URLs found on the page.")
else:
    print(f"Failed to fetch the page. Status code: {response.status_code}")
