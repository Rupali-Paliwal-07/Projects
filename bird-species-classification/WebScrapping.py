import requests
from bs4 import BeautifulSoup
import csv

# Step 1: Define the URL of the website with .wav audio files
url = "https://indianbirdsong.org/explore/all-recordings"

# Step 2: Send an HTTP request and parse the HTML content
try:
    response = requests.get(url)
    response.raise_for_status()  # Check for HTTP request errors
    soup = BeautifulSoup(response.content, "html.parser")
except requests.exceptions.RequestException as e:
    print(f"Error: Unable to retrieve data from the website - {e}")
    exit()

# Step 3: Find .wav audio links and metadata
audio_links = []
metadata = []

for link in soup.find_all("a"):
    href = link.get("href")
    if href and href.endswith(".wav"):
        audio_links.append(href)
        # Extract metadata
        parent_div = link.find_parent("div", class_="row")
        if parent_div:
            metadata_info = parent_div.text
            metadata.append(metadata_info)

# Step 4: Define the number of records to scrape
num_records_to_scrape = 10  # Modify as needed

# Step 5: Download .wav audio files and create a CSV dataset for the subset
try:
    with open("wav_dataset.csv", mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Species", "Recordist", "Location", "Uploaded Date", "Audio_Path"])

        for i in range(min(num_records_to_scrape, len(audio_links))):
            species, recordist, location, uploaded_date = metadata[i].split("\n")
            species = species.strip()
            recordist = recordist.split(":")[1].strip()
            location = location.split(":")[1].strip()
            uploaded_date = uploaded_date.split(":")[1].strip()
            audio_path = f"audio_{i}.wav"  # Save as .wav
            csv_writer.writerow([species, recordist, location, uploaded_date, audio_path])

            # Download .wav audio file
            try:
                audio_response = requests.get(audio_links[i])
                audio_response.raise_for_status()  # Check for download errors
                with open(f"audio_{i}.wav", "wb") as audio_file:
                    audio_file.write(audio_response.content)
            except requests.exceptions.RequestException as e:
                print(f"Error downloading audio {i}: {e}")
except IOError as e:
    print(f"Error writing to CSV file: {e}")

print(f"Dataset creation complete for the first {num_records_to_scrape} .wav records.")
