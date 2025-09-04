import os
import re
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# ===== SETUP CHROME DRIVER =====
options = Options()
options.add_argument("--start-maximized")  # For loading full content
driver = webdriver.Chrome(options=options)

# ===== PATH TO FILE WITH LINKS (comma-separated quoted links) =====
input_file_path = os.path.join(os.path.expanduser("~"), "Downloads", "Formatted_Links_with_Quotes.txt")

# ===== READ LINKS FROM FILE =====
with open(input_file_path, "r") as f:
    content = f.read()
    links = [link.strip().strip('"') for link in content.split(",") if link.strip()]

# ===== EXCEL FILE PATH =====
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
excel_path = os.path.join(desktop, "Dining_Tables_Scraped.xlsx")

# ===== LOAD EXISTING DATA =====
if os.path.exists(excel_path):
    existing_df = pd.read_excel(excel_path)
    existing_links = set(
        existing_df["Product Link"].dropna().tolist()) if "Product Link" in existing_df.columns else set()
else:
    existing_df = pd.DataFrame()
    existing_links = set()


# ===== DIMENSION PARSER =====
def extract_dimensions(text):
    patterns = [
        r'([\d.]+)\s*"\s*[Ww][i]?[d]?[t]?[h]?\s*[xX×*]\s*([\d.]+)\s*"\s*[Hh][e]?[i]?[g]?[h]?[t]?\s*[xX×*]\s*([\d.]+)\s*"\s*[Dd][e]?[p]?[t]?[h]?',
        r'([\d.]+)\s*"\s*[xX×*]\s*([\d.]+)\s*"\s*[xX×*]\s*([\d.]+)\s*"'
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            dims = sorted([float(match.group(i)) for i in range(1, 4)])
            return str(dims[2]), str(dims[1]), str(dims[0])  # W, H, D
    return "Not Found", "Not Found", "Not Found"


# ===== SCRAPE FUNCTION =====
def scrape_amazon_page(url):
    try:
        driver.get(url)
        time.sleep(4)  # Allow page to load

        body = driver.find_element(By.TAG_NAME, "body").text

        width, height, depth = extract_dimensions(body)
        weight = re.search(r'Item Weight\s+([\d.]+)\s+Pounds', body)
        color = re.search(r'Color\s+([A-Za-z \-/]+)', body)
        material = re.search(r'Top Material Type\s+([A-Za-z \-/]+)', body)
        shape = re.search(r'Shape\s+([A-Za-z \-/]+)', body)

        image_link = "Not Found"
        try:
            image = driver.find_element(By.CSS_SELECTOR, "#imgTagWrapperId img")
            image_link = image.get_attribute("src")
        except:
            pass

        return {
            "Name": "Dining Table",
            "Width (in)": width,
            "Height (in)": height,
            "Depth (in)": depth,
            "Weight (lbs)": weight.group(1) if weight else "Not Found",
            "Color": color.group(1).strip() if color else "Not Found",
            "Material": material.group(1).strip() if material else "Not Found",
            "Shape": shape.group(1).strip() if shape else "Not Found",
            "Image Link": image_link,
            "Product Link": url
        }

    except Exception as e:
        print(f"[ERROR] Failed for {url}: {e}")
        return None


# ===== PROCESS IN BATCHES OF 20 =====
batch_size = 20
new_data = []

for i in range(0, len(links), batch_size):
    batch = links[i:i + batch_size]
    print(f"\n🔍 Scraping Batch {i // batch_size + 1}: {len(batch)} links...")

    for url in batch:
        if url in existing_links:
            print(f"[SKIPPED] Already scraped: {url}")
            continue
        print(f"[SCRAPING] {url}")
        result = scrape_amazon_page(url)
        if result:
            new_data.append(result)
        time.sleep(2)  # polite delay

    print("⏳ Sleeping before next batch...")
    time.sleep(5)  # Wait between batches

# ===== SAVE DATA =====
if new_data:
    new_df = pd.DataFrame(new_data)
    final_df = pd.concat([existing_df, new_df], ignore_index=True)
    final_df.to_excel(excel_path, index=False)
    print(f"\n✅ DONE! Saved to: {excel_path}")
else:
    print("\n📁 No new data scraped.")

driver.quit()
