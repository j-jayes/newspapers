import os
import re
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import urlparse, unquote


class KBNewspaperScraper:
    def __init__(self, download_dir="newspaper_downloads"):
        """Initialize the scraper with optional download directory."""
        self.download_dir = download_dir
        os.makedirs(download_dir, exist_ok=True)
        
        # Setup Selenium with Chrome
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run headless if you don't need to see the browser
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--no-sandbox")
        
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )
        self.wait = WebDriverWait(self.driver, 20)

    def extract_manifest_id_from_html(self, html_content):
        """Extract the manifest ID from the HTML content."""
        # Look for the pattern in data-src or src attributes
        pattern = r'data-src="https://data\.kb\.se/iiif/\d+/([^/%]+)'
        match = re.search(pattern, html_content)
        
        if match:
            return match.group(1)  # Extract only the ID part before any % character
        
        # Try alternative pattern if the first one didn't match
        pattern = r'src="https://data\.kb\.se/iiif/\d+/([^/%]+)'
        match = re.search(pattern, html_content)
        
        if match:
            return match.group(1)
        
        return None
        
    def extract_date_from_html(self, html_content):
        """Extract the newspaper date from HTML content or page title."""
        # Try to find date in the search result item date field
        date_pattern = r'<p class="search-result-item-date[^>]*>([^<]+)</p>'
        match = re.search(date_pattern, html_content)
        if match:
            return match.group(1).strip()
            
        # Try to extract from title tag if available
        title_pattern = r'<title>([^|]+)\s+(\d{4}-\d{2}-\d{2})\s*[|]'
        match = re.search(title_pattern, html_content)
        if match:
            return match.group(2).strip()
            
        # Try to extract from filename in image source
        filename_pattern = r'bib\d+_(\d{4})(\d{2})(\d{2})_'
        match = re.search(filename_pattern, html_content)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
            
        return None
        
    def extract_filenames_from_html(self, html_content):
        """Extract potential JP2 filenames from HTML content."""
        # Extract filenames from image URLs
        filename_pattern = r'(bib\d+_\d+_\d+_\d+_\d+\.jp2)'
        matches = re.findall(filename_pattern, html_content)
        return list(set(matches))  # Return unique filenames

    def extract_jp2_from_manifest_data(self, manifest_url):
        """Extract JP2 file URLs directly from the manifest data."""
        try:
            print(f"Fetching manifest data from: {manifest_url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/json, text/plain, */*',
                'Referer': 'https://tidningar.kb.se/',
            }
            
            response = requests.get(f"{manifest_url}/manifest", headers=headers)
            response.raise_for_status()
            
            manifest_data = response.json()
            jp2_urls = []
            filenames = []
            
            # Extract JP2 URLs and filenames from the manifest items
            if 'items' in manifest_data:
                for canvas in manifest_data['items']:
                    if 'items' in canvas:
                        for annotation_page in canvas['items']:
                            if 'items' in annotation_page:
                                for annotation in annotation_page['items']:
                                    if 'body' in annotation and 'id' in annotation['body']:
                                        body_id = annotation['body']['id']
                                        if body_id.endswith('.jp2'):
                                            jp2_urls.append(body_id)
                                            # Extract filename from the URL
                                            filename = body_id.split('/')[-1]
                                            filenames.append(filename)
            
            # If we extracted filenames, print them for debugging
            if filenames:
                print(f"Extracted {len(filenames)} filenames from manifest:")
                for filename in filenames[:5]:  # Print first few
                    print(f" - {filename}")
            
            return jp2_urls
        
        except Exception as e:
            print(f"Error extracting JP2 files from manifest: {e}")
            return []
            
    def extract_title_and_date_from_page_head(self, page_source):
        """Extract title and date from the HTML head section (works on manifest page)."""
        try:
            # Extract from title tag
            title_pattern = r'<title>([^|]+?)(?:\s+(\d{4}-\d{2}-\d{2}))?\s*[|]'
            match = re.search(title_pattern, page_source)
            
            if match:
                title = match.group(1).strip()
                date = match.group(2).strip() if match.group(2) else None
                
                # If date wasn't in the title tag directly, try meta tags
                if not date:
                    meta_pattern = r'<meta[^>]*og:title[^>]*content="[^"]*?\s+(\d{4}-\d{2}-\d{2})"'
                    meta_match = re.search(meta_pattern, page_source)
                    if meta_match:
                        date = meta_match.group(1)
                
                return title, date
            
            return None, None
        except Exception as e:
            print(f"Error extracting title and date from head: {e}")
            return None, None
    
    def scrape_by_date_range(self, start_date, end_date, paper_id=None):
        """
        Scrape newspapers within a date range.
        
        Args:
            start_date: String in format 'YYYY-MM-DD'
            end_date: String in format 'YYYY-MM-DD'
            paper_id: Optional paper ID to filter by
        """
        # Construct the URL with date filters
        base_url = "https://tidningar.kb.se/search?q=%2a"
        url = f"{base_url}&from={start_date}&to={end_date}"
        
        # Add paper filter if provided
        if paper_id:
            url += f"&isPartOf.%40id={paper_id}"
        else:
            # Use the Dagens Nyheter paper ID from your example
            url += "&isPartOf.%40id=https%3A%2F%2Flibris.kb.se%2Fm5z2w4lz3m2zxpk%23it"
            
        print(f"Using search URL: {url}")
        
        print(f"Navigating to search page: {url}")
        self.driver.get(url)
        time.sleep(3)  # Allow page to load

        # Find and process each newspaper result
        results = self.wait.until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.search-result-item"))
        )
        
        print(f"Found {len(results)} newspaper issues")
        
        # Process each result
        for i, result in enumerate(results):
            try:
                # Extract date and title before clicking
                newspaper_title = None
                newspaper_date = None
                
                # First try to find the date at the bottom of the result
                date_elements = result.find_elements(By.CSS_SELECTOR, "p.search-result-item-date")
                if date_elements:
                    newspaper_date = date_elements[0].text.strip()
                
                # Try to find the title
                title_elements = result.find_elements(By.CSS_SELECTOR, "div.search-result-item-title")
                if title_elements:
                    newspaper_title = title_elements[0].text.strip()
                
                print(f"Processing issue {i+1}/{len(results)}: {newspaper_title} - {newspaper_date}")
                
                # Get the inner HTML to extract the manifest ID directly
                inner_html = result.get_attribute('innerHTML')
                print("Extracting manifest ID from inner HTML")
                
                manifest_id = self.extract_manifest_id_from_html(inner_html)
                
                # Extract or verify date from HTML if not already found
                if not newspaper_date or newspaper_date == "Unknown_Date":
                    extracted_date = self.extract_date_from_html(inner_html)
                    if extracted_date:
                        newspaper_date = extracted_date
                        print(f"Extracted date from HTML: {newspaper_date}")
                
                # Try to extract potential JP2 filenames directly from the HTML
                potential_filenames = self.extract_filenames_from_html(inner_html)
                if potential_filenames:
                    print(f"Found potential JP2 filenames in HTML: {len(potential_filenames)}")
                    for filename in potential_filenames[:3]:  # Print first few for debugging
                        print(f" - {filename}")
                
                if manifest_id:
                    print(f"Found manifest ID from HTML: {manifest_id}")
                    manifest_url = f"https://data.kb.se/{manifest_id}"
                    
                    # Clean newspaper_title for folder name
                    if newspaper_title:
                        newspaper_title = re.sub(r'[^\w\s-]', '', newspaper_title).strip()
                    else:
                        newspaper_title = "Unknown"
                        
                    if newspaper_date:
                        # Convert date format if needed
                        newspaper_date = newspaper_date.replace('/', '-')
                    else:
                        newspaper_date = "Unknown_Date"
                    
                    # Create folder for this newspaper and date
                    folder_path = os.path.join(self.download_dir, newspaper_title, newspaper_date)
                    os.makedirs(folder_path, exist_ok=True)
                    
                    # Get JP2 files from the manifest
                    jp2_urls = self.extract_jp2_from_manifest_data(manifest_url)
                    print(f"Found {len(jp2_urls)} JP2 files from manifest data")
                    
                    # Download each JP2 file
                    for file_url in jp2_urls:
                        # Ensure the URL is properly formatted
                        if '\\' in file_url:
                            file_url = file_url.replace('\\', '')
                            
                        filename = os.path.basename(unquote(file_url))
                        file_path = os.path.join(folder_path, filename)
                        
                        # Download the file
                        self.download_file(file_url, file_path)
                else:
                    print("Failed to extract manifest ID from HTML, trying alternative method")
                    
                    # Click on the result to open the newspaper
                    result.click()
                    time.sleep(3)  # Wait for newspaper page to load
                    
                    # Get current URL for organization purposes
                    current_url = self.driver.current_url
                    
                    # Extract the manifest information from the URL or page source
                    try:
                        print("Extracting manifest data from page source")
                        
                        # Get the page source
                        page_source = self.driver.page_source
                        
                        # Look for the manifest ID in the NUXT data
                        manifest_pattern = r'"id":"(https:\\\/\\\/data\.kb\.se\\\/([^\/\\]+)\\\/manifest)"'
                        match = re.search(manifest_pattern, page_source)
                        
                        if match:
                            manifest_id = match.group(2)  # This gets the ID portion
                            print(f"Found manifest ID: {manifest_id}")
                            manifest_url = f"https://data.kb.se/{manifest_id}"
                        else:
                            # Alternative extraction method
                            parsed_url = urlparse(current_url)
                            path_parts = parsed_url.path.split('/')
                            if len(path_parts) > 1:
                                manifest_id = path_parts[1]
                                print(f"Extracted manifest ID from URL: {manifest_id}")
                                manifest_url = f"https://data.kb.se/{manifest_id}"
                            else:
                                raise Exception("Could not extract manifest ID")
                        
                        # Navigate to the manifest page
                        self.driver.get(manifest_url)
                        time.sleep(2)
                        
                        # Extract newspaper info for folder organization if not already obtained
                        if not newspaper_date or not newspaper_title:
                            # Try to extract from page head first
                            page_source = self.driver.page_source
                            extracted_title, extracted_date = self.extract_title_and_date_from_page_head(page_source)
                            
                            if extracted_title:
                                newspaper_title = extracted_title
                                print(f"Extracted title from page head: {newspaper_title}")
                            
                            if extracted_date:
                                newspaper_date = extracted_date
                                print(f"Extracted date from page head: {newspaper_date}")
                            
                            # If still not found, use page title as fallback
                            if not newspaper_date or not newspaper_title:
                                page_title = self.driver.title
                                title_match = re.search(r'([^|]+)', page_title)
                                if title_match:
                                    combined_title = title_match.group(1).strip()
                                    # Try to split title and date
                                    parts = combined_title.split()
                                    if len(parts) >= 2:
                                        if not newspaper_title:
                                            newspaper_title = ' '.join(parts[:-1])
                                        if not newspaper_date and re.match(r'\d{4}-\d{2}-\d{2}', parts[-1]):
                                            newspaper_date = parts[-1]
                        
                        # Clean newspaper_title for folder name
                        if newspaper_title:
                            newspaper_title = re.sub(r'[^\w\s-]', '', newspaper_title).strip()
                        else:
                            newspaper_title = "Unknown"
                            
                        if newspaper_date:
                            # Convert date format if needed
                            newspaper_date = newspaper_date.replace('/', '-')
                        else:
                            newspaper_date = "Unknown_Date"
                        
                        # Create folder for this newspaper and date
                        folder_path = os.path.join(self.download_dir, newspaper_title, newspaper_date)
                        os.makedirs(folder_path, exist_ok=True)
                        
                        # Use multiple methods to extract JP2 file URLs
                        jp2_urls = []
                        
                        # Method 1: Extract from manifest data
                        jp2_urls = self.extract_jp2_from_manifest_data(manifest_url)
                        print(f"Method 1: Found {len(jp2_urls)} JP2 files from manifest data")
                        
                        # Method 2: Extract from page source JSON data if method 1 failed
                        if not jp2_urls:
                            page_source = self.driver.page_source
                            jp2_pattern = r'"id":"(https:\\\/\\\/data\.kb\.se\\\/[^"]+\.jp2)"'
                            jp2_matches = re.findall(jp2_pattern, page_source)
                            jp2_urls = [url.replace('\/', '/') for url in jp2_matches]
                            print(f"Method 2: Found {len(jp2_urls)} JP2 files in page source")
                        
                        # Method 3: Look for links on the page if methods 1 and 2 failed
                        if not jp2_urls:
                            jp2_links = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='.jp2']")
                            if jp2_links:
                                jp2_urls = [link.get_attribute('href') for link in jp2_links]
                                print(f"Method 3: Found {len(jp2_urls)} JP2 files from page links")
                        
                        # Download each JP2 file
                        for file_url in jp2_urls:
                            # Ensure the URL is properly formatted
                            if '\\' in file_url:
                                file_url = file_url.replace('\\', '')
                                
                            filename = os.path.basename(unquote(file_url))
                            file_path = os.path.join(folder_path, filename)
                            
                            # Download the file
                            self.download_file(file_url, file_path)
                            
                    except Exception as e:
                        print(f"Error processing manifest: {e}")
                
                    # Go back to search results
                    self.driver.back()
                    time.sleep(2)
                    
                    # If we went back too far, go forward once
                    if "search" not in self.driver.current_url:
                        self.driver.get(url)
                        time.sleep(3)
                
            except Exception as e:
                print(f"Error processing newspaper issue: {e}")
                # Return to search page
                self.driver.get(url)
                time.sleep(3)
                
            # Refresh the results list
            results = self.wait.until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.search-result-item"))
            )

    def download_file(self, url, filepath):
        """Download a file from URL to the specified filepath."""
        try:
            print(f"Downloading {url} to {filepath}")
            
            # Skip if file already exists
            if os.path.exists(filepath):
                print(f"File already exists: {filepath}")
                return True
            
            # Make request with appropriate headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Referer': 'https://tidningar.kb.se/',
                'Accept': 'image/jpeg, image/png, image/jp2, */*'
            }
            
            # Sometimes the URL might be malformed with double backslashes
            clean_url = url.replace('\\\\', '/')
            
            # Retry mechanism for robustness
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.get(clean_url, headers=headers, stream=True, timeout=30)
                    response.raise_for_status()
                    
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    print(f"Downloaded: {filepath}")
                    return True
                    
                except (requests.exceptions.RequestException, requests.exceptions.Timeout) as req_err:
                    if attempt < max_retries - 1:
                        print(f"Retry {attempt+1}/{max_retries} downloading {clean_url}: {req_err}")
                        time.sleep(2)  # Wait before retrying
                    else:
                        raise
            
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False

    def close(self):
        """Close the browser and clean up."""
        self.driver.quit()


# Example usage
if __name__ == "__main__":
    scraper = KBNewspaperScraper(download_dir="kb_newspapers")
    
    try:
        # Scrape newspapers from January 1-2, 1865
        scraper.scrape_by_date_range("1865-01-04", "1865-01-05")
    finally:
        scraper.close()