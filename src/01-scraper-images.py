#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KB Newspaper Scraper

A production-ready scraper for the KB newspaper archive.
Fetches newspaper images from the KB digital archive based on date ranges.
"""

import os
import re
import time
import logging
import argparse
import requests
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from selenium.webdriver.remote.webelement import WebElement
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import urlparse, unquote
from pathlib import Path


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("kb_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("KBNewspaperScraper")


@dataclass
class NewspaperIssue:
    """Data class to store information about a newspaper issue."""
    title: str
    date: str
    manifest_id: str
    filenames: List[str] = None
    
    def __post_init__(self):
        """Initialize optional fields with empty lists."""
        if self.filenames is None:
            self.filenames = []


class KBNewspaperScraper:
    """
    A scraper for the KB (Kungliga biblioteket) digital newspaper archive.
    
    Fetches newspaper pages as JP2 files based on date ranges and optional filters.
    Uses Selenium for web navigation and direct HTTP requests for file downloads.
    """
    
    BASE_URL = "https://tidningar.kb.se"
    SEARCH_URL = f"{BASE_URL}/search"
    API_BASE_URL = "https://data.kb.se"
    
    def __init__(self, download_dir: str = "newspaper_downloads", headless: bool = True, 
                 retry_count: int = 3, wait_time: int = 20):
        """
        Initialize the scraper with configurable options.
        
        Args:
            download_dir: Directory where newspaper files will be downloaded
            headless: Whether to run the browser in headless mode
            retry_count: Number of times to retry failed operations
            wait_time: Maximum wait time in seconds for page elements to load
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.retry_count = retry_count
        
        # Setup Selenium with Chrome
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        # Set user agent to mimic a real browser
        chrome_options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        
        logger.info("Initializing Chrome WebDriver")
        try:
            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=chrome_options
            )
            self.wait = WebDriverWait(self.driver, wait_time)
            logger.info("WebDriver initialized successfully")
        except WebDriverException as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            raise
    
    def extract_manifest_id_from_html(self, html_content: str) -> Optional[str]:
        """
        Extract the manifest ID from the HTML content.
        
        Args:
            html_content: HTML content to extract manifest ID from
            
        Returns:
            The manifest ID if found, otherwise None
        """
        # Look for the pattern in data-src or src attributes
        pattern = r'data-src="https://data\.kb\.se/iiif/\d+/([^/%]+)'
        match = re.search(pattern, html_content)
        
        if match:
            return match.group(1)
        
        # Try alternative pattern if the first one didn't match
        pattern = r'src="https://data\.kb\.se/iiif/\d+/([^/%]+)'
        match = re.search(pattern, html_content)
        
        if match:
            return match.group(1)
        
        return None
    
    def extract_date_from_html(self, html_content: str) -> Optional[str]:
        """
        Extract the newspaper date from HTML content.
        
        Args:
            html_content: HTML content to extract date from
            
        Returns:
            Date string in format 'YYYY-MM-DD' if found, otherwise None
        """
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
    
    def extract_filenames_from_html(self, html_content: str) -> List[str]:
        """
        Extract potential JP2 filenames from HTML content.
        
        Args:
            html_content: HTML content to extract filenames from
            
        Returns:
            List of JP2 filenames found in the HTML
        """
        # Extract filenames from image URLs
        filename_pattern = r'(bib\d+_\d+_\d+_\d+_\d+\.jp2)'
        matches = re.findall(filename_pattern, html_content)
        return list(set(matches))  # Return unique filenames
    
    def extract_title_and_date_from_page_head(self, page_source: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract title and date from the HTML head section.
        
        Args:
            page_source: HTML source of the page
            
        Returns:
            Tuple of (title, date) if found, otherwise (None, None)
        """
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
            logger.error(f"Error extracting title and date from head: {e}")
            return None, None
    
    def extract_jp2_from_manifest_data(self, manifest_url: str) -> List[str]:
        """
        Extract JP2 file URLs directly from the manifest data.
        
        Args:
            manifest_url: Base URL of the manifest without /manifest suffix
            
        Returns:
            List of JP2 file URLs
        """
        try:
            logger.info(f"Fetching manifest data from: {manifest_url}")
            
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
            
            # If we extracted filenames, log them for debugging
            if filenames:
                logger.info(f"Extracted {len(filenames)} filenames from manifest")
                for filename in filenames[:5]:  # Log first few
                    logger.debug(f" - {filename}")
            
            return jp2_urls
        
        except Exception as e:
            logger.error(f"Error extracting JP2 files from manifest: {e}", exc_info=True)
            return []
    
    def download_file(self, url: str, filepath: Path) -> bool:
        """
        Download a file from URL to the specified filepath.
        
        Args:
            url: URL of the file to download
            filepath: Local path where the file will be saved
            
        Returns:
            True if download was successful, False otherwise
        """
        try:
            logger.info(f"Downloading {url} to {filepath}")
            
            # Skip if file already exists
            if filepath.exists():
                logger.info(f"File already exists: {filepath}")
                return True
            
            # Make request with appropriate headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Referer': 'https://tidningar.kb.se/',
                'Accept': 'image/jpeg, image/png, image/jp2, */*'
            }
            
            # Clean URL for any encoding issues
            clean_url = url.replace('\\\\', '/').replace('\\/', '/')
            
            # Retry mechanism for robustness
            for attempt in range(self.retry_count):
                try:
                    response = requests.get(clean_url, headers=headers, stream=True, timeout=30)
                    response.raise_for_status()
                    
                    # Create directory if it doesn't exist
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    logger.info(f"Downloaded: {filepath}")
                    return True
                    
                except (requests.exceptions.RequestException, requests.exceptions.Timeout) as req_err:
                    if attempt < self.retry_count - 1:
                        logger.warning(f"Retry {attempt+1}/{self.retry_count} downloading {clean_url}: {req_err}")
                        time.sleep(2)  # Wait before retrying
                    else:
                        raise
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}", exc_info=True)
            return False
    
    def process_search_result(self, result: WebElement) -> Optional[NewspaperIssue]:
        """
        Process a single search result element.
        
        Args:
            result: Selenium WebElement representing a search result
            
        Returns:
            NewspaperIssue object if successfully processed, None otherwise
        """
        try:
            # Extract date and title
            newspaper_title = None
            newspaper_date = None
            
            # Find date element
            date_elements = result.find_elements(By.CSS_SELECTOR, "p.search-result-item-date")
            if date_elements:
                newspaper_date = date_elements[0].text.strip()
            
            # Find title element
            title_elements = result.find_elements(By.CSS_SELECTOR, "div.search-result-item-title")
            if title_elements:
                newspaper_title = title_elements[0].text.strip()
            
            logger.info(f"Processing issue: {newspaper_title} - {newspaper_date}")
            
            # Get the inner HTML to extract the manifest ID directly
            inner_html = result.get_attribute('innerHTML')
            logger.debug("Extracting manifest ID from inner HTML")
            
            manifest_id = self.extract_manifest_id_from_html(inner_html)
            
            # Extract or verify date from HTML if not already found
            if not newspaper_date:
                extracted_date = self.extract_date_from_html(inner_html)
                if extracted_date:
                    newspaper_date = extracted_date
                    logger.info(f"Extracted date from HTML: {newspaper_date}")
            
            # Try to extract potential JP2 filenames directly from the HTML
            potential_filenames = self.extract_filenames_from_html(inner_html)
            if potential_filenames:
                logger.info(f"Found {len(potential_filenames)} potential JP2 filenames in HTML")
                for filename in potential_filenames[:3]:  # Log first few for debugging
                    logger.debug(f" - {filename}")
            
            if manifest_id:
                logger.info(f"Found manifest ID from HTML: {manifest_id}")
                
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
                
                return NewspaperIssue(
                    title=newspaper_title,
                    date=newspaper_date,
                    manifest_id=manifest_id,
                    filenames=potential_filenames
                )
            else:
                logger.warning("Failed to extract manifest ID from HTML")
                return None
                
        except Exception as e:
            logger.error(f"Error processing search result: {e}", exc_info=True)
            return None
    
    def download_newspaper_issue(self, issue: NewspaperIssue) -> bool:
        """
        Download all files for a newspaper issue.
        
        Args:
            issue: NewspaperIssue object containing metadata
            
        Returns:
            True if all downloads were successful, False otherwise
        """
        try:
            manifest_url = f"{self.API_BASE_URL}/{issue.manifest_id}"
            
            # Create folder for this newspaper and date
            folder_path = self.download_dir / issue.title / issue.date
            folder_path.mkdir(parents=True, exist_ok=True)
            
            # Get JP2 files from the manifest
            jp2_urls = self.extract_jp2_from_manifest_data(manifest_url)
            logger.info(f"Found {len(jp2_urls)} JP2 files from manifest data")
            
            if not jp2_urls:
                logger.warning(f"No JP2 files found for {issue.title} - {issue.date}")
                return False
            
            # Download each JP2 file
            success_count = 0
            for file_url in jp2_urls:
                filename = Path(unquote(file_url)).name
                file_path = folder_path / filename
                
                if self.download_file(file_url, file_path):
                    success_count += 1
            
            return success_count == len(jp2_urls)
            
        except Exception as e:
            logger.error(f"Error downloading newspaper issue: {e}", exc_info=True)
            return False
    
    def scrape_by_date_range(self, start_date: str, end_date: str, paper_id: str = None) -> List[NewspaperIssue]:
        """
        Scrape newspapers within a date range.
        
        Args:
            start_date: String in format 'YYYY-MM-DD'
            end_date: String in format 'YYYY-MM-DD'
            paper_id: Optional paper ID to filter by
            
        Returns:
            List of NewspaperIssue objects that were successfully scraped
        """
        # Validate date formats
        date_pattern = r'^\d{4}-\d{2}-\d{2}$'
        if not re.match(date_pattern, start_date) or not re.match(date_pattern, end_date):
            logger.error(f"Invalid date format. Must be YYYY-MM-DD. Got: {start_date} to {end_date}")
            return []
        
        # Construct the URL with date filters
        url = f"{self.SEARCH_URL}?q=%2a&from={start_date}&to={end_date}"
        
        # Add paper filter if provided
        if paper_id:
            url += f"&isPartOf.%40id={paper_id}"
        else:
            # Use the Dagens Nyheter paper ID by default
            url += "&isPartOf.%40id=https%3A%2F%2Flibris.kb.se%2Fm5z2w4lz3m2zxpk%23it"
            
        logger.info(f"Using search URL: {url}")
        
        logger.info(f"Navigating to search page: {url}")
        self.driver.get(url)
        time.sleep(3)  # Allow page to load

        # Find all search results
        try:
            results = self.wait.until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.search-result-item"))
            )
            
            logger.info(f"Found {len(results)} newspaper issues")
            
            processed_issues = []
            
            # Process each result
            for i, result in enumerate(results):
                try:
                    logger.info(f"Processing result {i+1}/{len(results)}")
                    
                    # Extract information from the search result
                    issue = self.process_search_result(result)
                    
                    if issue:
                        # Download the issue
                        success = self.download_newspaper_issue(issue)
                        if success:
                            processed_issues.append(issue)
                            logger.info(f"Successfully downloaded issue: {issue.title} - {issue.date}")
                        else:
                            logger.warning(f"Failed to download issue: {issue.title} - {issue.date}")
                    
                except Exception as e:
                    logger.error(f"Error processing result {i+1}: {e}", exc_info=True)
                    continue
            
            return processed_issues
                
        except TimeoutException:
            logger.error("Timeout waiting for search results")
            return []
        except Exception as e:
            logger.error(f"Error during scraping: {e}", exc_info=True)
            return []
    
    def close(self):
        """Close the browser and clean up resources."""
        logger.info("Closing WebDriver")
        try:
            self.driver.quit()
        except:
            pass


def main():
    """Main entry point for the scraper."""
    parser = argparse.ArgumentParser(description='KB Newspaper Scraper')
    parser.add_argument('--start-date', type=str, required=True, help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end-date', type=str, required=True, help='End date in YYYY-MM-DD format')
    parser.add_argument('--paper-id', type=str, help='Optional paper ID to filter by')
    parser.add_argument('--download-dir', type=str, default='kb_newspapers', help='Directory to save downloads')
    parser.add_argument('--headless', action='store_true', help='Run browser in headless mode')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', 
                        help='Logging level')
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    scraper = KBNewspaperScraper(download_dir=args.download_dir, headless=args.headless)
    
    try:
        issues = scraper.scrape_by_date_range(args.start_date, args.end_date, args.paper_id)
        logger.info(f"Completed scraping. Downloaded {len(issues)} issues successfully.")
    finally:
        scraper.close()


if __name__ == "__main__":
    main()