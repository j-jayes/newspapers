"""KB archive ingestion — download and convert JPEG2000 scans.

Provides utilities for:
- Querying the KB (tidningar.kb.se) API for newspaper metadata.
- Downloading JPEG2000 (.jp2) page scans.
- Converting .jp2 files to optimised .jpg / .png for downstream processing.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JPEG2000 → standard format conversion
# ---------------------------------------------------------------------------


def convert_jp2(
    input_path: Path,
    output_dir: Path,
    *,
    low_res_size: tuple[int, int] = (1280, 1280),
) -> tuple[Path, Path]:
    """Convert a JPEG2000 file to both a low-res JPEG and a high-res PNG.

    Parameters
    ----------
    input_path:
        Path to the source ``.jp2`` file.
    output_dir:
        Directory where the converted files will be written.
    low_res_size:
        Maximum (width, height) for the low-resolution JPEG thumbnail
        used by the segmentation model.

    Returns
    -------
    tuple[Path, Path]
        ``(jpg_path, png_path)`` — paths to the generated files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    img = Image.open(input_path)

    # High-resolution lossless PNG (for Vision LLM extraction)
    png_path = output_dir / f"{stem}.png"
    img.save(png_path, format="PNG")

    # Low-resolution JPEG (for YOLOv11 segmentation)
    jpg_path = output_dir / f"{stem}.jpg"
    thumbnail = img.copy()
    thumbnail.thumbnail(low_res_size, Image.LANCZOS)
    thumbnail.save(jpg_path, format="JPEG", quality=85)

    logger.info("Converted %s → %s, %s", input_path.name, jpg_path.name, png_path.name)
    return jpg_path, png_path

import os
import io
import google.auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
TARGET_FOLDER_ID = '1uwu7l_8Xm9W3F9x8kakamWzXsFoi_07A'  # Provided Google Drive Folder ID
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'raw')

def authenticate_gdrive():
    """Authenticates and returns a Drive API service object.
    Uses Application Default Credentials.
    """
    try:
        credentials, project_id = google.auth.default(scopes=SCOPES)
        service = build('drive', 'v3', credentials=credentials)
        return service
    except Exception as e:
        print("Error during authentication. Please make sure you have run:")
        print("  gcloud auth application-default login --scopes=https://www.googleapis.com/auth/drive.readonly")
        raise e

def list_and_download_samples(max_downloads=5):
    """Maps the directory structure and downloads a few sample .jp2 files."""
    service = authenticate_gdrive()
    os.makedirs(DATA_DIR, exist_ok=True)
    
    downloaded = [0] # Use a list to mutate inside recursive function

    def crawl_folder(folder_id, folder_name="Root"):
        if downloaded[0] >= max_downloads:
            return
            
        print(f"Crawling folder: {folder_name}")
        query = f"'{folder_id}' in parents and trashed=false"
        # Support pagination if there are many items
        page_token = None
        while True:
            results = service.files().list(
                q=query, 
                pageToken=page_token,
                fields="nextPageToken, files(id, name, mimeType)"
            ).execute()
            
            items = results.get('files', [])
            for item in items:
                if downloaded[0] >= max_downloads:
                    break
                    
                if item['mimeType'] == 'application/vnd.google-apps.folder':
                    crawl_folder(item['id'], item['name'])
                elif item['name'].endswith('.jp2'):
                    download_file(service, item['id'], item['name'])
                    downloaded[0] += 1
            
            page_token = results.get('nextPageToken', None)
            if not page_token or downloaded[0] >= max_downloads:
                break

    crawl_folder(TARGET_FOLDER_ID)
    print(f"\nDownloaded {downloaded[0]} sample image(s) to {os.path.abspath(DATA_DIR)}")

def download_file(service, file_id, file_name):
    """Downloads a single file from Google Drive."""
    print(f"   -> Downloading {file_name}...")
    request = service.files().get_media(fileId=file_id)
    file_path = os.path.join(DATA_DIR, file_name)
    
    with io.FileIO(file_path, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"      Download {int(status.progress() * 100)}%.")

if __name__ == '__main__':
    list_and_download_samples()
