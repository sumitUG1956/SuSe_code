#!/usr/bin/env python3

"""
Download and Extract Module - Fetches Upstox instrument list
Downloads compressed instrument data from Upstox API and extracts it locally
"""

import gzip  # For decompressing .gz files
import shutil  # For file operations
from pathlib import Path  # For filesystem path handling
from urllib.request import Request, urlopen  # For HTTP requests

# Upstox API URL for complete instrument list (compressed JSON)
UPSTOX_URL = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"
# Local directory for storing downloaded files
DATA_DIR = Path("data")


def download_file(url, destination):
    """
    Download a file from URL and save to local destination
    
    Args:
        url: Source URL to download from
        destination: Path object where file should be saved
    
    Process:
        1. Create parent directories if they don't exist
        2. Create HTTP request with User-Agent header
        3. Stream download to local file (memory efficient)
    
    Note: Uses streaming copy to handle large files efficiently
    """
    destination.parent.mkdir(parents=True, exist_ok=True)  # Create directories if needed
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})  # HTTP request with headers
    # Open remote URL and local file, then copy data in chunks
    with urlopen(request) as response, destination.open("wb") as local_file:
        shutil.copyfileobj(response, local_file)  # Stream copy (chunk by chunk)


def extract_gzip(source, destination):
    """
    Extract a .gz compressed file to uncompressed destination
    
    Args:
        source: Path to .gz file
        destination: Path where extracted file should be written
    
    Process:
        Open compressed file and destination file, then stream decompress
    """
    # Open .gz file for reading and destination for writing
    with gzip.open(source, "rb") as gz_file, destination.open("wb") as out_file:
        shutil.copyfileobj(gz_file, out_file)  # Stream decompress (chunk by chunk)


def clean_previous_downloads(data_dir=DATA_DIR):
    """
    Remove previously downloaded instrument files for fresh download
    
    Args:
        data_dir: Directory containing downloaded files (default: DATA_DIR)
    
    Process:
        Delete both .gz compressed file and extracted .json file if they exist
    
    Why needed: Ensures fresh data download without stale files
    """
    gz_path = data_dir / "complete.json.gz"  # Path to compressed file
    json_path = data_dir / "complete.json"  # Path to extracted JSON
    for path in (gz_path, json_path):  # Iterate over both paths
        if path.exists():  # If file exists
            path.unlink()  # Delete the file


def download_and_extract(data_dir=DATA_DIR):
    """
    Complete workflow: download compressed file and extract it
    
    Args:
        data_dir: Directory for storing files (default: DATA_DIR)
    
    Returns:
        tuple: (download_path, extracted_path)
            - download_path: Path to downloaded .gz file
            - extracted_path: Path to extracted .json file
    
    Process:
        1. Define paths for downloaded and extracted files
        2. Download compressed file from Upstox
        3. Extract to plain JSON
        4. Return both paths
    
    Usage:
        gz_file, json_file = download_and_extract()
    """
    download_path = data_dir / "complete.json.gz"  # Where to save downloaded file
    extracted_path = download_path.with_suffix("")  # Remove .gz to get .json path

    download_file(UPSTOX_URL, download_path)  # Download from Upstox
    extract_gzip(download_path, extracted_path)  # Extract to JSON
    return download_path, extracted_path  # Return both file paths


def main():
    """
    CLI entry point - runs when script is executed directly
    
    Process:
        1. Clean any previous downloads
        2. Download and extract fresh instrument data
        3. Print paths to console for confirmation
    
    Usage:
        python download_extract.py
    """
    clean_previous_downloads()  # Remove old files
    download_path, extracted_path = download_and_extract()  # Download and extract

    # Print confirmation messages with absolute paths
    print(f"Downloaded to: {download_path.resolve()}")
    print(f"Extracted to: {extracted_path.resolve()}")


# Run main() if script is executed directly (not imported)
if __name__ == "__main__":
    main()
