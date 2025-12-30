#!/usr/bin/env python3
# Shebang line - Python 3 से script run करने के लिए

"""
DOWNLOAD & EXTRACT MODULE (डाउनलोड और एक्सट्रैक्ट मॉड्यूल)
========================================================

Purpose: Upstox से complete instrument list को download और extract करना
यह module daily trading instruments की master list को Upstox API से fetch करता है

Process:
1. Upstox servers से compressed .json.gz file download करो
2. File को extract करके plain JSON बनाओ
3. data/ directory में store करो

Data Contains: सभी NSE/BSE instruments की information (stocks, indices, options, futures)
- Instrument keys, trading symbols, expiry dates, strike prices, etc.
"""

import gzip  # .gz compressed files को handle करने के लिए
import shutil  # File operations (copy, move) के लिए
from pathlib import Path  # File paths को handle करने के लिए
from urllib.request import Request, urlopen  # HTTP requests के लिए

# CONFIGURATION CONSTANTS

# Upstox API URL जहां से instrument list download होती है (compressed format में)
UPSTOX_URL = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"

# Local directory जहां downloaded files store होंगी
DATA_DIR = Path("data")  # "./data" folder


def download_file(url, destination):
    """
    URL से file download करके local destination पर save करो (streaming copy के साथ)
    
    Purpose: Large files को efficiently download करना without loading entire file into memory
    
    Args:
        url: Download URL (string) - जहां से file download करनी है
        destination: Local file path (Path object) - जहां file save करनी है
    
    Process:
        1. Destination के parent directories create करो (अगर exist नहीं करती)
        2. URL को HTTP request में wrap करो (proper headers के साथ)
        3. Remote file को chunks में read करके local file में write करो
    
    Example:
        download_file("https://example.com/data.gz", Path("data/data.gz"))
    
    Note: shutil.copyfileobj() streaming copy करता है - memory efficient!
    """
    destination.parent.mkdir(parents=True, exist_ok=True)  # Parent directories create करो (if not exist), exist_ok=True means error नहीं अगर already exist
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})  # HTTP request बनाओ User-Agent header के साथ (कुछ servers इसकी requirement होती है)
    with urlopen(request) as response, destination.open("wb") as local_file:  # Remote URL open करो और local file open करो write binary mode में
        shutil.copyfileobj(response, local_file)  # Remote से local में data copy करो (streaming - chunk by chunk)


def extract_gzip(source, destination):
    """
    .gz compressed file को extract करके plain file बनाओ
    
    Purpose: Compressed .json.gz file को decompress करके readable .json file बनाना
    
    Args:
        source: Input .gz file path (Path object)
        destination: Output uncompressed file path (Path object)
    
    Process:
        1. Source .gz file को binary read mode में open करो
        2. Destination file को binary write mode में open करो
        3. Decompressed data को streaming copy करो (memory efficient)
    
    Example:
        extract_gzip(Path("data/file.json.gz"), Path("data/file.json"))
    
    Note: gzip.open() automatically decompression handle करता है
    """
    with gzip.open(source, "rb") as gz_file, destination.open("wb") as out_file:  # .gz file open करो read mode में और output file write mode में
        shutil.copyfileobj(gz_file, out_file)  # Decompressed data को output file में copy करो (chunk by chunk)


def clean_previous_downloads(data_dir=DATA_DIR):
    """
    पुरानी downloaded files को delete करो (fresh download के लिए)
    
    Purpose: पुराने instrument data को remove करना ताकि नया fresh data download हो सके
    
    Args:
        data_dir: Data directory path (default: DATA_DIR)
    
    Process:
        1. .gz और .json दोनों files के paths बनाओ
        2. अगर files exist करती हैं तो delete करो
    
    Example:
        clean_previous_downloads()  # Deletes data/complete.json.gz and data/complete.json
    
    Why needed: Daily fresh data चाहिए क्योंकि instruments change होते हैं (expiry, new listings)
    """
    gz_path = data_dir / "complete.json.gz"  # Compressed file path
    json_path = data_dir / "complete.json"  # Extracted JSON file path
    for path in (gz_path, json_path):  # दोनों files के लिए iterate करो
        if path.exists():  # अगर file exist करती है
            path.unlink()  # File delete करो (unlink = delete का POSIX term)


def download_and_extract(data_dir=DATA_DIR):
    """
    Complete workflow: Download + Extract instruments list
    
    Purpose: Upstox archive को download करना और extract करना (one-step process)
    
    Args:
        data_dir: Data directory जहां files save होंगी (default: DATA_DIR)
    
    Returns:
        tuple: (download_path, extracted_path)
            - download_path: Downloaded .gz file का path
            - extracted_path: Extracted .json file का path
    
    Process:
        1. Download path और extracted path define करो
        2. Upstox URL से .gz file download करो
        3. .gz file को extract करके .json बनाओ
        4. दोनों paths return करो
    
    Example:
        gz_file, json_file = download_and_extract()
        print(f"Downloaded: {gz_file}")
        print(f"Extracted: {json_file}")
    
    Used by: fast_api.py में startup time पर और /download endpoint से
    """
    download_path = data_dir / "complete.json.gz"  # Download होने वाली compressed file का path
    extracted_path = download_path.with_suffix("")  # .gz suffix remove करके .json path बनाओ

    download_file(UPSTOX_URL, download_path)  # Upstox से file download करो
    extract_gzip(download_path, extracted_path)  # Downloaded file को extract करो
    return download_path, extracted_path  # दोनों paths return करो


def main():
    """
    CLI entry point - जब script directly run हो (python download_extract.py)
    
    Purpose: Command line से manually instruments download करने के लिए
    
    Process:
        1. पुराने downloads clean करो
        2. नई files download और extract करो
        3. Paths को console पर print करो (user को feedback के लिए)
    
    Example usage:
        $ python download_extract.py
        Downloaded to: /path/to/data/complete.json.gz
        Extracted to: /path/to/data/complete.json
    """
    clean_previous_downloads()  # पुराने files delete करो
    download_path, extracted_path = download_and_extract()  # Download और extract करो

    # Success messages print करो with absolute paths
    print(f"Downloaded to: {download_path.resolve()}")  # .resolve() absolute path return करता है
    print(f"Extracted to: {extracted_path.resolve()}")


# यह check करता है कि script directly run हो रही है (not imported as module)
if __name__ == "__main__":
    main()  # main() function call करो
