#!/usr/bin/env python3

"""Download the Upstox complete instrument list and extract the JSON payload."""

import gzip
import shutil
from pathlib import Path
from urllib.request import Request, urlopen


UPSTOX_URL = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"
DATA_DIR = Path("data")


def download_file(url, destination):
    """Download ``url`` to ``destination`` using a streaming copy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request) as response, destination.open("wb") as local_file:
        shutil.copyfileobj(response, local_file)


def extract_gzip(source, destination):
    """Extract a .gz file to its uncompressed destination."""
    with gzip.open(source, "rb") as gz_file, destination.open("wb") as out_file:
        shutil.copyfileobj(gz_file, out_file)


def clean_previous_downloads(data_dir=DATA_DIR):
    """Remove previously downloaded artifacts if they exist."""
    gz_path = data_dir / "complete.json.gz"
    json_path = data_dir / "complete.json"
    for path in (gz_path, json_path):
        if path.exists():
            path.unlink()


def download_and_extract(data_dir=DATA_DIR):
    """Download the Upstox archive and extract it into ``data_dir``."""
    download_path = data_dir / "complete.json.gz"
    extracted_path = download_path.with_suffix("")

    download_file(UPSTOX_URL, download_path)
    extract_gzip(download_path, extracted_path)
    return download_path, extracted_path


def main():
    clean_previous_downloads()
    download_path, extracted_path = download_and_extract()

    print(f"Downloaded to: {download_path.resolve()}")
    print(f"Extracted to: {extracted_path.resolve()}")


if __name__ == "__main__":
    main()
