"""Download the UCI Online Retail dataset into data/raw/.

    python scripts/download_data.py

If the download fails (firewall, UCI site down), download it manually from
https://archive.ics.uci.edu/dataset/352/online+retail and put
"Online Retail.xlsx" in data/raw/.
"""

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
DEST = Path(__file__).resolve().parents[1] / "data" / "raw"
TARGET = DEST / "Online Retail.xlsx"


def main():
    if TARGET.exists():
        print(f"Already downloaded: {TARGET}")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL} ...")
    try:
        with urllib.request.urlopen(URL, timeout=120) as r:
            payload = r.read()
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Download failed: {e}\nDownload manually (see docstring) and place the file at {TARGET}")

    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        xlsx = [n for n in z.namelist() if n.lower().endswith(".xlsx")]
        if not xlsx:
            sys.exit(f"No .xlsx found in archive: {z.namelist()}")
        TARGET.write_bytes(z.read(xlsx[0]))
    print(f"Saved {TARGET} ({TARGET.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
