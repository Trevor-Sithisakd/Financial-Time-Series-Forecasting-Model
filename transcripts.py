from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup
import os
import re

# Top 5 companies by market cap
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

dl = Downloader("YourName", "your-email@example.com", "./transcripts_raw")

# Download latest 5 8-K filings per company
for ticker in TICKERS:
    print(f"Downloading {ticker}...")
    dl.get("8-K", ticker, limit=5)

print("Done.")


def extract_text_from_doc(doc):
    text_match = re.search(r"<TEXT>(.*?)</TEXT>", doc, re.DOTALL | re.IGNORECASE)
    if not text_match:
        return None
    html_body = text_match.group(1).strip()
    soup = BeautifulSoup(html_body, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def extract_primary_document(full_submission_path):
    """Extract the press release (EX-99.1) if present, otherwise the 8-K body."""
    with open(full_submission_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    documents = re.split(r"<DOCUMENT>", content, flags=re.IGNORECASE)

    exhibit = None
    primary = None

    for doc in documents:
        doc_type_match = re.search(r"<TYPE>(.+)", doc)
        if not doc_type_match:
            continue
        doc_type = doc_type_match.group(1).strip()

        if doc_type in ("EX-99.1", "EX-99") and exhibit is None:
            exhibit = extract_text_from_doc(doc)
        elif doc_type in ("8-K", "8-K/A") and primary is None:
            primary = extract_text_from_doc(doc)

    return exhibit or primary


# Walk the downloaded folders and save clean .txt files
output_dir = "./transcripts_clean"
os.makedirs(output_dir, exist_ok=True)

for ticker in TICKERS:
    ticker_dir = f"./transcripts_raw/sec-edgar-filings/{ticker}/8-K"
    if not os.path.exists(ticker_dir):
        continue
    for filing in os.listdir(ticker_dir):
        filing_path = os.path.join(ticker_dir, filing)
        full_sub = os.path.join(filing_path, "full-submission.txt")
        if not os.path.exists(full_sub):
            continue

        text = extract_primary_document(full_sub)
        if text:
            out_file = f"{output_dir}/{ticker}_{filing}.txt"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Saved: {out_file}")
        else:
            print(f"No primary 8-K document found in: {full_sub}")
