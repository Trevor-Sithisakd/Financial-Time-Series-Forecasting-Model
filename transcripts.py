from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup
import os

# Top 5 companies by market cap
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

dl = Downloader("YourName", "your-email@example.com", "./transcripts_raw")

# Download latest 5 8-K filings per company
for ticker in TICKERS:
    print(f"Downloading {ticker}...")
    dl.get("8-K", ticker, limit=5)

print("Done.")


def extract_text(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
        return soup.get_text(separator="\n", strip=True)

# Walk the downloaded folders and save clean .txt files
output_dir = "./transcripts_clean"
os.makedirs(output_dir, exist_ok=True)

for ticker in TICKERS:
    ticker_dir = f"./transcripts_raw/sec-edgar-filings/{ticker}/8-K"
    if not os.path.exists(ticker_dir):
        continue
    for filing in os.listdir(ticker_dir):
        filing_path = os.path.join(ticker_dir, filing)
        for fname in os.listdir(filing_path):
            if fname.endswith(".htm") or fname.endswith(".html"):
                text = extract_text(os.path.join(filing_path, fname))
                out_file = f"{output_dir}/{ticker}_{filing}.txt"
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"Saved: {out_file}")