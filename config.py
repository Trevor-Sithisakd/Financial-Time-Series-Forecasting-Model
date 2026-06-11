MLFLOW_EXPERIMENT = "financial-forecasting"

TICKERS       = ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "BRK-B", "JPM",
    "LLY", "V", "XOM", "MA", "COST", "UNH", "NFLX", "HD", "WMT", "PG",
    "JNJ", "BAC", "CRM", "AMD", "ORCL", "CVX", "MRK", "ABBV", "KO", "WFC",
    "CSCO", "ACN", "NOW", "IBM", "MS", "GS", "PM", "PEP", "DIS", "TMO",
    "INTU", "TXN", "ISRG", "AXP", "AMGN", "GE", "UBER", "BKNG", "QCOM", "RTX",
    "CAT", "SPGI", "BLK", "LOW", "HON", "PFE", "DE", "UNP", "NEE", "BSX",
    "AMAT", "SYK", "MU", "ADI", "PANW", "ADP", "REGN", "GILD", "ETN", "LRCX",
    "MMC", "BA", "VRTX", "CB", "SO", "MCD", "KLAC", "CME", "CEG", "BMY",
    "CI", "PLD", "SNPS", "CDNS", "MCO", "ICE", "WM", "EOG", "CTAS", "ZTS",
    "TJX", "AON", "APH", "MSI", "MDLZ", "PH", "NSC", "COF", "WELL", "ITW",]
START_DATE    = "2018-01-01"
END_DATE      = "2024-12-31"
FORWARD_DAYS  = 5   # trading days ahead to predict
MIN_TRAIN_YRS = 3   # minimum years of history before first test window
