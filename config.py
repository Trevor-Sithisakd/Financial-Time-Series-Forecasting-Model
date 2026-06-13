MLFLOW_EXPERIMENT = "financial-forecasting"

# Full S&P 500 universe (~500 names). The first 100 are kept in market-cap
# order (the original curated leaders); the remaining constituents follow
# alphabetically. Symbols use yfinance format (dashes, e.g. BRK-B, BF-B).
# Sourced from the current S&P 500 constituent list, with recent ticker
# changes applied (MMC->MRSH, Block->XYZ, BNY Mellon->BNY, Fiserv->FI, etc.).
# Ticker membership drifts over time; the pipeline now skips any symbol
# yfinance can't return data for, so a few stale names won't break a run.
TICKERS       = [
    "AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "BRK-B", "JPM",
    "LLY", "V", "XOM", "MA", "COST", "UNH", "NFLX", "HD", "WMT", "PG",
    "JNJ", "BAC", "CRM", "AMD", "ORCL", "CVX", "MRK", "ABBV", "KO", "WFC",
    "CSCO", "ACN", "NOW", "IBM", "MS", "GS", "PM", "PEP", "DIS", "TMO",
    "INTU", "TXN", "ISRG", "AXP", "AMGN", "GE", "UBER", "BKNG", "QCOM", "RTX",
    "CAT", "SPGI", "BLK", "LOW", "HON", "PFE", "DE", "UNP", "NEE", "BSX",
    "AMAT", "SYK", "MU", "ADI", "PANW", "ADP", "REGN", "GILD", "ETN", "LRCX",
    "MRSH", "BA", "VRTX", "CB", "SO", "MCD", "KLAC", "CME", "CEG", "BMY",
    "CI", "PLD", "SNPS", "CDNS", "MCO", "ICE", "WM", "EOG", "CTAS", "ZTS",
    "TJX", "AON", "APH", "MSI", "MDLZ", "PH", "NSC", "COF", "WELL", "ITW",
    "MMM", "AOS", "ABT", "ADBE", "AES", "AFL", "A", "APD", "ABNB", "AKAM",
    "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL", "GOOG", "MO", "AMCR", "AEE",
    "AEP", "AIG", "AMT", "AWK", "AMP", "AME", "APA", "APO", "APP", "APTV",
    "ACGL", "ADM", "ARES", "ANET", "AJG", "AIZ", "T", "ATO", "ADSK", "AZO",
    "AVB", "AVY", "AXON", "BKR", "BALL", "BAX", "BDX", "BBY", "TECH", "BIIB",
    "BX", "XYZ", "BNY", "BR", "BRO", "BF-B", "BLDR", "BG", "BXP", "CHRW",
    "CPT", "CPB", "CAH", "CCL", "CARR", "CVNA", "CASY", "CBOE", "CBRE", "CDW",
    "COR", "CNC", "CNP", "CF", "CRL", "SCHW", "CHTR", "CMG", "CHD", "CIEN",
    "CINF", "C", "CFG", "CLX", "CMS", "CTSH", "COHR", "COIN", "CL", "CMCSA",
    "FIX", "CAG", "COP", "ED", "STZ", "COO", "CPRT", "GLW", "CPAY", "CTVA",
    "CSGP", "CRH", "CRWD", "CCI", "CSX", "CMI", "CVS", "DHR", "DRI", "DDOG",
    "DVA", "DECK", "DELL", "DAL", "DVN", "DXCM", "FANG", "DLR", "DG", "DLTR",
    "D", "DPZ", "DASH", "DOV", "DOW", "DHI", "DTE", "DUK", "DD", "EBAY",
    "SATS", "ECL", "EIX", "EW", "EA", "ELV", "EME", "EMR", "ETR", "EPAM",
    "EQT", "EFX", "EQIX", "EQR", "ERIE", "ESS", "EL", "EG", "EVRG", "ES",
    "EXC", "EXE", "EXPE", "EXPD", "EXR", "FFIV", "FDS", "FICO", "FAST", "FRT",
    "FDX", "FIS", "FITB", "FSLR", "FE", "FI", "F", "FTNT", "FTV", "FOXA",
    "FOX", "BEN", "FCX", "GRMN", "IT", "GEHC", "GEV", "GEN", "GNRC", "GD",
    "GIS", "GM", "GPC", "GPN", "GL", "GDDY", "HAL", "HIG", "HAS", "HCA",
    "DOC", "HSIC", "HSY", "HPE", "HLT", "HRL", "HST", "HWM", "HPQ", "HUBB",
    "HUM", "HBAN", "HII", "IEX", "IDXX", "INCY", "IR", "PODD", "INTC", "IBKR",
    "IFF", "IP", "IVZ", "INVH", "IQV", "IRM", "JBHT", "JBL", "JKHY", "J",
    "JCI", "KVUE", "KDP", "KEY", "KEYS", "KMB", "KIM", "KMI", "KKR", "KHC",
    "KR", "LHX", "LH", "LVS", "LDOS", "LEN", "LII", "LIN", "LYV", "LMT",
    "L", "LULU", "LITE", "LYB", "MTB", "MPC", "MAR", "MLM", "MAS", "MKC",
    "MCK", "MDT", "MET", "MTD", "MGM", "MCHP", "MAA", "MRNA", "TAP", "MPWR",
    "MNST", "MOS", "MSCI", "NDAQ", "NTAP", "NEM", "NWSA", "NWS", "NKE", "NI",
    "NDSN", "NTRS", "NOC", "NCLH", "NRG", "NUE", "NVR", "NXPI", "ORLY", "OXY",
    "ODFL", "OMC", "ON", "OKE", "OTIS", "PCAR", "PKG", "PLTR", "PSKY", "PAYX",
    "PYPL", "PNR", "PCG", "PSX", "PNW", "PNC", "POOL", "PPG", "PPL", "PFG",
    "PGR", "PRU", "PEG", "PTC", "PSA", "PHM", "PWR", "DGX", "Q", "RL",
    "RJF", "O", "REG", "RF", "RSG", "RMD", "RVTY", "HOOD", "ROK", "ROL",
    "ROP", "ROST", "RCL", "SNDK", "SBAC", "SLB", "STX", "SRE", "SHW", "SPG",
    "SWKS", "SJM", "SW", "SNA", "SOLV", "LUV", "SWK", "SBUX", "STT", "STLD",
    "STE", "SMCI", "SYF", "SYY", "TMUS", "TROW", "TTWO", "TPR", "TRGP", "TGT",
    "TEL", "TDY", "TER", "TPL", "TXT", "TKO", "TTD", "TSCO", "TT", "TDG",
    "TRV", "TRMB", "TFC", "TYL", "TSN", "USB", "UDR", "ULTA", "UAL", "UPS",
    "URI", "UHS", "VLO", "VEEV", "VTR", "VLTO", "VRSN", "VRSK", "VZ", "VRT",
    "VTRS", "VICI", "VST", "VMC", "WRB", "GWW", "WAB", "WBD", "WAT", "WEC",
    "WST", "WDC", "WY", "WSM", "WMB", "WTW", "WDAY", "WYNN", "XEL", "XYL",
    "YUM", "ZBRA", "ZBH",
]
START_DATE    = "2018-01-01"
END_DATE      = "2024-12-31"
FORWARD_DAYS  = 5   # trading days ahead to predict
MIN_TRAIN_YRS = 3   # minimum years of history before first test window
