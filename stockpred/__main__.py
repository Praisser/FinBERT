"""Command line: python -m stockpred AAPL [--years 5] [--query "Apple stock"]"""

import argparse
import logging

from .config import HISTORY_YEARS
from .pipeline import run

parser = argparse.ArgumentParser(prog="stockpred", description="Run the full prediction pipeline for a stock.")
parser.add_argument("ticker", help="Yahoo Finance ticker, e.g. AAPL or RELIANCE.NS")
parser.add_argument("--years", type=int, default=HISTORY_YEARS, help="years of history to use")
parser.add_argument("--query", help="news search query (default: '<TICKER> stock')")
args = parser.parse_args()

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("stockpred").setLevel(logging.INFO)
run(args.ticker, years=args.years, news_query=args.query)
