"""
Data Ingestion Script for 0DTE Commander Backtesting Engine
==========================================================

This script demonstrates how to fetch 6 months of 5-minute OHLCV data
for SPX/SPY using the Schwab API or CSV fallback.

Usage:
    python data_ingestion.py --symbol SPX --months 6 --timeframe 5
    python data_ingestion.py --csv spy_5min_data.csv --symbol SPY
"""

import argparse
import pandas as pd
from datetime import datetime
import logging
import numpy as np
from BacktestingEngine import BacktestingEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_schwab_historical_data(symbol: str, months: int = 6, timeframe_minutes: int = 5) -> pd.DataFrame:
    """
    Fetch historical data using Schwab API.

    Note: Schwab API may have limitations on historical data depth.
    For extensive backtesting, consider using CSV data from other sources.
    """
    logger.info(f"Attempting to fetch {months} months of {timeframe_minutes}min data for {symbol} via Schwab API")

    engine = BacktestingEngine()

    try:
        df = engine.run_data_pipeline(
            symbol=symbol,
            months=months,
            timeframe_minutes=timeframe_minutes,
            use_csv=False
        )

        if not df.empty:
            logger.info(f"Successfully processed {len(df)} bars for {symbol}")
            return df
        else:
            logger.warning("No data retrieved from Schwab API")
            return pd.DataFrame()

    finally:
        engine.close()

def load_csv_historical_data(csv_path: str, symbol: str, timeframe_minutes: int = 5) -> pd.DataFrame:
    """
    Load historical data from CSV file.

    Expected CSV format (headers):
    timestamp,open,high,low,close,volume

    Where timestamp is Unix seconds (integer) or ISO datetime string.

    You can get this data from:
    - Yahoo Finance (via yfinance library)
    - Alpha Vantage
    - Polygon.io
    - Your broker's historical data export
    """
    logger.info(f"Loading historical data from {csv_path} for {symbol}")

    engine = BacktestingEngine()

    try:
        df = engine.run_data_pipeline(
            symbol=symbol,
            timeframe_minutes=timeframe_minutes,
            use_csv=True,
            csv_path=csv_path
        )

        if not df.empty:
            logger.info(f"Successfully processed {len(df)} bars from CSV")
            return df
        else:
            logger.warning("No data loaded from CSV")
            return pd.DataFrame()

    finally:
        engine.close()

def create_sample_csv_template(output_path: str = 'sample_data_template.csv'):
    """
    Create a sample CSV template showing the expected format.

    This is useful for understanding the required data format.
    """
    # Create sample data
    timestamps = pd.date_range('2023-08-01', periods=100, freq='5min')
    np.random.seed(42)  # For reproducible sample data

    # Generate realistic SPY-like price data
    base_price = 450.0
    prices = []
    current_price = base_price

    for i in range(len(timestamps)):
        # Random walk with slight upward trend
        change = np.random.normal(0.001, 0.02)  # mean=0.1%, std=2%
        current_price *= (1 + change)
        prices.append(current_price)

    # Create OHLCV data
    sample_data = []
    for i, price in enumerate(prices):
        # Add some noise to create realistic OHLC
        high_noise = np.random.uniform(0, 0.005)
        low_noise = np.random.uniform(0, 0.005)
        volume = np.random.randint(10000, 100000)

        ohlc = {
            'timestamp': int(timestamps[i].timestamp()),
            'datetime': timestamps[i].strftime('%Y-%m-%d %H:%M:%S'),
            'open': round(price * (1 + np.random.normal(0, 0.002)), 2),
            'high': round(price * (1 + high_noise), 2),
            'low': round(price * (1 - low_noise), 2),
            'close': round(price, 2),
            'volume': volume
        }
        sample_data.append(ohlc)

    df = pd.DataFrame(sample_data)
    df.to_csv(output_path, index=False)

    logger.info(f"Sample CSV template created: {output_path}")
    logger.info("Columns: timestamp,datetime,open,high,low,close,volume")
    logger.info("Note: 'datetime' column is optional (for readability)")

    return df

def demonstrate_vectorized_ema():
    """
    Demonstrate the vectorized EMA calculation without loops.
    """
    print("\n" + "="*60)
    print("VECTORIZED EMA CALCULATION TUTORIAL")
    print("="*60)

    # Create sample price data
    prices = [100.0, 101.5, 102.2, 103.8, 104.1, 105.3, 106.7, 107.2, 108.9, 109.4,
              110.1, 111.3, 112.5, 113.2, 114.8, 115.6, 116.9, 117.3, 118.7, 119.2]

    df = pd.DataFrame({'close': prices})

    print("Sample close prices (first 10):")
    print(df['close'].head(10).to_string())
    print()

    # VECTORIZED EMA CALCULATION - No loops needed!
    # This is the "Pandas mindset" vs "loop-based OOP mindset"

    # 8-period EMA (Drift component)
    df['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()

    # 24-period EMA (would need more data for full calculation)
    df['ema_24'] = df['close'].ewm(span=24, adjust=False).mean()

    print("8-period EMA calculation (vectorized):")
    print(df[['close', 'ema_8']].head(15).to_string())
    print()

    print("Key concepts for transitioning from loops to vectorized:")
    print("1. .ewm(span=N) - Exponential Weighted Moving Average")
    print("2. adjust=False - Uses the finance/EWMA formula (no centering)")
    print("3. .mean() - Calculates the EMA values")
    print("4. No explicit loops - Pandas handles all iteration internally")
    print("5. NumPy optimized - Much faster than manual loops")
    print()

    # Show how to combine multiple EMAs
    df['ema_diff'] = df['ema_8'] - df['ema_24']  # Drift strength
    df['ema_slope'] = df['ema_8'].diff()  # Drift direction

    print("Combining EMAs for drift analysis:")
    print(df[['close', 'ema_8', 'ema_24', 'ema_diff', 'ema_slope']].tail(10).to_string())

def main():
    parser = argparse.ArgumentParser(description='Data Ingestion for 0DTE Commander Backtesting')
    parser.add_argument('--symbol', type=str, default='SPY',
                       help='Trading symbol (default: SPY)')
    parser.add_argument('--months', type=int, default=6,
                       help='Months of historical data (default: 6)')
    parser.add_argument('--timeframe', type=int, default=5,
                       help='Timeframe in minutes (default: 5)')
    parser.add_argument('--csv', type=str,
                       help='Path to CSV file (use instead of API)')
    parser.add_argument('--create-template', action='store_true',
                       help='Create sample CSV template')
    parser.add_argument('--demo-ema', action='store_true',
                       help='Demonstrate vectorized EMA calculation')

    args = parser.parse_args()

    if args.create_template:
        create_sample_csv_template()
        return

    if args.demo_ema:
        demonstrate_vectorized_ema()
        return

    if args.csv:
        # Load from CSV
        df = load_csv_historical_data(args.csv, args.symbol, args.timeframe)
    else:
        # Try Schwab API
        df = fetch_schwab_historical_data(args.symbol, args.months, args.timeframe)

    if not df.empty:
        print(f"\n✅ Successfully processed {len(df)} bars for {args.symbol}")
        print(f"Date range: {df['datetime_utc'].min()} to {df['datetime_utc'].max()}")
        print(f"Sample data:")
        print(df[['datetime_utc', 'open', 'high', 'low', 'close', 'volume']].head().to_string())

        # Show calculated indicators
        indicator_cols = [col for col in ['ema_8', 'ema_24', 'velocity', 'volatility', 'regime']
                         if col in df.columns]
        if indicator_cols:
            print(f"\nCalculated indicators: {', '.join(indicator_cols)}")
            print(df[['datetime_utc'] + indicator_cols].head().to_string())
    else:
        print("❌ No data was processed. Check logs for details.")

        if not args.csv:
            print("\n💡 Tips for Schwab API:")
            print("1. Ensure you have valid Schwab API credentials")
            print("2. Check if your API plan supports historical data")
            print("3. Consider using CSV data from Yahoo Finance or other sources")
            print("4. Run with --create-template to see expected CSV format")

if __name__ == '__main__':
    main()