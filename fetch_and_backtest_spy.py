#!/usr/bin/env python3
"""
Download 1 month SPY 5-minute data via yfinance, run the backtest,
and print performance metrics.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import logging
from BacktestingEngine import BacktestingEngine, CommanderStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CSV_PATH = 'spy_1mo_5m.csv'


def fetch_spy_csv(csv_path: str = CSV_PATH):
    logger.info('Downloading SPY 1 month 5m from yfinance')
    df = yf.download('SPY', period='1mo', interval='5m', progress=False)
    if df.empty:
        logger.error('No data downloaded from yfinance')
        return None

    # If yfinance returned MultiIndex columns (ticker, field), normalize to stable string names
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(p) for p in col]).strip() for col in df.columns]

    df = df.reset_index()
    # Ensure timezone-aware Datetime (first column after reset_index is the datetime index)
    datetime_col = df.columns[0]
    df = df.rename(columns={datetime_col: 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    if df['Datetime'].dt.tz is None:
        try:
            df['Datetime'] = df['Datetime'].dt.tz_localize('America/New_York')
        except Exception:
            df['Datetime'] = df['Datetime'].dt.tz_localize('UTC')

    # Convert to UTC and unix timestamp seconds
    df['datetime_utc'] = df['Datetime'].dt.tz_convert('UTC')
    df['timestamp'] = (df['datetime_utc'].astype('int64') // 10**9).astype(int)

    # Build output DataFrame using robust column selection (case-insensitive)
    # Find OHLCV column names in the downloaded frame (case-insensitive, robust)
    col_candidates = {}
    col_names = [str(c) for c in df.columns]
    for field in ['open', 'high', 'low', 'close', 'volume']:
        found = None
        for cname in col_names:
            lname = cname.lower()
            if field == lname or lname.startswith(field) or lname.endswith(field) or field in lname:
                found = cname
                break
        if not found:
            logger.error(f'Missing required column for {field} in downloaded data; available: {col_names[:10]}')
            return None
        col_candidates[field] = found

    if 'Datetime' not in df.columns:
        logger.error('Datetime column not found after reset_index')
        return None

    out = df[[
        'Datetime',
        col_candidates['open'], col_candidates['high'], col_candidates['low'],
        col_candidates['close'], col_candidates['volume']
    ]].copy()

    out = out.rename(columns={
        'Datetime': 'datetime_utc',
        col_candidates['open']: 'open', col_candidates['high']: 'high',
        col_candidates['low']: 'low', col_candidates['close']: 'close',
        col_candidates['volume']: 'volume'
    })

    # Ensure timezone-aware UTC datetimes and unix timestamp seconds
    out['datetime_utc'] = pd.to_datetime(out['datetime_utc'], utc=True)
    # Convert to timezone-aware UTC and then to timezone-naive for integer conversion
    out['datetime_utc'] = out['datetime_utc'].dt.tz_convert('UTC')
    out['timestamp'] = out['datetime_utc'].dt.tz_convert('UTC').dt.tz_localize(None).astype('datetime64[s]').astype(int)

    out['symbol'] = 'SPY'
    out['timeframe'] = '5min'

    out.to_csv(csv_path, index=False)
    logger.info(f'Written {len(out)} bars to {csv_path}')
    return csv_path


def run_backtest_on_csv(csv_path: str = CSV_PATH):
    engine = BacktestingEngine()

    # Use CSV ingestion path in run_data_pipeline
    df = engine.run_data_pipeline('SPY', use_csv=True, csv_path=csv_path)
    if df.empty:
        logger.error('Processed DataFrame is empty')
        return

    strategy = CommanderStrategy()
    df = strategy.generate_entry_signals(df)
    trades = strategy.simulate_trades(df, capital=10000.0)
    metrics = strategy.calculate_performance_metrics(trades)

    print('\n=== REAL SPY 1-MONTH BACKTEST ===')
    if 'error' in metrics:
        print('No trades executed:', metrics['error'])
        return

    summary = metrics['summary']
    print(f"Total Trades: {summary['total_trades']}")
    print(f"Win Rate: {summary['win_rate']:.1%}")
    print(f"Total P&L: ${summary['total_pnl']:.2f}")
    print(f"Profit Factor: {summary['profit_factor']:.2f}")
    print('Regime distribution:', df['regime'].value_counts().to_dict())

    return metrics, trades, df


if __name__ == '__main__':
    csv = fetch_spy_csv()
    if csv:
        run_backtest_on_csv(csv)
