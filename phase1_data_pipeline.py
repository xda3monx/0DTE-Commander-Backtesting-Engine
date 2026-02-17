"""
Phase 1: Data Pipeline for 0DTE Commander System
===============================================

Loads SPY, VIX, and Mag7 historical data from SQLite database,
verifies 5-minute alignment, fills gaps if necessary, and merges
into a single DataFrame for indicator calculation.

Author: AI Assistant
Date: February 14, 2026
"""

import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import yfinance as yf
import schwab
from schwab.auth import easy_client

# Schwab API credentials
API_KEY = 'BR0o3XFTyp4HUy5z7dWYX3IzgWGlJrIN'
APP_SECRET = 'mgG7N0t2AKGSAIKu'
CALLBACK_URL = 'https://127.0.0.1:8182'
TOKEN_PATH = 'schwab_token.json'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPipeline:
    """
    Handles data ingestion, storage, and merging for the 0DTE Commander System.
    
    Supports fetching from yFinance and storing in SQLite for efficient querying.
    """
    
    def __init__(self, db_path: str = 'backtesting_data.db'):
        """
        Initialize the data pipeline.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
        self.schwab_client = None
        
    def _create_tables(self):
        """Create database tables for storing historical data."""
        # Historical OHLCV data table (if not exists)
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS historical_data (
            symbol TEXT,
            timeframe TEXT,
            timestamp INTEGER,
            datetime_utc TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (symbol, timeframe, timestamp)
        )
        ''')
        self.conn.commit()
        logger.info("Database tables ready")
    
    def authenticate_schwab(self):
        """Authenticate with Schwab API."""
        if self.schwab_client is not None:
            return True
            
        try:
            self.schwab_client = easy_client(
                api_key=API_KEY,
                app_secret=APP_SECRET,
                callback_url=CALLBACK_URL,
                token_path=TOKEN_PATH,
                enforce_enums=False
            )
            logger.info("Schwab API authenticated successfully")
            return True
        except Exception as e:
            logger.error(f"Schwab authentication failed: {e}")
            return False
    
    def fetch_yfinance_data(self, symbol: str, period: str = '1mo', interval: str = '5m') -> pd.DataFrame:
        """
        Fetch historical data from yFinance.
        
        Args:
            symbol: Ticker symbol (e.g., 'SPY', '^VIX')
            period: Time period (e.g., '1mo', '6mo')
            interval: Data interval (e.g., '5m', '1h')
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching {period} {interval} data for {symbol} from yFinance")
        
        try:
            ticker = yf.Ticker(symbol)
            
            # For indices and VIX, use 1h for longer history if 5m fails
            if interval == '5m' and symbol in ['^VIX', '^GSPC']:
                try:
                    df = ticker.history(period='6mo', interval='1h', prepost=True)
                    interval = '1h'  # Update for metadata
                except:
                    logger.info(f"1h data not available for {symbol}, trying 1d")
                    df = ticker.history(period='2y', interval='1d', prepost=True)
                    interval = '1d'
            else:
                df = ticker.history(period=period, interval=interval, prepost=True)
            
            if df.empty:
                logger.warning(f"No data retrieved for {symbol}")
                return pd.DataFrame()
            
            # Reset index to get datetime
            df = df.reset_index()
            
            # Standardize column names - handle both 'Date' and 'Datetime'
            rename_dict = {
                'Date': 'datetime_utc',
                'Datetime': 'datetime_utc',
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }
            df = df.rename(columns=rename_dict)
            
            # Convert datetime to UTC and add timestamp
            if 'datetime_utc' in df.columns:
                df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], utc=True)
                df['timestamp'] = df['datetime_utc'].astype(int) // 10**9
            
            # Add metadata
            df['symbol'] = symbol
            df['timeframe'] = f"{interval.replace('m', '').replace('h', '')}{'min' if 'm' in interval else 'hour' if 'h' in interval else interval}"
            
            # Select and order columns
            columns = ['symbol', 'timeframe', 'timestamp', 'datetime_utc', 
                      'open', 'high', 'low', 'close', 'volume']
            df = df[columns]
            
            logger.info(f"Successfully fetched {len(df)} bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_schwab_historical_data(self, symbol: str, period: str = '6mo', interval: str = '5m') -> pd.DataFrame:
        """
        Fetch historical data from Schwab API.
        
        Args:
            symbol: Ticker symbol
            period: Time period ('1mo', '3mo', '6mo', '1y', etc.)
            interval: Data interval ('1m', '5m', '1h', '1d')
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.authenticate_schwab():
            logger.warning("Schwab authentication failed, falling back to yFinance")
            return self.fetch_yfinance_data(symbol, period, interval)
        
        logger.info(f"Fetching {period} {interval} data for {symbol} from Schwab API")
        
        try:
            # Convert interval to Schwab format
            if interval == '5m':
                frequency_type = 'minute'
                frequency = 5
                # Minute data requires period_type='day'
                req_period_type = 'day'
                req_period = 10
            elif interval == '1h':
                frequency_type = 'minute'
                frequency = 60
                req_period_type = 'day'
                req_period = 10
            elif interval == '1d':
                frequency_type = 'daily'
                frequency = 1
                req_period_type = 'month'
                req_period = 6
            else:
                frequency_type = 'minute'
                frequency = 5
                req_period_type = 'day'
                req_period = 5
            
            # Fetch historical data
            response = self.schwab_client.get_price_history(
                symbol=symbol,
                period_type=req_period_type,
                period=req_period,
                frequency_type=frequency_type,
                frequency=frequency,
                need_extended_hours_data=False
            )
            
            if response.status_code != 200:
                logger.warning(f"Schwab API returned status {response.status_code}, falling back to yFinance")
                return self.fetch_yfinance_data(symbol, period, interval)
            
            data = response.json()
            candles = data.get('candles', [])
            
            if not candles:
                logger.warning(f"No data from Schwab API for {symbol}, falling back to yFinance")
                return self.fetch_yfinance_data(symbol, period, interval)
            
            df = pd.DataFrame(candles)
            
            # Handle missing volume for indices
            if 'volume' not in df.columns:
                df['volume'] = 0
            
            # Standardize column names
            df = df.rename(columns={
                'datetime': 'timestamp_ms'
            })
            
            # Convert timestamp from ms to datetime
            df['datetime_utc'] = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True)
            df['timestamp'] = df['timestamp_ms'] // 1000
            
            # Add metadata
            df['symbol'] = symbol
            df['timeframe'] = f"{frequency}min" if frequency_type == 'minute' else f"{frequency}d"
            
            # Select columns
            columns = ['symbol', 'timeframe', 'timestamp', 'datetime_utc', 
                      'open', 'high', 'low', 'close', 'volume']
            df = df[columns]
            
            logger.info(f"Successfully fetched {len(df)} bars for {symbol} from Schwab API")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Schwab data for {symbol}: {e}, falling back to yFinance")
            return self.fetch_yfinance_data(symbol, period, interval)
    
    def store_data(self, df: pd.DataFrame):
        """
        Store DataFrame in SQLite database.
        
        Args:
            df: DataFrame with historical data
        """
        if df.empty:
            logger.warning("No data to store")
            return
        
        try:
            # Convert datetime to string for SQLite
            df_to_store = df.copy()
            df_to_store['datetime_utc'] = df_to_store['datetime_utc'].astype(str)
            
            # Check for existing data to avoid duplicates
            symbol = df_to_store.iloc[0]['symbol']
            timeframe = df_to_store.iloc[0]['timeframe']
            
            existing_timestamps = set()
            try:
                cur = self.conn.cursor()
                cur.execute("SELECT timestamp FROM historical_data WHERE symbol=? AND timeframe=?",
                           (symbol, timeframe))
                existing_timestamps = {row[0] for row in cur.fetchall()}
            except Exception:
                pass
            
            # Filter out existing timestamps
            if existing_timestamps:
                df_to_store = df_to_store[~df_to_store['timestamp'].isin(existing_timestamps)]
            
            if df_to_store.empty:
                logger.info(f"All data for {symbol} already exists in database")
                return
            
            # Store new data
            try:
                df_to_store.to_sql('historical_data', self.conn, if_exists='append', index=False)
                logger.info(f"Stored {len(df_to_store)} new bars for {symbol}")
            except Exception as store_error:
                logger.warning(f"Some data may already exist, attempting to insert only new records: {store_error}")
                # If append fails, try to insert row by row, skipping duplicates
                inserted = 0
                for _, row in df_to_store.iterrows():
                    try:
                        self.conn.execute('''
                            INSERT OR IGNORE INTO historical_data 
                            (symbol, timeframe, timestamp, datetime_utc, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (row['symbol'], row['timeframe'], row['timestamp'], 
                              row['datetime_utc'], row['open'], row['high'], 
                              row['low'], row['close'], row['volume']))
                        inserted += 1
                    except Exception:
                        pass
                self.conn.commit()
                logger.info(f"Inserted {inserted} new bars for {symbol} (skipped duplicates)")
            
        except Exception as e:
            logger.exception(f"Error storing data: {e}")
    
    def load_symbol_data(self, symbol: str, timeframe: str = '5min') -> pd.DataFrame:
        """
        Load historical data for a symbol from SQLite.
        
        Args:
            symbol: Ticker symbol
            timeframe: Timeframe string
            
        Returns:
            DataFrame with historical data
        """
        query = """
        SELECT * FROM historical_data 
        WHERE symbol = ? AND timeframe = ?
        ORDER BY timestamp ASC
        """
        
        df = pd.read_sql_query(query, self.conn, params=(symbol, timeframe))
        
        if not df.empty:
            df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], utc=True)
        
        logger.info(f"Loaded {len(df)} bars for {symbol}")
        return df
    
    def load_internals_data(self, lookback_periods: int = 100) -> pd.DataFrame:
        """
        Load Internals data (UVOL/DVOL/TRIN/TICK/Mag7) from database.
        
        Args:
            lookback_periods: Number of periods to load
            
        Returns:
            DataFrame with Internals data aligned to timestamps
        """
        try:
            # Query internals data from database
            query = '''
            SELECT DISTINCT timestamp FROM internals_data
            WHERE timeframe = '5m'
            ORDER BY timestamp DESC
            LIMIT ?
            '''
            
            timestamps = self.conn.execute(query, (lookback_periods,)).fetchall()
            
            if not timestamps:
                logger.warning("No internals data found in database")
                return pd.DataFrame()
            
            timestamps = sorted([t[0] for t in timestamps])
            
            # Load each symbol (use uppercase as stored in DB)
            symbols = ['$UVOL', '$DVOL', '$TRIN', '$TICK', '^MAG7']
            dfs = {}
            
            for symbol in symbols:
                query = '''
                SELECT timestamp, datetime_utc, close FROM internals_data
                WHERE symbol = ? AND timeframe = '5m'
                ORDER BY timestamp
                '''
                
                df = pd.read_sql_query(query, self.conn, params=(symbol,))
                
                if not df.empty:
                    clean_symbol = symbol.replace('^', '').lower()
                    df = df.rename(columns={'close': f'{clean_symbol}_close'})
                    df = df.drop('datetime_utc', axis=1)
                    dfs[symbol] = df
            
            if not dfs:
                logger.warning("No internals data could be retrieved")
                return pd.DataFrame()
            
            # Merge on timestamp
            merged = pd.DataFrame({'timestamp': timestamps})
            for symbol, df in dfs.items():
                merged = merged.merge(df, on='timestamp', how='left')
            
            # Forward fill gaps
            for col in merged.columns:
                if col != 'timestamp':
                    merged[col] = merged[col].ffill()
            
            logger.info(f"Loaded {len(merged)} internals data points with {len(merged.columns)} columns")
            return merged
            
        except Exception as e:
            logger.error(f"Failed to load internals data: {e}")
            return pd.DataFrame()
    
    def merge_dataframes(self, dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge multiple symbol DataFrames on timestamp.
        
        Args:
            dfs: Dictionary of symbol -> DataFrame
            
        Returns:
            Merged DataFrame with aligned timestamps
        """
        if not dfs:
            return pd.DataFrame()
        
        # Start with the first DataFrame
        merged_df = None
        
        for symbol, df in dfs.items():
            if df.empty:
                continue
                
            # Rename columns to include symbol prefix (remove ^ and make lowercase)
            clean_symbol = symbol.replace('^', '').lower()
            rename_dict = {}
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    rename_dict[col] = f"{clean_symbol}_{col}"
            
            df_renamed = df.rename(columns=rename_dict)
            
            # Keep only essential columns
            keep_cols = ['timestamp', 'datetime_utc'] + list(rename_dict.values())
            df_renamed = df_renamed[keep_cols]
            
            if merged_df is None:
                merged_df = df_renamed
            else:
                # Merge on timestamp
                merged_df = pd.merge(merged_df, df_renamed, on=['timestamp', 'datetime_utc'], how='outer')
        
        if merged_df is not None:
            # Sort by timestamp
            merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
            
            # Forward fill missing values (limited to avoid gaps)
            merged_df = merged_df.ffill(limit=5)
            
            logger.info(f"Merged data has {len(merged_df)} rows with {len(merged_df.columns)} columns")
        
        return merged_df if merged_df is not None else pd.DataFrame()
    
    def verify_alignment(self, df: pd.DataFrame, expected_interval: int = 300) -> pd.DataFrame:
        """
        Verify 5-minute alignment and fill gaps if necessary.
        
        Args:
            df: Merged DataFrame
            expected_interval: Expected interval in seconds (300 for 5min)
            
        Returns:
            DataFrame with verified alignment
        """
        if df.empty or 'timestamp' not in df.columns:
            return df
        
        df = df.copy()
        
        # Check for gaps in timestamps
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['timestamp_diff'] = df['timestamp'].diff()
        
        gaps = df[df['timestamp_diff'] > expected_interval]
        if not gaps.empty:
            logger.warning(f"Found {len(gaps)} gaps in data (>{expected_interval}s)")
            
            # For now, we'll forward fill gaps up to a reasonable limit
            # In production, you might want to re-fetch missing data
            df = df.ffill(limit=12)  # Max 1 hour gap
        
        df = df.drop(columns=['timestamp_diff'])
        logger.info("Data alignment verified")
        
        return df
    
    def run_phase1_pipeline(self) -> pd.DataFrame:
        """
        Execute Phase 1: Load SPY, VIX, Mag7 data and merge.
        
        Returns:
            Merged DataFrame ready for indicator calculation
        """
        logger.info("Starting Phase 1: Data Pipeline")
        
        # Define symbols to fetch/load
        symbols = ['SPY', '^VIX', '^VIX1D', '^GSPC']  # Added ^VIX1D for previous day VIX close
        
        # Mag7 symbols with weights (for later calculation)
        self.mag7_symbols = {
            'NVDA': 0.186,
            'AAPL': 0.217, 
            'MSFT': 0.199,
            'AMZN': 0.124,
            'GOOGL': 0.130,
            'META': 0.087,
            'TSLA': 0.056
        }
        
        # Load or fetch data for each symbol
        dataframes = {}
        
        for symbol in symbols:
            # Try to load from database first
            df = self.load_symbol_data(symbol)
            
            if df.empty:
                # Special handling for different symbols
                if symbol == '^VIX':
                    # Try 5-minute data first, fall back to hourly
                    df = self.fetch_yfinance_data(symbol, period='2mo', interval='5m')
                    if df.empty:
                        df = self.fetch_yfinance_data(symbol, period='6mo', interval='1h')
                elif symbol == '^VIX1D':
                    # VIX1D is previous day VIX close - fetch as daily data
                    df = self.fetch_yfinance_data(symbol, period='6mo', interval='1d')
                elif symbol in ['^GSPC']:
                    # For indices, try hourly first, then daily
                    df = self.fetch_yfinance_data(symbol, period='6mo', interval='1h')
                    if df.empty:
                        df = self.fetch_yfinance_data(symbol, period='2y', interval='1d')
                else:
                    df = self.fetch_yfinance_data(symbol, period='2mo', interval='5m')
                
                if not df.empty:
                    self.store_data(df)
            
            if not df.empty:
                dataframes[symbol] = df
        
        # Load Mag7 data
        mag7_data = {}
        for mag_symbol in self.mag7_symbols.keys():
            try:
                df = self.load_symbol_data(mag_symbol)
                if df.empty:
                    # For stocks, try 2 months of 5m data
                    df = self.fetch_schwab_historical_data(mag_symbol, period='2mo', interval='5m')
                    if not df.empty:
                        self.store_data(df)
                if not df.empty:
                    mag7_data[mag_symbol] = df
            except Exception as e:
                logger.warning(f"Skipping {mag_symbol} due to fetch error (likely date mismatch): {e}")
                continue
        
        # Add Mag7 to dataframes for merging
        dataframes.update(mag7_data)
        
        # Merge all dataframes
        merged_df = self.merge_dataframes(dataframes)
        
        # Special handling for VIX1D: align daily data to intraday timestamps
        if not merged_df.empty and 'vix1d_close' in merged_df.columns:
            # VIX1D is previous day close, so shift it forward to represent previous day
            merged_df['vix1d_close'] = merged_df['vix1d_close'].shift(1)
            # Forward fill to align with intraday data
            merged_df['vix1d_close'] = merged_df['vix1d_close'].ffill()
        
        # Verify alignment
        merged_df = self.verify_alignment(merged_df)
        
        logger.info("Phase 1 completed successfully")
        return merged_df
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

# Example usage
if __name__ == "__main__":
    pipeline = DataPipeline()
    
    try:
        # Run Phase 1
        df = pipeline.run_phase1_pipeline()
        
        if not df.empty:
            print("First 10 rows of merged DataFrame:")
            print(df.head(10))
            print(f"\nDataFrame shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
        else:
            print("No data available")
            
    finally:
        pipeline.close()