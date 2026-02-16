"""
Unified Internals Data Handler for 0DTE Commander System
========================================================

Fetches and manages market internals indicators:
- UVOL: Up Volume (bullish volume indicator)
- DVOL: Down Volume (bearish volume indicator)
- TRIN: Arms Index (market breadth indicator)
- TICK: NYSE Advance/Decline indicator
- Mag7: Magnificent 7 index (mega-cap tech concentration)

These provide macro-level veto conditions for the Commander system.

Author: AI Assistant
Date: February 16, 2026
"""

import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging
import json
from schwab.auth import easy_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Schwab API credentials
API_KEY = 'BR0o3XFTyp4HUy5z7dWYX3IzgWGlJrIN'
APP_SECRET = 'mgG7N0t2AKGSAIKu'
CALLBACK_URL = 'https://127.0.0.1:8182'
TOKEN_PATH = 'schwab_token.json'


class InternalsDataHandler:
    """
    Handles fetching and managing Unified Internals data from Schwab API.
    
    Internals provide macro-level market health indicators:
    - UVOL/DVOL: Volume by direction
    - TRIN: Breadth index
    - TICK: Market breadth indicator
    - Mag7: Tech concentration risk
    """
    
    def __init__(self, db_path: str = 'backtesting_data.db'):
        """
        Initialize the Internals data handler.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.schwab_client = None
        self._create_internals_table()
        
    def _create_internals_table(self):
        """Create table for storing Internals data."""
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS internals_data (
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
        
        # Veto indicators table
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS internals_veto (
            timestamp INTEGER,
            datetime_utc TEXT,
            uvol REAL,
            dvol REAL,
            uvol_dvol_ratio REAL,
            trin REAL,
            tick INTEGER,
            mag7_concentration REAL,
            veto_uvol_dvol BOOLEAN,
            veto_trin_extreme BOOLEAN,
            veto_tick_extreme BOOLEAN,
            veto_mag7_concentration BOOLEAN,
            combined_veto BOOLEAN,
            PRIMARY KEY (timestamp)
        )
        ''')
        
        self.conn.commit()
        logger.info("Internals tables created")
    
    def refresh_schwab_token(self) -> bool:
        """
        Refresh Schwab API token using stored refresh token.
        
        Returns:
            True if refresh successful, False otherwise
        """
        try:
            # Read current token file
            with open(TOKEN_PATH, 'r') as f:
                token_data = json.load(f)
            
            refresh_token = token_data['token']['refresh_token']
            logger.info("Attempting to refresh Schwab token")
            
            # Use easy_client which handles token refresh
            self.schwab_client = easy_client(
                api_key=API_KEY,
                app_secret=APP_SECRET,
                callback_url=CALLBACK_URL,
                token_path=TOKEN_PATH
            )
            logger.info("Schwab token refreshed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            logger.info("Please re-authenticate with Schwab API manually")
            return False
    
    def authenticate_schwab(self, force_refresh: bool = False) -> bool:
        """
        Authenticate or refresh Schwab API connection.
        
        Args:
            force_refresh: Force token refresh even if client exists
            
        Returns:
            True if authenticated successfully
        """
        if self.schwab_client is not None and not force_refresh:
            return True
        
        try:
            self.schwab_client = easy_client(
                api_key=API_KEY,
                app_secret=APP_SECRET,
                callback_url=CALLBACK_URL,
                token_path=TOKEN_PATH
            )
            logger.info("Schwab API authenticated successfully")
            return True
        except Exception as e:
            logger.error(f"Schwab authentication failed: {e}")
            logger.info("Attempting token refresh...")
            return self.refresh_schwab_token()
    
    def fetch_internals_data(self, 
                            symbol: str,
                            period_days: int = 5,
                            interval: str = '5m') -> Optional[pd.DataFrame]:
        """
        Fetch Internals data from Schwab API.
        
        Args:
            symbol: Internals symbol (e.g., '$UVOL', '$DVOL', '$TRIN', '$TICK', '^MAG7')
            period_days: Number of days to fetch
            interval: Data interval ('5m', '1h', '1d')
            
        Returns:
            DataFrame with OHLCV data or None if fetch fails
        """
        if not self.authenticate_schwab():
            logger.error("Cannot fetch internals: Schwab authentication failed")
            return None
        
        try:
            # Get price history from Schwab
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)
            
            # Map interval to Schwab API format
            freq_map = {
                '5m': 'every_5_minutes',
                '15m': 'every_15_minutes',
                '1h': 'every_hour',
                '1d': 'daily'
            }
            
            freq = freq_map.get(interval, 'every_5_minutes')
            
            # Fetch using pricehistory
            response = self.schwab_client.pricehistory(
                symbol=symbol,
                period_type='day',
                period=period_days,
                frequency_type=freq.split('_')[1],  # Extract type (minute, hour, day)
                frequency=int(freq.split('_')[0]) if 'minute' in freq else 1,
                extended_hours=True
            )
            
            if response['candles'] is None or len(response['candles']) == 0:
                logger.warning(f"No data returned for {symbol}")
                return None
            
            # Convert to DataFrame
            data = []
            for candle in response['candles']:
                data.append({
                    'timestamp': candle['datetime'] // 1000,  # Convert ms to seconds
                    'datetime_utc': datetime.fromtimestamp(candle['datetime'] // 1000).isoformat(),
                    'open': candle['open'],
                    'high': candle['high'],
                    'low': candle['low'],
                    'close': candle['close'],
                    'volume': candle['volume']
                })
            
            df = pd.DataFrame(data)
            logger.info(f"Fetched {len(df)} bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            logger.info("Token may have expired. Attempting refresh...")
            if self.refresh_schwab_token():
                logger.info("Retrying fetch after token refresh...")
                return self.fetch_internals_data(symbol, period_days, interval)
            return None
    
    def fetch_all_internals(self, period_days: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Fetch all Internals data (UVOL, DVOL, TRIN, TICK, Mag7).
        
        Args:
            period_days: Number of days to fetch
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        internals_config = {
            '$UVOL': 'Up Volume - Total share volume of stocks trading on upticks',
            '$DVOL': 'Down Volume - Total share volume of stocks trading on downticks',
            '$TRIN': 'ARMS Index - Market breadth indicator (1.0 = balanced)',
            '$TICK': 'NYSE Advance/Decline Line',
            '^MAG7': 'Magnificent 7 Index - Mega-cap tech concentration'
        }
        
        all_data = {}
        for symbol, description in internals_config.items():
            logger.info(f"Fetching {symbol}: {description}")
            df = self.fetch_internals_data(symbol, period_days)
            if df is not None:
                all_data[symbol] = df
            else:
                logger.warning(f"Skipping {symbol} due to fetch failure")
        
        return all_data
    
    def store_internals_data(self, symbol: str, df: pd.DataFrame, timeframe: str = '5m'):
        """
        Store Internals data in SQLite database.
        
        Args:
            symbol: Internals symbol
            df: DataFrame with OHLCV data
            timeframe: Time interval
        """
        try:
            for _, row in df.iterrows():
                self.conn.execute('''
                INSERT OR REPLACE INTO internals_data 
                (symbol, timeframe, timestamp, datetime_utc, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, timeframe, row['timestamp'], row['datetime_utc'],
                    row['open'], row['high'], row['low'], row['close'], row['volume']
                ))
            
            self.conn.commit()
            logger.info(f"Stored {len(df)} bars for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to store internals data: {e}")
    
    def get_internals_dataframe(self, 
                               lookback_periods: int = 100,
                               timeframe: str = '5m') -> pd.DataFrame:
        """
        Get all internals data from database as merged DataFrame.
        
        Args:
            lookback_periods: Number of periods to retrieve
            timeframe: Time interval
            
        Returns:
            DataFrame with all internals aligned by timestamp
        """
        try:
            # Get unique timestamps (most recent ones)
            query = f'''
            SELECT DISTINCT timestamp FROM internals_data
            WHERE timeframe = ?
            ORDER BY timestamp DESC
            LIMIT ?
            '''
            
            timestamps = self.conn.execute(query, (timeframe, lookback_periods)).fetchall()
            timestamps = [t[0] for t in timestamps]
            
            if not timestamps:
                logger.warning("No internals data found in database")
                return pd.DataFrame()
            
            # Pivot data by symbol
            all_dfs = {}
            symbols = ['$UVOL', '$DVOL', '$TRIN', '$TICK', '^MAG7']
            
            for symbol in symbols:
                query = '''
                SELECT timestamp, datetime_utc, close FROM internals_data
                WHERE symbol = ? AND timeframe = ?
                ORDER BY timestamp
                '''
                df = pd.read_sql_query(query, self.conn, params=(symbol, timeframe))
                
                if not df.empty:
                    df = df.rename(columns={'close': f'{symbol.lower()}_close'})
                    df = df.drop('datetime_utc', axis=1)
                    all_dfs[symbol] = df
            
            # Merge all symbols by timestamp
            if not all_dfs:
                logger.warning("No internals data could be retrieved")
                return pd.DataFrame()
            
            merged = pd.DataFrame({'timestamp': timestamps})
            for symbol, df in all_dfs.items():
                merged = merged.merge(df, on='timestamp', how='left')
            
            # Forward fill any gaps
            merged = merged.fillna(method='ffill')
            
            merged['datetime_utc'] = pd.to_datetime(merged['timestamp'], unit='s', utc=True)
            merged = merged.sort_values('timestamp')
            
            logger.info(f"Retrieved {len(merged)} internals bars")
            return merged
            
        except Exception as e:
            logger.error(f"Failed to retrieve internals data: {e}")
            return pd.DataFrame()
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


if __name__ == '__main__':
    # Example usage
    handler = InternalsDataHandler()
    
    # Attempt authentication with Schwab (will trigger re-auth if token expired)
    authenticated = handler.authenticate_schwab(force_refresh=True)
    
    if authenticated:
        logger.info("Successfully authenticated with Schwab API")
        
        # Fetch all internals data
        all_data = handler.fetch_all_internals(period_days=5)
        
        # Store in database
        for symbol, df in all_data.items():
            if not df.empty:
                handler.store_internals_data(symbol, df)
        
        # Retrieve merged data
        internals_df = handler.get_internals_dataframe()
        logger.info(f"Internals DataFrame shape: {internals_df.shape}")
        logger.info(f"Columns: {list(internals_df.columns)}")
        
    else:
        logger.warning("Schwab authentication failed")
        logger.info("To use Internals data with Schwab API, manually authenticate:")
        logger.info("  1. Run: python")
        logger.info("  2. from schwab.auth import easy_client")
        logger.info("  3. easy_client(api_key='...', app_secret='...', callback_url='...', token_path='schwab_token.json')")
        logger.info("  4. Follow browser authentication flow")
        logger.info("")
        logger.info("After authentication, re-run internals_data.py to fetch data")
    
    handler.close()
