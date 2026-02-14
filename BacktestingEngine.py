"""
Backtesting Engine for 0DTE Commander System
==========================================

A robust vectorized backtesting framework for high-frequency trading strategies
based on Velocity, Drift, and Volatility Physics.

Author: Senior Quant Developer & Data Scientist
Date: February 2026

Architecture:
- Vectorized Pandas operations for performance
- SQLite for data persistence
- Modular indicator calculations
- Comprehensive backtesting metrics
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BacktestingEngine:
    """
    Main backtesting engine for the 0DTE Commander System.

    Features:
    - Vectorized data processing
    - Historical data ingestion
    - Indicator calculations (Drift: EMAs, Velocity: momentum, Volatility: ATR/VIX)
    - Strategy backtesting
    - Performance analytics
    """

    def __init__(self, db_path: str = 'backtesting_data.db'):
        """
        Initialize the backtesting engine.

        Args:
            db_path: Path to SQLite database for data storage
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        """Create necessary database tables for backtesting data."""

        # Historical OHLCV data table
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

        # Calculated indicators table
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS indicators (
                symbol TEXT,
                timeframe TEXT,
                timestamp INTEGER,
                datetime_utc TEXT,
                close REAL,
                ema_8 REAL,        -- Drift component: 8-period EMA
                ema_24 REAL,       -- Drift component: 24-period EMA
                velocity REAL,     -- Momentum/Rate of change
                volatility REAL,   -- ATR-based volatility
                vix_close REAL,    -- VIX for market volatility context
                regime TEXT,       -- Market regime classification
                PRIMARY KEY (symbol, timeframe, timestamp)
            )
        ''')

        # Backtest results table
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS backtest_results (
                strategy_name TEXT,
                symbol TEXT,
                timeframe TEXT,
                start_date TEXT,
                end_date TEXT,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                win_rate REAL,
                avg_win REAL,
                avg_loss REAL,
                profit_factor REAL,
                max_drawdown REAL,
                sharpe_ratio REAL,
                total_return REAL,
                created_at TEXT,
                PRIMARY KEY (strategy_name, symbol, timeframe, start_date, end_date)
            )
        ''')

        self.conn.commit()
        logger.info("Database tables created successfully")

    def fetch_historical_data_schwab(self, symbol: str, months: int = 6,
                                   timeframe_minutes: int = 5) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Schwab API.

        Args:
            symbol: Trading symbol (e.g., '$SPX', '/ES')
            months: Number of months of historical data to fetch
            timeframe_minutes: Bar timeframe in minutes

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Import Schwab client here to avoid circular imports
            from SchwabClient import authenticate_client

            client = authenticate_client()
            if not client:
                raise ConnectionError("Failed to authenticate with Schwab API")

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30 * months)

            logger.info(f"Fetching {months} months of {timeframe_minutes}min data for {symbol}")

            # Schwab API call for historical data
            # Note: Adjust based on actual Schwab API method for historical data
            if timeframe_minutes == 1:
                resp = client.get_price_history_every_minute(symbol)
            else:
                # For other timeframes, we'll need to use the appropriate method
                # This may require resampling or using different API endpoints
                resp = client.get_price_history(
                    symbol=symbol,
                    period_type='month',
                    period=str(months),
                    frequency_type='minute',
                    frequency=str(timeframe_minutes)
                )

            data = resp.json()

            if 'candles' not in data:
                logger.warning(f"No candle data found for {symbol}")
                return pd.DataFrame()

            df = pd.DataFrame(data['candles'])

            # Standardize column names and data types
            df = df.rename(columns={
                'datetime': 'timestamp'
            })

            # Convert timestamp from milliseconds to seconds
            df['timestamp'] = (df['timestamp'] / 1000).astype(int)

            # Add datetime column
            df['datetime_utc'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)

            # Ensure numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col not in df.columns:
                    df[col] = 0.0 if col != 'volume' else 0
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            # Add metadata columns
            df['symbol'] = symbol
            df['timeframe'] = f'{timeframe_minutes}min'

            # Reorder columns
            df = df[['symbol', 'timeframe', 'timestamp', 'datetime_utc',
                    'open', 'high', 'low', 'close', 'volume']]

            logger.info(f"Successfully fetched {len(df)} bars for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching data from Schwab API: {e}")
            return pd.DataFrame()

    def load_csv_data(self, csv_path: str, symbol: str, timeframe_minutes: int = 5) -> pd.DataFrame:
        """
        Load historical data from CSV file (fallback when API is limited).

        Expected CSV format: timestamp,open,high,low,close,volume
        Timestamp should be in Unix seconds format.

        Args:
            csv_path: Path to CSV file
            symbol: Trading symbol
            timeframe_minutes: Bar timeframe in minutes

        Returns:
            DataFrame with standardized OHLCV data
        """
        try:
            df = pd.read_csv(csv_path)

            # Standardize column names (handle various formats)
            column_mapping = {
                'timestamp': 'timestamp',
                'time': 'timestamp',
                'datetime': 'timestamp',
                'date': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'vol': 'volume'
            }

            df = df.rename(columns=column_mapping)

            # Convert timestamp if it's not already Unix seconds
            if 'timestamp' in df.columns:
                # Try to parse various timestamp formats
                if df['timestamp'].dtype == 'object':
                    # Try parsing as datetime string first
                    try:
                        df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) // 10**9
                    except:
                        # Assume it's already Unix timestamp
                        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
                else:
                    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')

            # Add datetime column
            df['datetime_utc'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)

            # Ensure numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col not in df.columns:
                    df[col] = 0.0 if col != 'volume' else 0
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            # Add metadata
            df['symbol'] = symbol
            df['timeframe'] = f'{timeframe_minutes}min'

            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Remove duplicates
            df = df.drop_duplicates(subset=['timestamp'])

            logger.info(f"Successfully loaded {len(df)} bars from CSV for {symbol}")
            return df[['symbol', 'timeframe', 'timestamp', 'datetime_utc',
                      'open', 'high', 'low', 'close', 'volume']]

        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            return pd.DataFrame()

    def store_historical_data(self, df: pd.DataFrame):
        """
        Store historical OHLCV data in the database.

        Args:
            df: DataFrame with historical data
        """
        if df.empty:
            logger.warning("No data to store")
            return

        try:
            df.to_sql('historical_data', self.conn, if_exists='append', index=False)
            logger.info(f"Stored {len(df)} historical bars in database")
        except Exception as e:
            logger.error(f"Error storing historical data: {e}")

    def calculate_drift_indicators_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Drift components using vectorized operations (no loops).

        Drift represents the trend direction and strength using EMAs.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with drift indicators added
        """
        # Make a copy to avoid modifying original
        df = df.copy()

        # Calculate 8-period EMA (fast drift component)
        df['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()

        # Calculate 24-period EMA (slow drift component)
        df['ema_24'] = df['close'].ewm(span=24, adjust=False).mean()

        # Calculate drift strength (difference between fast and slow EMA)
        df['drift_strength'] = df['ema_8'] - df['ema_24']

        # Calculate drift direction (slope of fast EMA)
        df['drift_direction'] = df['ema_8'].diff()

        logger.info("Calculated drift indicators (EMAs) using vectorized operations")
        return df

    def calculate_velocity_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Velocity components (momentum and rate of change).

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with velocity indicators added
        """
        df = df.copy()

        # Rate of Change (ROC) - momentum indicator
        df['roc_5'] = df['close'].pct_change(periods=5) * 100

        # Momentum (close - close_5_periods_ago)
        df['momentum_5'] = df['close'] - df['close'].shift(5)

        # Velocity as combination of ROC and momentum
        df['velocity'] = (df['roc_5'] * 0.7) + (df['momentum_5'].rolling(5).mean() * 0.3)

        logger.info("Calculated velocity indicators")
        return df

    def calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Volatility components using ATR and other measures.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with volatility indicators added
        """
        df = df.copy()

        # True Range calculation (vectorized)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = (df['high'] - df['close'].shift(1)).abs()
        df['tr3'] = (df['low'] - df['close'].shift(1)).abs()
        df['true_range'] = pd.concat([df['tr1'], df['tr2'], df['tr3']], axis=1).max(axis=1)

        # Average True Range (ATR) - 14 period
        df['volatility'] = df['true_range'].rolling(window=14).mean()

        # Normalized volatility (ATR as % of close)
        df['volatility_pct'] = (df['volatility'] / df['close']) * 100

        # Clean up temporary columns
        df = df.drop(['tr1', 'tr2', 'tr3', 'true_range'], axis=1)

        logger.info("Calculated volatility indicators (ATR)")
        return df

    def classify_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify market regime based on drift, velocity, and volatility.

        Args:
            df: DataFrame with calculated indicators

        Returns:
            DataFrame with regime classification
        """
        df = df.copy()

        # Regime classification logic
        conditions = [
            # High volatility trending up
            (df['drift_direction'] > 0) & (df['volatility_pct'] > 2.0) & (df['velocity'] > 0),
            # High volatility trending down
            (df['drift_direction'] < 0) & (df['volatility_pct'] > 2.0) & (df['velocity'] < 0),
            # Low volatility ranging
            (df['volatility_pct'] <= 2.0) & (abs(df['drift_strength']) < df['close'] * 0.005),
            # High volatility ranging (choppy)
            (df['volatility_pct'] > 2.0) & (abs(df['drift_strength']) < df['close'] * 0.01)
        ]

        choices = ['bull_trend', 'bear_trend', 'ranging', 'choppy']
        df['regime'] = np.select(conditions, choices, default='unknown')

        logger.info("Classified market regimes")
        return df

    def run_data_pipeline(self, symbol: str, months: int = 6, timeframe_minutes: int = 5,
                         use_csv: bool = False, csv_path: Optional[str] = None) -> pd.DataFrame:
        """
        Complete data ingestion and processing pipeline.

        Args:
            symbol: Trading symbol
            months: Months of historical data
            timeframe_minutes: Bar timeframe
            use_csv: Whether to use CSV instead of API
            csv_path: Path to CSV file (if use_csv=True)

        Returns:
            Processed DataFrame with all indicators
        """
        logger.info(f"Starting data pipeline for {symbol}")

        # 1. Data Ingestion
        if use_csv and csv_path:
            df = self.load_csv_data(csv_path, symbol, timeframe_minutes)
        else:
            df = self.fetch_historical_data_schwab(symbol, months, timeframe_minutes)

        if df.empty:
            logger.error("No data available for processing")
            return df

        # 2. Data Cleaning
        df = self.clean_data(df)

        # 3. Store raw data
        self.store_historical_data(df)

        # 4. Calculate Indicators (Vectorized)
        df = self.calculate_drift_indicators_vectorized(df)
        df = self.calculate_velocity_indicators(df)
        df = self.calculate_volatility_indicators(df)

        # 5. Classify Regimes
        df = self.classify_market_regime(df)

        # 6. Store processed data
        self.store_processed_data(df)

        logger.info(f"Data pipeline completed for {symbol}. Processed {len(df)} bars.")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate the data.

        Args:
            df: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        # Remove rows with invalid prices
        df = df[df['close'] > 0]
        df = df[df['open'] > 0]
        df = df[df['high'] >= df['close']]
        df = df[df['low'] <= df['close']]

        # Remove duplicate timestamps
        df = df.drop_duplicates(subset=['timestamp'])

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Forward fill missing values (limited to prevent lookahead bias)
        df = df.fillna(method='ffill', limit=5)

        # Drop remaining NaN values
        df = df.dropna()

        logger.info(f"Data cleaned: {len(df)} valid bars remaining")
        return df

    def store_processed_data(self, df: pd.DataFrame):
        """
        Store processed data with indicators in database.

        Args:
            df: DataFrame with calculated indicators
        """
        try:
            # Select columns for indicators table
            indicator_cols = [
                'symbol', 'timeframe', 'timestamp', 'datetime_utc', 'close',
                'ema_8', 'ema_24', 'velocity', 'volatility', 'regime'
            ]

            # Check if all required columns exist
            available_cols = [col for col in indicator_cols if col in df.columns]
            indicator_df = df[available_cols].copy()

            indicator_df.to_sql('indicators', self.conn, if_exists='append', index=False)
            logger.info(f"Stored {len(indicator_df)} processed bars with indicators")

        except Exception as e:
            logger.error(f"Error storing processed data: {e}")

    def get_historical_data(self, symbol: str, timeframe: str,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve historical data with indicators from database.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string (e.g., '5min')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with historical data and indicators
        """
        query = f"""
        SELECT h.*, i.ema_8, i.ema_24, i.velocity, i.volatility, i.regime
        FROM historical_data h
        LEFT JOIN indicators i ON h.symbol = i.symbol
            AND h.timeframe = i.timeframe
            AND h.timestamp = i.timestamp
        WHERE h.symbol = '{symbol}' AND h.timeframe = '{timeframe}'
        """

        if start_date:
            start_ts = int(pd.to_datetime(start_date).timestamp())
            query += f" AND h.timestamp >= {start_ts}"

        if end_date:
            end_ts = int(pd.to_datetime(end_date).timestamp())
            query += f" AND h.timestamp <= {end_ts}"

        query += " ORDER BY h.timestamp ASC"

        df = pd.read_sql_query(query, self.conn)

        if not df.empty:
            df['datetime_utc'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)

        return df

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


# Example usage and testing
if __name__ == '__main__':
    # Initialize the backtesting engine
    engine = BacktestingEngine()

    # Example: Load data using CSV (since API may have limitations for historical data)
    # For SPY data, you would download from Yahoo Finance or similar source
    # csv_path = 'spy_5min_6months.csv'  # Example CSV file

    # For demonstration, let's show the API approach
    print("Backtesting Engine initialized successfully!")
    print("\nTo fetch data, use:")
    print("df = engine.run_data_pipeline('$SPX', months=6, timeframe_minutes=5)")
    print("\nOr with CSV fallback:")
    print("df = engine.run_data_pipeline('SPY', use_csv=True, csv_path='spy_data.csv')")

    # Example of vectorized EMA calculation demonstration
    print("\n" + "="*60)
    print("VECTORIZED EMA CALCULATION DEMONSTRATION")
    print("="*60)

    # Create sample data
    sample_data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109] * 3
    })

    print("Sample close prices:")
    print(sample_data['close'].head(10).to_string())

    # Vectorized EMA calculation (no loops!)
    sample_data['ema_8_vectorized'] = sample_data['close'].ewm(span=8, adjust=False).mean()

    print("\n8-period EMA (vectorized, no loops):")
    print(sample_data[['close', 'ema_8_vectorized']].head(10).to_string())

    print("\nKey advantages of vectorized operations:")
    print("✅ No explicit loops - Pandas handles iteration internally")
    print("✅ NumPy optimized C code for maximum performance")
    print("✅ Handles NaN values gracefully")
    print("✅ Easy to combine with other vectorized operations")