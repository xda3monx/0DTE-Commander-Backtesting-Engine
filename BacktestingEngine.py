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

# Import our vectorized indicators
from vectorized_indicators import VectorizedIndicators

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
            ema_8 REAL,
            ema_24 REAL,
            velocity REAL,
            drift_velocity REAL,
            drift_spread REAL,
            fast_line REAL,
            slow_line REAL,
            volatility REAL,
            vwap REAL,
            vwap_upper REAL,
            vwap_lower REAL,
            adx REAL,
            regime TEXT,
            regime_state INTEGER,
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

    def load_csv_data(self, csv_path: str, symbol: str, timeframe_minutes: int = 5) -> pd.DataFrame:
        """
        Load historical data from CSV file.

        Expected CSV format: timestamp,open,high,low,close,volume
        Timestamp should be in Unix seconds format.
        """
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded CSV with columns: {list(df.columns)}")

            # Standardize column names
            timestamp_col = None
            for col in ['timestamp', 'time', 'datetime', 'date']:
                if col in df.columns:
                    timestamp_col = col
                    break

            if timestamp_col and timestamp_col != 'timestamp':
                df = df.rename(columns={timestamp_col: 'timestamp'})

            # Handle other columns
            column_mapping = {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'vol': 'volume'
            }
            df = df.rename(columns=column_mapping)

            # Convert timestamp if needed
            if 'timestamp' in df.columns:
                if df['timestamp'].dtype == 'object':
                    try:
                        df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) // 10**9
                    except:
                        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
                else:
                    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            else:
                logger.error("No timestamp column found")
                return pd.DataFrame()

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

            # Sort and remove duplicates
            df = df.sort_values('timestamp').reset_index(drop=True)
            df = df.drop_duplicates(subset=['timestamp'])

            logger.info(f"Successfully loaded {len(df)} bars from CSV for {symbol}")
            return df[['symbol', 'timeframe', 'timestamp', 'datetime_utc',
                      'open', 'high', 'low', 'close', 'volume']]

        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            return pd.DataFrame()

    def store_historical_data(self, df: pd.DataFrame):
        """Store historical OHLCV data in the database."""
        if df.empty:
            logger.warning("No data to store")
            return

        try:
            # Ensure datetime is stored as text to avoid sqlite adapter issues
            df_to_store = df.copy()
            if 'datetime_utc' in df_to_store.columns:
                df_to_store['datetime_utc'] = df_to_store['datetime_utc'].astype(str)

            # Avoid UNIQUE constraint errors by skipping timestamps already present
            try:
                cur = self.conn.cursor()
                cur.execute("SELECT timestamp FROM historical_data WHERE symbol=? AND timeframe=?",
                            (df_to_store.iloc[0]['symbol'], df_to_store.iloc[0]['timeframe']))
                existing = {r[0] for r in cur.fetchall()}
            except Exception:
                existing = set()

            if 'timestamp' in df_to_store.columns and existing:
                new_df = df_to_store[~df_to_store['timestamp'].isin(existing)].copy()
            else:
                new_df = df_to_store

            if new_df.empty:
                logger.info("No new historical bars to store (all timestamps already present)")
                return

            new_df.to_sql('historical_data', self.conn, if_exists='append', index=False)
            logger.info(f"Stored {len(new_df)} new historical bars in database")
        except Exception as e:
            logger.exception(f"Error storing historical data: {e}")

    def calculate_drift_indicators_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Drift components using vectorized operations.
        Drift represents trend direction and strength using EMAs.
        """
        df = df.copy()

        # Calculate EMAs
        df['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
        df['ema_24'] = df['close'].ewm(span=24, adjust=False).mean()

        # Calculate drift metrics
        df['drift_strength'] = df['ema_8'] - df['ema_24']
        df['drift_direction'] = df['ema_8'].diff()
        df['drift_velocity'] = df['drift_strength'].diff()
        df['drift_spread'] = (df['ema_8'] / df['ema_24'] - 1) * 100

        logger.info("Calculated drift indicators (EMAs) using vectorized operations")
        return df

    def calculate_velocity_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Velocity components (momentum and rate of change).
        Includes VWAP and Williams %R.
        """
        df = df.copy()

        # Rate of Change and Momentum
        df['roc_5'] = df['close'].pct_change(periods=5) * 100
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['velocity'] = (df['roc_5'] * 0.7) + (df['momentum_5'].rolling(5).mean() * 0.3)

        # Session-based VWAP reset
        df['datetime_et'] = df['datetime_utc'].dt.tz_convert('America/New_York')
        df['session_start'] = (
            (df['datetime_et'].dt.hour == 9) &
            (df['datetime_et'].dt.minute == 30) &
            (df['datetime_et'].dt.dayofweek < 5)
        )
        df['session_id'] = df['session_start'].cumsum()

        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['vp'] = df['volume'] * df['typical_price']

        df['vwap'] = (
            df.groupby('session_id')['vp'].cumsum() /
            df.groupby('session_id')['volume'].cumsum()
        )

        # Williams %R (momentum oscillator)
        highest_high_5 = df['high'].rolling(5).max()
        lowest_low_5 = df['low'].rolling(5).min()
        df['fast_line'] = -100 * (highest_high_5 - df['close']) / (highest_high_5 - lowest_low_5)

        highest_high_10 = df['high'].rolling(10).max()
        lowest_low_10 = df['low'].rolling(10).min()
        df['slow_line'] = -100 * (highest_high_10 - df['close']) / (highest_high_10 - lowest_low_10)

        logger.info("Calculated velocity indicators")
        return df

    def calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Volatility components using ATR.
        Also adds VWAP bands (requires ATR to be calculated first).
        """
        df = df.copy()

        # True Range calculation
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = (df['high'] - df['close'].shift(1)).abs()
        df['tr3'] = (df['low'] - df['close'].shift(1)).abs()
        df['true_range'] = pd.concat([df['tr1'], df['tr2'], df['tr3']], axis=1).max(axis=1)

        # Average True Range (ATR)
        df['volatility'] = df['true_range'].rolling(window=14).mean()
        df['volatility_pct'] = (df['volatility'] / df['close']) * 100

        # Add VWAP bands (now that ATR is calculated)
        if 'vwap' in df.columns:
            df['vwap_upper'] = df['vwap'] + (df['volatility'] * 1.5)
            df['vwap_lower'] = df['vwap'] - (df['volatility'] * 1.5)

        # Clean up temporary columns
        df = df.drop(['tr1', 'tr2', 'tr3', 'true_range'], axis=1)

        logger.info("Calculated volatility indicators (ATR) and VWAP bands")
        return df

    def calculate_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average Directional Index (ADX)."""
        df = df.copy()

        # Calculate +DM and -DM
        df['high_diff'] = df['high'] - df['high'].shift(1)
        df['low_diff'] = df['low'].shift(1) - df['low']

        df['+DM'] = np.where((df['high_diff'] > df['low_diff']) & (df['high_diff'] > 0), df['high_diff'], 0)
        df['-DM'] = np.where((df['low_diff'] > df['high_diff']) & (df['low_diff'] > 0), df['low_diff'], 0)

        # Smooth +DM and -DM
        df['+DM_smooth'] = df['+DM'].rolling(14).mean()
        df['-DM_smooth'] = df['-DM'].rolling(14).mean()

        # +DI and -DI
        df['+DI'] = 100 * df['+DM_smooth'] / df['volatility']
        df['-DI'] = 100 * df['-DM_smooth'] / df['volatility']

        # DX and ADX
        df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
        df['adx'] = df['DX'].rolling(14).mean()

        # Clean up temporary columns
        temp_cols = ['high_diff', 'low_diff', '+DM', '-DM', '+DM_smooth', '-DM_smooth', '+DI', '-DI', 'DX']
        df = df.drop(columns=[col for col in temp_cols if col in df.columns])

        logger.info("Calculated ADX")
        return df

    def classify_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify market regime using realistic thresholds based on actual data ranges.
        """
        df = df.copy()

        # Initialize
        df['regime'] = 'neutral'
        df['regime_state'] = 5

        # Calculate percentile-based thresholds (adaptive to data)
        velocity_75th = df['velocity'].quantile(0.75)
        velocity_25th = df['velocity'].quantile(0.25)
        drift_vel_75th = df['drift_velocity'].quantile(0.75)
        drift_vel_25th = df['drift_velocity'].quantile(0.25)

        logger.info(f"Velocity range: {velocity_25th:.2f} to {velocity_75th:.2f}")
        logger.info(f"Drift velocity range: {drift_vel_25th:.4f} to {drift_vel_75th:.4f}")

        # 1. LAUNCH (strong momentum + expanding drift)
        # Top 25% of velocity + positive drift expansion
        launch_mask = (
            (df['velocity'] > velocity_75th) &  # Top quartile momentum
            (df['drift_velocity'] > 0) &  # Drift expanding (any positive value)
            (df['adx'] > 15)  # Some trend (lowered threshold)
        )
        df.loc[launch_mask, 'regime'] = 'launch'
        df.loc[launch_mask, 'regime_state'] = 10

        # 2. BEAR PEG (strong negative momentum + contracting drift)
        bear_peg_mask = (
            (df['velocity'] < velocity_25th) &  # Bottom quartile momentum
            (df['drift_velocity'] < 0) &  # Drift contracting
            (df['adx'] > 15) &
            (df['regime_state'] != 10)  # Not already Launch
        )
        df.loc[bear_peg_mask, 'regime'] = 'bear_peg'
        df.loc[bear_peg_mask, 'regime_state'] = 4

        # 3. SNIPER (price near VWAP bands)
        if 'vwap_upper' in df.columns and 'vwap_lower' in df.columns:
            # Within 1% of VWAP bands
            near_lower = (df['close'] - df['vwap_lower']).abs() / df['close'] < 0.01
            near_upper = (df['vwap_upper'] - df['close']).abs() / df['close'] < 0.01

            sniper_mask = (
                (near_lower | near_upper) &
                (df['volatility_pct'] > 1.0) &
                (~launch_mask) &
                (~bear_peg_mask)
            )
            df.loc[sniper_mask, 'regime'] = 'sniper'
            df.loc[sniper_mask, 'regime_state'] = 13

        # 4. BULL PEG (moderate positive momentum)
        bull_peg_mask = (
            (df['velocity'] > 0) &
            (df['drift_strength'] > 0) &
            (df['regime_state'] == 5)  # Still neutral
        )
        df.loc[bull_peg_mask, 'regime'] = 'bull_peg'
        df.loc[bull_peg_mask, 'regime_state'] = 3

        regime_counts = df['regime'].value_counts().to_dict()
        logger.info(f"Regime distribution: {regime_counts}")

        return df

    def run_data_pipeline(self, symbol: str, months: int = 6, timeframe_minutes: int = 5,
                         use_csv: bool = False, csv_path: Optional[str] = None) -> pd.DataFrame:
        """
        Complete data ingestion and processing pipeline.
        """
        logger.info(f"Starting data pipeline for {symbol}")

        # Data Ingestion
        if use_csv and csv_path:
            df = self.load_csv_data(csv_path, symbol, timeframe_minutes)
        else:
            logger.error("API ingestion not implemented - use CSV")
            return pd.DataFrame()

        if df.empty:
            logger.error("No data available for processing")
            return df

        # Data Cleaning
        df = self.clean_data(df)

        # Store raw data
        self.store_historical_data(df)

        # Calculate Indicators (Vectorized)
        df = self.calculate_drift_indicators_vectorized(df)
        df = self.calculate_velocity_indicators(df)
        df = self.calculate_volatility_indicators(df)
        df = self.calculate_adx(df)

        # Classify Regimes
        df = self.classify_market_regime(df)

        # Store processed data
        self.store_processed_data(df)

        logger.info(f"Data pipeline completed for {symbol}. Processed {len(df)} bars.")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the data."""
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

        # Forward fill missing values (limited)
        df = df.ffill(limit=5)

        # Drop remaining NaN values
        df = df.dropna()

        logger.info(f"Data cleaned: {len(df)} valid bars remaining")
        return df

    def store_processed_data(self, df: pd.DataFrame):
        """Store processed data with indicators in database."""
        try:
            indicator_cols = [
                'symbol', 'timeframe', 'timestamp', 'datetime_utc', 'close',
                'ema_8', 'ema_24', 'velocity', 'drift_velocity', 'drift_spread',
                'fast_line', 'slow_line', 'volatility', 'vwap', 'vwap_upper', 'vwap_lower',
                'adx', 'regime', 'regime_state'
            ]

            available_cols = [col for col in indicator_cols if col in df.columns]
            indicator_df = df[available_cols].copy()
            # Preserve symbol/timeframe for DB checks
            symbol = df.iloc[0]['symbol'] if 'symbol' in df.columns else None
            timeframe = df.iloc[0]['timeframe'] if 'timeframe' in df.columns else None

            # Convert datetime to text before writing
            if 'datetime_utc' in indicator_df.columns:
                indicator_df['datetime_utc'] = indicator_df['datetime_utc'].astype(str)

            # Determine which columns actually exist in the DB table
            try:
                cur = self.conn.cursor()
                cur.execute("PRAGMA table_info(indicators)")
                db_cols = [r[1] for r in cur.fetchall()]
            except Exception:
                db_cols = []

            # Keep only columns that exist in the DB to avoid OperationalError
            final_cols = [c for c in indicator_df.columns if c in db_cols]
            if not final_cols:
                logger.info('No overlapping columns to store in indicators table; skipping write')
                return

            indicator_df = indicator_df[final_cols].copy()

            # Avoid duplicate inserts by checking existing timestamps if possible
            existing = set()
            if symbol and timeframe and 'timestamp' in indicator_df.columns:
                try:
                    cur.execute("SELECT timestamp FROM indicators WHERE symbol=? AND timeframe=?", (symbol, timeframe))
                    existing = {r[0] for r in cur.fetchall()}
                except Exception:
                    existing = set()

            if 'timestamp' in indicator_df.columns and existing:
                new_ind = indicator_df[~indicator_df['timestamp'].isin(existing)].copy()
            else:
                new_ind = indicator_df

            if new_ind.empty:
                logger.info("No new indicator rows to store (all timestamps already present or no timestamps)")
                return

            new_ind.to_sql('indicators', self.conn, if_exists='append', index=False)
            logger.info(f"Stored {len(new_ind)} processed bars with indicators")
        except Exception as e:
            logger.exception(f"Error storing processed data: {e}")

    def get_historical_data(self, symbol: str, timeframe: str,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """Retrieve historical data with indicators from database."""
        query = f"""
        SELECT h.*, i.ema_8, i.ema_24, i.velocity, i.drift_velocity, i.drift_spread,
               i.fast_line, i.slow_line, i.volatility, i.vwap, i.vwap_upper, i.vwap_lower,
               i.adx, i.regime, i.regime_state
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


class CommanderStrategy:
    """
    0DTE Commander Strategy execution logic.
    Implements Hunter/Manager/Guard roles from the Field Manual.
    """

    def __init__(self, risk_per_trade: float = 0.005, max_daily_loss: float = 0.02):
        """
        Initialize the Commander Strategy.

        Args:
            risk_per_trade: Maximum risk per trade (0.5% default)
            max_daily_loss: Maximum daily loss limit (2% default)
        """
        self.risk_per_trade = risk_per_trade
        self.max_daily_loss = max_daily_loss

    def generate_entry_signals(self, df: pd.DataFrame, vix_data: pd.DataFrame = None,
                               volume_multiplier: float = 2.0,
                               adx_cutoff: Optional[float] = None,
                               mvv_scale: float = 1.0) -> pd.DataFrame:
        """Generate entry signals with realistic thresholds."""
        df = df.copy()

        # Initialize
        df['entry_signal'] = 0
        df['signal_type'] = None
        df['signal_strength'] = 0.0

        # Calculate adaptive MVV (scaled to data)
        velocity_std = df['velocity'].std()
        df['mvv_threshold'] = df.apply(
            lambda row: self.calculate_mvv_adaptive(row['datetime_utc'], velocity_std),
            axis=1
        )
        # Apply user-specified MVV scaling for parameter sweeps
        df['mvv_threshold'] = df['mvv_threshold'] * float(mvv_scale)

        # LAUNCH SIGNALS (state 10) - Strong momentum breakout
        # Use absolute velocity threshold based on data
        velocity_75th = df['velocity'].quantile(0.75)

        # ADX threshold used in signal masks (default 15)
        adx_thresh = float(adx_cutoff) if adx_cutoff is not None else 15.0

        launch_mask = (
            (df['regime_state'] == 10) &
            (df['velocity'] > velocity_75th) &  # Top quartile
            (df['drift_velocity'] > 0) &  # Expanding
            (df['adx'] > adx_thresh)
        )
        df.loc[launch_mask, 'entry_signal'] = 1
        df.loc[launch_mask, 'signal_type'] = 'launch'
        df.loc[launch_mask, 'signal_strength'] = df.loc[launch_mask, 'velocity'] / velocity_75th

        # SNIPER BUY (state 13) - VWAP lower band bounce
        sniper_buy_mask = (
            (df['regime_state'] == 13) &
            (df['close'] < df['vwap_lower']) &
            (df['close'].shift(1) >= df['vwap_lower'].shift(1)) &  # Just crossed below
            (df['adx'] > adx_thresh)
        )
        df.loc[sniper_buy_mask, 'entry_signal'] = 1
        df.loc[sniper_buy_mask, 'signal_type'] = 'sniper_buy'
        df.loc[sniper_buy_mask, 'signal_strength'] = (df.loc[sniper_buy_mask, 'vwap_lower'] - df.loc[sniper_buy_mask, 'close']) / df.loc[sniper_buy_mask, 'close']

        # BULL PEG (state 3) - Moderate uptrend
        bull_peg_mask = (
            (df['regime_state'] == 3) &
            (df['adx'] > adx_thresh) &
            (df['velocity'] > 0)
        )
        df.loc[bull_peg_mask, 'entry_signal'] = 1
        df.loc[bull_peg_mask, 'signal_type'] = 'bull_peg'
        df.loc[bull_peg_mask, 'signal_strength'] = df.loc[bull_peg_mask, 'adx'] / 40.0

        # BEAR PEG (state 4) - Moderate downtrend (SHORT)
        bear_peg_mask = (
            (df['regime_state'] == 4) &
            (df['adx'] > 15) &
            (df['velocity'] < 0)
        )
        df.loc[bear_peg_mask, 'entry_signal'] = -1  # SHORT
        df.loc[bear_peg_mask, 'signal_type'] = 'bear_peg'
        df.loc[bear_peg_mask, 'signal_strength'] = abs(df.loc[bear_peg_mask, 'velocity']) / velocity_75th

        logger.info(f"Generated {df['entry_signal'].abs().sum()} entry signals")
        logger.info(f"  Launch: {(df['signal_type'] == 'launch').sum()}")
        logger.info(f"  Sniper: {(df['signal_type'] == 'sniper_buy').sum()}")
        logger.info(f"  Bull PEG: {(df['signal_type'] == 'bull_peg').sum()}")
        logger.info(f"  Bear PEG: {(df['signal_type'] == 'bear_peg').sum()}")

        # --- VETO LAYER IMPLEMENTATION ---
        # 1) VIX Regime Filter (use vix_data if provided, otherwise use ATR proxy)
        try:
            if vix_data is not None and not vix_data.empty:
                vix_df = vix_data.copy()
                # Normalize datetime column
                if 'timestamp' in vix_df.columns:
                    vix_df['datetime_utc'] = pd.to_datetime(vix_df['timestamp'], unit='s', utc=True)
                elif 'datetime_utc' in vix_df.columns:
                    vix_df['datetime_utc'] = pd.to_datetime(vix_df['datetime_utc'], utc=True)

                # find a suitable vix price column
                vix_col = None
                for c in ['vix', 'close', 'price', 'last']:
                    if c in vix_df.columns:
                        vix_col = c
                        break

                if vix_col is None:
                    vix_series = None
                else:
                    vix_series = pd.merge_asof(
                        df[['datetime_utc']].sort_values('datetime_utc'),
                        vix_df.sort_values('datetime_utc')[[ 'datetime_utc', vix_col]],
                        left_on='datetime_utc', right_on='datetime_utc', direction='backward')
                    vix_series = vix_series[vix_col].values
            else:
                vix_series = None

            # ATR proxy if no VIX series
            if vix_series is None:
                proxy = df['volatility_pct'].rolling(20).mean()
                p25 = proxy.quantile(0.25)
                p75 = proxy.quantile(0.75)
                df['vix_regime'] = pd.np.where(proxy > p75, 'high', pd.np.where(proxy < p25, 'low', 'medium'))
            else:
                # vix_series is an array aligned to df
                vix_s = pd.Series(vix_series, index=df.index)
                df['vix_value'] = vix_s
                df['vix_regime'] = 'medium'
                df.loc[df['vix_value'] > 25, 'vix_regime'] = 'high'
                df.loc[df['vix_value'] < 15, 'vix_regime'] = 'low'
        except Exception:
            df['vix_regime'] = 'medium'

        # 2) Volume Climax Filter
        df['vol_ma20'] = df['volume'].rolling(20).mean()
        df['volume_spike'] = df['volume'] > (df['vol_ma20'] * float(volume_multiplier))
        df['volume_climax'] = False
        if 'drift_velocity' in df.columns and 'mvv_threshold' in df.columns:
            df['volume_climax'] = df['volume_spike'] & (df['drift_velocity'] < df['mvv_threshold'])

        # 3) Time-of-Day Filter (ET)
        try:
            dt_et = df['datetime_utc'].dt.tz_convert('America/New_York')
        except Exception:
            dt_et = pd.to_datetime(df['datetime_utc']).dt.tz_localize('UTC').dt.tz_convert('America/New_York')
        df['et_hour'] = dt_et.dt.hour
        df['et_min'] = dt_et.dt.minute
        df['is_open_avoid'] = ((df['et_hour'] == 9) & (df['et_min'] >= 30) & (df['et_min'] < 45))
        df['is_close_avoid'] = ((df['et_hour'] == 15) & (df['et_min'] >= 45))
        df['avoid_tod'] = df['is_open_avoid'] | df['is_close_avoid']

        # 4) ADX Divergence Filter
        df['adx_falling'] = False
        if 'adx' in df.columns:
            df['adx_falling'] = df['adx'] < df['adx'].shift(1)

        # Also consider ADX absolute cutoff as additional veto (if provided)
        if adx_cutoff is not None and 'adx' in df.columns:
            df['adx_below_cutoff'] = df['adx'] < float(adx_cutoff)
        else:
            df['adx_below_cutoff'] = False

        # 5) VIX Crush Detection (requires vix_data)
        df['vix_crush'] = False
        try:
            if vix_data is not None and not vix_data.empty:
                vix_df = vix_data.copy()
                if 'timestamp' in vix_df.columns:
                    vix_df['datetime_utc'] = pd.to_datetime(vix_df['timestamp'], unit='s', utc=True)
                elif 'datetime_utc' in vix_df.columns:
                    vix_df['datetime_utc'] = pd.to_datetime(vix_df['datetime_utc'], utc=True)

                vix_df['date_et'] = vix_df['datetime_utc'].dt.tz_convert('America/New_York').dt.date
                # attempt to find a high/low column
                found = None
                for c in ['high', 'vix_high', 'vix_high_price', 'high_price']:
                    if c in vix_df.columns:
                        found = c
                        break
                # fallback to close as proxy for high/low
                if found is None and 'close' in vix_df.columns:
                    vix_df['v_temp'] = vix_df['close']
                    found = 'v_temp'

                if found is not None:
                    daily = vix_df.groupby('date_et')[found].agg(['max','min']).reset_index()
                    daily['crush_pct'] = (daily['max'] - daily['min']) / daily['max']
                    daily['vix_crush_flag'] = daily['crush_pct'] > 0.10
                    # map back to df by date
                    df['date_et'] = dt_et.dt.date
                    df = df.merge(daily[['date_et','vix_crush_flag']], on='date_et', how='left')
                    df['vix_crush'] = df['vix_crush_flag'].fillna(False)
                    df = df.drop(columns=['date_et','vix_crush_flag'])
        except Exception:
            df['vix_crush'] = False

        # 6) Internals Veto (Unified Internals)
        # Ensure we respect the internals veto if calculated
        has_internals_veto = 'veto_internals' in df.columns

        # Apply veto rules
        initial_signals = df['entry_signal'].copy()

        # Skip all if volume climax
        df.loc[df['volume_climax'] == True, ['entry_signal','signal_type']] = (0, None)

        # Skip all in avoid TOD
        df.loc[df['avoid_tod'] == True, ['entry_signal','signal_type']] = (0, None)

        # Skip all if VIX crushed
        df.loc[df['vix_crush'] == True, ['entry_signal','signal_type']] = (0, None)
        
        # Skip all if Internals Veto is active
        if has_internals_veto:
            df.loc[df['veto_internals'] == True, ['entry_signal','signal_type']] = (0, None)

        # VIX regime enforcement
        # Launch -> only high
        df.loc[(df['signal_type'] == 'launch') & (df['vix_regime'] != 'high'), ['entry_signal','signal_type']] = (0, None)
        # Bull peg -> only medium
        df.loc[(df['signal_type'] == 'bull_peg') & (df['vix_regime'] != 'medium'), ['entry_signal','signal_type']] = (0, None)
        # Sniper -> medium/high only
        df.loc[(df['signal_type'] == 'sniper_buy') & (df['vix_regime'] == 'low'), ['entry_signal','signal_type']] = (0, None)

        # ADX divergence: skip Launch and Bull PEG when ADX falling or below cutoff
        # Only veto falling ADX if the trend is already weak (< 25)
        df.loc[((df['adx_falling'] == True) & (df['adx'] < 25) | (df['adx_below_cutoff'] == True)) & (df['signal_type'].isin(['launch','bull_peg'])), ['entry_signal','signal_type']] = (0, None)

        # Log veto counts
        vetoed = (initial_signals.abs() - df['entry_signal'].abs()).sum()
        
        # Detailed Veto Breakdown
        if vetoed > 0:
            logger.info(f"Veto layer applied. Total signals vetoed: {int(vetoed)}")
            
            # Calculate which specific conditions were active during valid signals
            signal_mask = initial_signals != 0
            
            veto_stats = {
                'Volume Climax': (signal_mask & df['volume_climax']).sum(),
                'Time of Day (Avoid)': (signal_mask & df['avoid_tod']).sum(),
                'VIX Crush': (signal_mask & df['vix_crush']).sum(),
                'ADX Falling': (signal_mask & df['adx_falling']).sum(),
                'ADX Low (<Cutoff)': (signal_mask & df['adx_below_cutoff']).sum()
            }
            if has_internals_veto:
                veto_stats['Internals Health'] = (signal_mask & df['veto_internals']).sum()
                # Breakdown of internals vetoes
                if 'veto_uvol_dvol' in df.columns:
                    veto_stats['  - UVOL/DVOL'] = (signal_mask & df['veto_uvol_dvol']).sum()
                if 'veto_trin_extreme' in df.columns:
                    veto_stats['  - TRIN'] = (signal_mask & df['veto_trin_extreme']).sum()
                if 'veto_tick_extreme' in df.columns:
                    veto_stats['  - TICK'] = (signal_mask & df['veto_tick_extreme']).sum()
                if 'veto_mag7_concentration' in df.columns:
                    veto_stats['  - Mag7'] = (signal_mask & df['veto_mag7_concentration']).sum()
                
            for reason, count in veto_stats.items():
                if count > 0:
                    logger.info(f"  - {reason}: blocked {count} signals")

        # Clean temporary columns used for veto logic
        temp_cols = ['mvv_threshold','vol_ma20','volume_spike','volume_climax','et_hour','et_min',
                     'is_open_avoid','is_close_avoid','avoid_tod','adx_falling','vix_value','vix_regime']
        for c in temp_cols:
            if c in df.columns:
                try:
                    df.drop(columns=[c], inplace=True)
                except Exception:
                    pass

        return df

    def calculate_mvv_adaptive(self, datetime_utc, velocity_std) -> float:
        """
        MVV scaled to actual data volatility.
        Instead of fixed 3/5/7/10, use multiples of velocity std dev.
        """
        if isinstance(datetime_utc, str):
            dt = pd.to_datetime(datetime_utc)
        else:
            dt = datetime_utc

        et_time = dt.tz_convert('America/New_York') if dt.tz else dt
        hour = et_time.hour
        minute = et_time.minute
        minutes_from_open = (hour - 9) * 60 + (minute - 30)

        # MVV as multiple of velocity standard deviation
        if minutes_from_open < 30:
            return velocity_std * 0.5  # Lower bar early
        elif minutes_from_open < 270:
            return velocity_std * 0.75  # Mid-day
        elif minutes_from_open < 360:
            return velocity_std * 1.0  # Afternoon
        else:
            return velocity_std * 1.5  # Late day (high bar)

    def calculate_mvv(self, datetime_utc) -> float:
        """
        Minimum Viable Velocity - dynamic threshold that rises throughout the day.
        MVV increases to beat theta decay in 0DTE options.
        """
        if isinstance(datetime_utc, str):
            dt = pd.to_datetime(datetime_utc)
        else:
            dt = datetime_utc

        # Convert to ET
        et_time = dt.tz_convert('America/New_York') if dt.tz else dt
        hour = et_time.hour
        minute = et_time.minute

        # Calculate time from market open (9:30 AM)
        minutes_from_open = (hour - 9) * 60 + (minute - 30)

        if minutes_from_open < 30:  # Before 10:00
            return 3.0
        elif minutes_from_open < 270:  # 10:00-14:00
            return 5.0
        elif minutes_from_open < 360:  # 14:00-15:30
            return 7.0
        else:  # After 15:30
            return 10.0

    def simulate_trades(self, df: pd.DataFrame, capital: float = 10000.0) -> List[Dict]:
        """
        Simulate trades with proper point-in-time execution.
        Enter at NEXT bar's open after signal detection.
        """
        trades = []
        active_positions = []

        for idx in range(len(df) - 1):  # Stop 1 bar early
            row = df.iloc[idx]
            next_row = df.iloc[idx + 1]
            current_time = row['datetime_utc']

            # Entry Logic: Signal on current bar, execute on NEXT bar's open
            if row['entry_signal'] != 0 and len(active_positions) < 2:
                # Calculate position size based on ATR risk
                volatility_pct = next_row['volatility'] / next_row['open']
                position_size = int((capital * self.risk_per_trade) / (next_row['open'] * max(volatility_pct, 0.01)))
                position_size = max(1, min(position_size, 10))  # Ensure 1-10 contracts

                trade = {
                    'signal_time': current_time,
                    'entry_time': next_row['datetime_utc'],
                    'entry_price': next_row['open'],
                    'position_size': position_size,
                    'signal_type': row['signal_type'],
                    'direction': 'long' if row['entry_signal'] > 0 else 'short',
                    'regime_state': row['regime_state'],
                    'entry_regime': row['regime'],
                    'exit_time': None,
                    'exit_price': None,
                    'exit_reason': None,
                    'pnl': 0.0,
                    'holding_period': 0
                }
                active_positions.append(trade)

            # Exit Logic: Check conditions on current bar
            positions_to_close = []

            for pos_idx, position in enumerate(active_positions):
                holding_period = (current_time - position['entry_time']).total_seconds() / 60
                exit_reason = None

                # Physics-based exits (Priority)
                if not exit_reason and row.get('is_theta_burn', False):
                    exit_reason = 'theta_burn'
                
                if not exit_reason and row.get('is_climax_vol', False):
                    exit_reason = 'volume_climax'
                
                if not exit_reason and row.get('is_snapback', False):
                    # is_snapback is bearish (drift < -9). Exits LONG positions.
                    if position['direction'] == 'long':
                        exit_reason = 'snapback_against'

                # 1. Time-based exit (15 minutes)
                if not exit_reason and holding_period >= 15:
                    exit_reason = 'time_stop'
                
                # 2. Profit target (1% gain)
                elif not exit_reason and position['direction'] == 'long' and row['close'] >= position['entry_price'] * 1.01:
                    exit_reason = 'profit_target'
                elif not exit_reason and position['direction'] == 'short' and row['close'] <= position['entry_price'] * 0.99:
                    exit_reason = 'profit_target'
                
                # 3. Stop loss (0.5% loss)
                elif not exit_reason and position['direction'] == 'long' and row['close'] <= position['entry_price'] * 0.995:
                    exit_reason = 'stop_loss'
                elif not exit_reason and position['direction'] == 'short' and row['close'] >= position['entry_price'] * 1.005:
                    exit_reason = 'stop_loss'
                
                # REMOVED: Regime change exit
                # This was cutting winners early
                
                if exit_reason:
                    # Exit at NEXT bar's open
                    exit_price = next_row['open']

                    if position['direction'] == 'long':
                        pnl = (exit_price - position['entry_price']) * position['position_size'] * 100
                    else:
                        pnl = (position['entry_price'] - exit_price) * position['position_size'] * 100

                    # Add slippage
                    slippage = position['entry_price'] * 0.0005 * position['position_size'] * 100
                    pnl -= slippage

                    position.update({
                        'exit_time': next_row['datetime_utc'],
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'pnl': pnl,
                        'holding_period': holding_period
                    })

                    trades.append(position)
                    positions_to_close.append(pos_idx)

            # Remove closed positions
            for pos_idx in reversed(positions_to_close):
                active_positions.pop(pos_idx)

        # Close remaining positions at end of data
        if active_positions:
            final_row = df.iloc[-1]
            for position in active_positions:
                exit_price = final_row['close']

                if position['direction'] == 'long':
                    pnl = (exit_price - position['entry_price']) * position['position_size'] * 100
                else:
                    pnl = (position['entry_price'] - exit_price) * position['position_size'] * 100

                position.update({
                    'exit_time': final_row['datetime_utc'],
                    'exit_price': exit_price,
                    'exit_reason': 'end_of_day',
                    'pnl': pnl
                })
                trades.append(position)

        logger.info(f"Simulated {len(trades)} trades")
        return trades

    def calculate_performance_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not trades:
            return {'error': 'No trades to analyze'}

        # Basic trade metrics
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]

        total_trades = len(trades)
        winning_trades_count = len(winning_trades)
        win_rate = winning_trades_count / total_trades if total_trades > 0 else 0

        total_pnl = sum(t['pnl'] for t in trades)
        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Average win/loss
        avg_win = gross_profit / winning_trades_count if winning_trades_count > 0 else 0
        avg_loss = gross_loss / len(losing_trades) if losing_trades else 0
        payoff_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else float('inf')

        # Risk metrics
        pnl_series = [sum(t['pnl'] for t in trades[:i+1]) for i in range(len(trades))]
        peak = 0
        max_drawdown = 0
        for pnl in pnl_series:
            if pnl > peak:
                peak = pnl
            drawdown = peak - pnl
            max_drawdown = max(max_drawdown, drawdown)

        # Sharpe ratio
        if len(trades) > 1:
            returns = [t['pnl'] for t in trades]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0

        # Performance by signal type
        signal_performance = {}
        for signal_type in set(t['signal_type'] for t in trades if t['signal_type']):
            signal_trades = [t for t in trades if t['signal_type'] == signal_type]
            signal_wins = len([t for t in signal_trades if t['pnl'] > 0])
            signal_win_rate = signal_wins / len(signal_trades) if signal_trades else 0
            signal_pnl = sum(t['pnl'] for t in signal_trades)

            signal_performance[signal_type] = {
                'trades': len(signal_trades),
                'win_rate': signal_win_rate,
                'total_pnl': signal_pnl
            }

        # Performance by exit reason
        exit_performance = {}
        for exit_reason in set(t['exit_reason'] for t in trades if t['exit_reason']):
            exit_trades = [t for t in trades if t['exit_reason'] == exit_reason]
            exit_wins = len([t for t in exit_trades if t['pnl'] > 0])
            exit_win_rate = exit_wins / len(exit_trades) if exit_trades else 0
            exit_pnl = sum(t['pnl'] for t in exit_trades)

            exit_performance[exit_reason] = {
                'trades': len(exit_trades),
                'win_rate': exit_win_rate,
                'total_pnl': exit_pnl
            }

        return {
            'summary': {
                'total_trades': total_trades,
                'winning_trades': winning_trades_count,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'payoff_ratio': payoff_ratio,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio
            },
            'by_signal_type': signal_performance,
            'by_exit_reason': exit_performance
        }


def run_commander_system_backtest(db_path: str = 'backtesting_data.db',
                                  symbol: str = 'SPY',
                                  start_date: str = '2024-01-01',
                                  end_date: str = '2024-12-31',
                                  capital: float = 10000.0,
                                  risk_per_trade: float = 0.005) -> Dict:
    """
    Run complete Commander system backtest with all indicators and veto conditions.

    Args:
        db_path: Path to SQLite database with data
        symbol: Primary symbol to backtest
        start_date: Start date for backtest period
        end_date: End date for backtest period
        capital: Starting capital
        risk_per_trade: Risk per trade as fraction of capital

    Returns:
        Dictionary with backtest results and performance metrics
    """
    try:
        # Load historical data
        conn = sqlite3.connect(db_path)
        query = f"""
        SELECT * FROM historical_data
        WHERE symbol = '{symbol}'
        AND datetime_utc >= '{start_date}'
        AND datetime_utc <= '{end_date}'
        ORDER BY datetime_utc
        """
        df = pd.read_sql_query(query, conn, parse_dates=['datetime_utc'])
        df.set_index('datetime_utc', inplace=True)
        conn.close()

        if df.empty:
            return {'error': f'No data found for {symbol} between {start_date} and {end_date}'}

        print(f"Loaded {len(df)} bars for {symbol}")

        # Calculate all indicators
        indicators = VectorizedIndicators(df)
        df_with_indicators = indicators.run_all_calculations()

        print(f"Calculated {len([col for col in df_with_indicators.columns if col not in df.columns])} indicators")

        # Generate signals with veto conditions
        df_with_signals = generate_commander_signals(df_with_indicators)

        # Count signals
        entry_signals = df_with_signals['entry_signal'].value_counts()
        exit_signals = df_with_signals['exit_signal'].value_counts()
        print(f"Entry signals: {dict(entry_signals)}")
        print(f"Exit signals: {dict(exit_signals)}")

        # Run backtest
        strategy = CommanderStrategyV2(risk_per_trade=risk_per_trade)
        trades = strategy.simulate_trades(df_with_signals, capital=capital)

        print(f"Simulated {len(trades)} trades")

        # Calculate performance metrics
        metrics = strategy.calculate_performance_metrics(trades)

        # Add signal analysis
        signal_analysis = analyze_signals(df_with_signals)

        return {
            'backtest_info': {
                'symbol': symbol,
                'period': f'{start_date} to {end_date}',
                'total_bars': len(df),
                'capital': capital,
                'risk_per_trade': risk_per_trade
            },
            'data_quality': {
                'bars_with_indicators': len(df_with_indicators),
                'bars_with_signals': len(df_with_signals),
                'entry_signals': dict(entry_signals),
                'exit_signals': dict(exit_signals)
            },
            'trades': trades,
            'performance': metrics,
            'signal_analysis': signal_analysis
        }

    except Exception as e:
        return {'error': f'Backtest failed: {str(e)}'}


def generate_commander_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate entry and exit signals using Commander system logic with veto conditions.

    Args:
        df: DataFrame with all indicators calculated

    Returns:
        DataFrame with entry_signal, exit_signal, signal_type, signal_strength columns
    """
    # Initialize signal columns
    df = df.copy()
    df['entry_signal'] = 0
    df['exit_signal'] = 0
    df['signal_type'] = ''
    df['signal_strength'] = 0.0

    # Veto conditions: IV Crush (always applied)
    veto_iv_crush = df.get('veto_iv_crush', pd.Series([False] * len(df)))

    # Internals veto: Applied if internals data available
    veto_internals = df.get('veto_internals', pd.Series([False] * len(df)))

    # Combine all veto conditions: ANY veto blocks all signals
    all_veto_conditions = veto_iv_crush | veto_internals

    # Market bias filter (prefer bias-aligned signals)
    market_bias_long = df['market_bias'] > 0
    market_bias_short = df['market_bias'] < 0

    # MVV gating for velocity signals (Commander requirement)
    velocity_allowed_regimes = [5, 10]  # Trend and Launch regimes

    df['velocity_ok_long'] = (
        df['normalized_velocity'] > df['mvv_threshold']
    )
    df['velocity_ok_short'] = (
        df['normalized_velocity'] < -df['mvv_threshold']
    )

    # Entry conditions for LONG positions
    long_velocity_signal = (
        (df['drift_velocity'] > df['drift_velocity'].quantile(0.8)) &
        (df['normalized_velocity'] > 0) &
        (df['atr'] > df['atr'].rolling(20).mean()) &
        df['velocity_ok_long'] &  # MVV gating
        df['regime_state'].isin(velocity_allowed_regimes)  # Regime gating
    )

    long_theta_signal = (
        (df['vega_factor'] < df['vega_factor'].quantile(0.2)) &
        (df['is_theta_burn']) &
        (df['drift_velocity'] > df['drift_velocity'].rolling(10).mean())
    )

    long_volume_signal = (
        (df['volume_ratio'] > df['volume_ratio'].quantile(0.9)) &
        (df['is_aggressive_buy']) &
        (df['avg_volume'] > df['avg_volume'].rolling(20).mean())
    )

    # Combined long entry (any strong signal, bias-aligned preferred)
    long_entry = (
        (long_velocity_signal | long_theta_signal | long_volume_signal) &
        ~all_veto_conditions &  # Use combined veto conditions
        (market_bias_long | (df['market_bias'] == 0))  # Allow neutral bias
    )

    # Entry conditions for SHORT positions
    short_velocity_signal = (
        (df['drift_velocity'] < df['drift_velocity'].quantile(0.2)) &
        (df['normalized_velocity'] < 0) &
        (df['atr'] < df['atr'].rolling(20).mean()) &
        df['velocity_ok_short'] &  # MVV gating
        df['regime_state'].isin(velocity_allowed_regimes)  # Regime gating
    )

    short_theta_signal = (
        (df['vega_factor'] > df['vega_factor'].quantile(0.8)) &
        (df['is_theta_burn']) &
        (df['drift_velocity'] < df['drift_velocity'].rolling(10).mean())
    )

    short_volume_signal = (
        (df['volume_ratio'] > df['volume_ratio'].quantile(0.9)) &
        (df['is_aggressive_sell']) &
        (df['avg_volume'] < df['avg_volume'].rolling(20).mean())
    )

    # Combined short entry
    short_entry = (
        (short_velocity_signal | short_theta_signal | short_volume_signal) &
        ~all_veto_conditions &  # Use combined veto conditions
        (market_bias_short | (df['market_bias'] == 0))
    )

    # Enhanced exit conditions (include theta burn, volume climax, snapback)
    long_exit = (
        short_entry |  # Exit long on short signals
        df['is_theta_burn'] |  # Exit on theta burn
        df['is_climax_vol'] |  # Exit on volume climax
        (df['is_snapback'] & (df['drift_velocity'] < -5))  # Exit on snapback against position
    )

    short_exit = (
        long_entry |  # Exit short on long signals
        df['is_theta_burn'] |  # Exit on theta burn
        df['is_climax_vol'] |  # Exit on volume climax
        (df['is_snapback'] & (df['drift_velocity'] > 5))  # Exit on snapback against position
    )

    # Apply signals
    df.loc[long_entry, 'entry_signal'] = 1
    df.loc[short_entry, 'entry_signal'] = -1
    df.loc[long_exit, 'exit_signal'] = -1  # Exit long positions
    df.loc[short_exit, 'exit_signal'] = 1   # Exit short positions

    # Signal type classification
    df.loc[long_velocity_signal & long_entry, 'signal_type'] = 'velocity_long'
    df.loc[long_theta_signal & long_entry, 'signal_type'] = 'theta_long'
    df.loc[long_volume_signal & long_entry, 'signal_type'] = 'volume_long'
    df.loc[short_velocity_signal & short_entry, 'signal_type'] = 'velocity_short'
    df.loc[short_theta_signal & short_entry, 'signal_type'] = 'theta_short'
    df.loc[short_volume_signal & short_entry, 'signal_type'] = 'volume_short'

    # Signal strength (combination of multiple confirming indicators)
    long_strength = (
        (df['drift_velocity'] - df['drift_velocity'].rolling(20).mean()) / df['drift_velocity'].rolling(20).std() +
        (df['normalized_velocity'] - df['normalized_velocity'].rolling(20).mean()) / df['normalized_velocity'].rolling(20).std() +
        (df['atr'] - df['atr'].rolling(20).mean()) / df['atr'].rolling(20).std()
    ) / 3

    short_strength = (
        -(df['drift_velocity'] - df['drift_velocity'].rolling(20).mean()) / df['drift_velocity'].rolling(20).std() +
        -(df['normalized_velocity'] - df['normalized_velocity'].rolling(20).mean()) / df['normalized_velocity'].rolling(20).std() +
        -(df['atr'] - df['atr'].rolling(20).mean()) / df['atr'].rolling(20).std()
    ) / 3

    df.loc[long_entry, 'signal_strength'] = long_strength[long_entry]
    df.loc[short_entry, 'signal_strength'] = short_strength[short_entry]

    return df


def analyze_signals(df: pd.DataFrame) -> Dict:
    """
    Analyze signal effectiveness and characteristics.
    """
    signals = df[df['entry_signal'] != 0].copy()

    if signals.empty:
        return {'error': 'No signals to analyze'}

    # Signal distribution by type
    signal_types = signals['signal_type'].value_counts()

    # Signal strength analysis
    strength_stats = signals['signal_strength'].describe()

    # Veto impact
    veto_signals = signals[signals['veto_iv_crush']]

    # Regime distribution
    regime_signals = signals['regime_state'].value_counts()

    # Market bias alignment
    bias_aligned = signals[
        ((signals['entry_signal'] == 1) & (signals['market_bias'] >= 0)) |
        ((signals['entry_signal'] == -1) & (signals['market_bias'] <= 0))
    ]

    return {
        'signal_types': dict(signal_types),
        'signal_strength': {
            'mean': strength_stats['mean'],
            'std': strength_stats['std'],
            'min': strength_stats['min'],
            'max': strength_stats['max']
        },
        'veto_impact': {
            'total_signals': len(signals),
            'veto_blocked': len(veto_signals),
            'veto_rate': len(veto_signals) / len(signals) if len(signals) > 0 else 0
        },
        'regime_distribution': dict(regime_signals),
        'bias_alignment': {
            'aligned_signals': len(bias_aligned),
            'alignment_rate': len(bias_aligned) / len(signals) if len(signals) > 0 else 0
        }
    }


class CommanderStrategyV2:
    """
    Enhanced Commander Strategy using full system indicators.
    Incorporates veto conditions, market bias, and unified internals.
    """

    def __init__(self, risk_per_trade: float = 0.005, max_daily_loss: float = 0.02):
        self.risk_per_trade = risk_per_trade
        self.max_daily_loss = max_daily_loss

    def simulate_trades(self, df: pd.DataFrame, capital: float = 10000.0) -> List[Dict]:
        """
        Simulate trades based on entry/exit signals with position sizing.
        """
        trades = []
        position = 0
        entry_price = 0
        entry_time = None
        daily_pnl = 0
        daily_start_capital = capital

        for idx, row in df.iterrows():
            current_price = row['close']
            current_time = idx

            # Check for new day (reset daily loss limit)
            if entry_time and current_time.date() != entry_time.date():
                daily_pnl = 0
                daily_start_capital = capital

            # Entry signals
            if position == 0 and row['entry_signal'] != 0:
                # Calculate position size based on risk
                risk_amount = capital * self.risk_per_trade
                stop_distance = row['atr'] * 2  # 2 ATR stop
                if stop_distance > 0:
                    shares = int(risk_amount / stop_distance)
                    if shares > 0:
                        position = row['entry_signal']  # +1 for long, -1 for short
                        entry_price = current_price
                        entry_time = current_time

                        # Adjust for short positions
                        if position == -1:
                            entry_price = current_price

                        trades.append({
                            'entry_time': entry_time,
                            'entry_price': entry_price,
                            'position': position,
                            'shares': shares,
                            'signal_type': row['signal_type'],
                            'signal_strength': row['signal_strength'],
                            'regime_state': row['regime_state'],
                            'veto_active': row['veto_iv_crush'],
                            'market_bias': row['market_bias']
                        })

            # Exit signals
            elif position != 0 and row['exit_signal'] != 0:
                exit_price = current_price
                exit_time = current_time

                # Calculate P&L
                if position == 1:  # Long position
                    pnl = (exit_price - entry_price) * abs(position) * trades[-1]['shares']
                else:  # Short position
                    pnl = (entry_price - exit_price) * abs(position) * trades[-1]['shares']

                # Update trade record
                trades[-1].update({
                    'exit_time': exit_time,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'exit_reason': 'signal' if row['exit_signal'] != 0 else 'manual',
                    'holding_period': (exit_time - entry_time).total_seconds() / 60  # minutes
                })

                # Update capital
                capital += pnl
                position = 0
                entry_time = None

        # Close any open positions at the end
        if position != 0 and len(trades) > 0:
            final_price = df.iloc[-1]['close']
            if position == 1:
                pnl = (final_price - entry_price) * trades[-1]['shares']
            else:
                pnl = (entry_price - final_price) * trades[-1]['shares']

            trades[-1].update({
                'exit_time': df.index[-1],
                'exit_price': final_price,
                'pnl': pnl,
                'exit_reason': 'end_of_data',
                'holding_period': (df.index[-1] - entry_time).total_seconds() / 60
            })
            capital += pnl

        return trades

    def calculate_performance_metrics(self, trades: List[Dict]) -> Dict:
        """
        Calculate comprehensive performance metrics.
        """
        if not trades:
            return {'error': 'No trades to analyze'}

        # Basic metrics
        total_trades = len(trades)
        completed_trades = [t for t in trades if 'pnl' in t]

        if not completed_trades:
            return {'error': 'No completed trades'}

        winning_trades = [t for t in completed_trades if t['pnl'] > 0]
        losing_trades = [t for t in completed_trades if t['pnl'] <= 0]

        winning_trades_count = len(winning_trades)
        losing_trades_count = len(losing_trades)

        win_rate = winning_trades_count / len(completed_trades) if completed_trades else 0

        total_pnl = sum(t['pnl'] for t in completed_trades)
        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = sum(t['pnl'] for t in losing_trades)

        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')

        avg_win = gross_profit / winning_trades_count if winning_trades_count > 0 else 0
        avg_loss = gross_loss / losing_trades_count if losing_trades_count > 0 else 0
        payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        # Drawdown calculation
        cumulative_pnl = np.cumsum([t['pnl'] for t in completed_trades])
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdowns = running_max - cumulative_pnl
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

        # Sharpe ratio (simplified, assuming daily returns)
        returns = [t['pnl'] for t in completed_trades]
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0

        # Performance by signal type
        signal_performance = {}
        for trade in completed_trades:
            signal_type = trade.get('signal_type', 'unknown')
            if signal_type not in signal_performance:
                signal_performance[signal_type] = {'count': 0, 'wins': 0, 'pnl': 0}
            signal_performance[signal_type]['count'] += 1
            if trade['pnl'] > 0:
                signal_performance[signal_type]['wins'] += 1
            signal_performance[signal_type]['pnl'] += trade['pnl']

        # Performance by exit reason
        exit_performance = {}
        for trade in completed_trades:
            exit_reason = trade.get('exit_reason', 'unknown')
            if exit_reason not in exit_performance:
                exit_performance[exit_reason] = {'count': 0, 'wins': 0, 'pnl': 0}
            exit_performance[exit_reason]['count'] += 1
            if trade['pnl'] > 0:
                exit_performance[exit_reason]['wins'] += 1
            exit_performance[exit_reason]['pnl'] += trade['pnl']

        return {
            'summary': {
                'total_trades': total_trades,
                'completed_trades': len(completed_trades),
                'winning_trades': winning_trades_count,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'payoff_ratio': payoff_ratio,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio
            },
            'by_signal_type': signal_performance,
            'by_exit_reason': exit_performance
        }


# Example usage
if __name__ == '__main__':
    print("=" * 70)
    print("0DTE COMMANDER SYSTEM BACKTESTING ENGINE")
    print("=" * 70)
    print("\nInitialized successfully!")
    print("\nTo run a complete Commander system backtest:")
    print("  from BacktestingEngine import run_commander_system_backtest")
    print("  ")
    print("  # Run backtest with default parameters")
    print("  results = run_commander_system_backtest(")
    print("      db_path='market_data.db',")
    print("      symbol='SPY',")
    print("      start_date='2024-01-01',")
    print("      end_date='2024-12-31',")
    print("      capital=10000.0,")
    print("      risk_per_trade=0.005")
    print("  )")
    print("  ")
    print("  # Access results")
    print("  print('Performance Summary:')")
    print("  print(results['performance']['summary'])")
    print("  print('\\nSignal Analysis:')")
    print("  print(results['signal_analysis'])")
    print("=" * 70)