"""
Mock Internals Data Generator for Testing
==========================================

Generates synthetic but realistic Internals data (UVOL/DVOL/TRIN/TICK/Mag7)
for testing the Commander system without requiring Schwab API authentication.

Author: AI Assistant
Date: February 16, 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockInternalsGenerator:
    """
    Generates realistic mock Internals data for testing.
    """
    
    def __init__(self, db_path: str = 'backtesting_data.db'):
        """Initialize mock generator."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_table()
    
    def _create_table(self):
        """Create internals data table if not exists."""
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
        self.conn.commit()
    
    def generate_internals_data(self, 
                               start_date: datetime,
                               end_date: datetime,
                               interval_minutes: int = 5) -> dict:
        """
        Generate realistic mock Internals data.
        
        Args:
            start_date: Start datetime
            end_date: End datetime
            interval_minutes: Data interval in minutes
            
        Returns:
            Dictionary with DataFrames for each symbol
        """
        
        # Create timestamp range
        timestamps = pd.date_range(start=start_date, end=end_date, freq=f'{interval_minutes}min')
        
        data = {}
        
        for symbol, config in self._get_symbol_config().items():
            logger.info(f"Generating mock data for {symbol}")
            
            base_value = config['base']
            volatility = config['volatility']
            mean_reversion = config['mean_reversion']
            
            # Generate price path (geometric brownian motion with mean reversion)
            prices = []
            current_price = base_value
            
            np.random.seed(42)  # Reproducible
            
            for i, ts in enumerate(timestamps):
                # Random walk with mean reversion
                noise = np.random.normal(0, volatility)
                drift = mean_reversion * (base_value - current_price) / base_value
                current_price = current_price * (1 + drift + noise)
                
                # Add intraday pattern (business hours bias)
                hour = ts.hour
                if 9 <= hour < 16:  # Market hours
                    intraday_factor = 1.0 + 0.05 * np.sin((hour - 9) * np.pi / 7)
                else:
                    intraday_factor = 1.0
                
                current_price = max(current_price * intraday_factor, 0.1)  # Prevent negative
                prices.append(current_price)
            
            # Create OHLCV candles
            df = pd.DataFrame({'timestamp': timestamps, 'close': prices})
            
            # Generate OHLC from closes (simplified)
            df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
            df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.01, len(df)))
            df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.01, len(df)))
            df['volume'] = np.random.randint(1000000, 10000000, len(df))
            
            # Convert timestamp to Unix seconds
            df['timestamp'] = df['timestamp'].astype(np.int64) // 10**9
            df['datetime_utc'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
            
            data[symbol] = df[['timestamp', 'datetime_utc', 'open', 'high', 'low', 'close', 'volume']]
        
        return data
    
    def _get_symbol_config(self) -> dict:
        """Get configuration for each symbol."""
        return {
            '$UVOL': {
                'description': 'Up Volume',
                'base': 500000000,  # 500M baseline
                'volatility': 0.02,
                'mean_reversion': 0.1
            },
            '$DVOL': {
                'description': 'Down Volume',
                'base': 450000000,  # Slightly lower baseline
                'volatility': 0.02,
                'mean_reversion': 0.1
            },
            '$TRIN': {
                'description': 'Arms Index',
                'base': 1.0,  # Mean of 1.0
                'volatility': 0.03,
                'mean_reversion': 0.2
            },
            '$TICK': {
                'description': 'NYSE Tick',
                'base': 100,  # Centered around 100
                'volatility': 50,
                'mean_reversion': 0.15
            },
            '^MAG7': {
                'description': 'Magnificent 7 Index',
                'base': 4500,  # Typical level
                'volatility': 0.015,
                'mean_reversion': 0.05
            }
        }
    
    def store_internals_data(self, data: dict, timeframe: str = '5m'):
        """Store generated mock data in database."""
        total_rows = 0
        
        for symbol, df in data.items():
            try:
                for _, row in df.iterrows():
                    self.conn.execute('''
                    INSERT OR REPLACE INTO internals_data
                    (symbol, timeframe, timestamp, datetime_utc, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol, timeframe, int(row['timestamp']), row['datetime_utc'],
                        float(row['open']), float(row['high']), float(row['low']),
                        float(row['close']), int(row['volume'])
                    ))
                
                self.conn.commit()
                logger.info(f"Stored {len(df)} mock bars for {symbol}")
                total_rows += len(df)
                
            except Exception as e:
                logger.error(f"Failed to store {symbol}: {e}")
        
        logger.info(f"Total mock internals data stored: {total_rows} rows")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


def setup_mock_internals_data(days_back: int = 5):
    """
    Convenience function to generate and store mock Internals data.
    
    Args:
        days_back: Number of days of data to generate
    """
    generator = MockInternalsGenerator()
    
    # Generate data for the past N days
    end_date = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)  # 4 PM
    start_date = end_date - timedelta(days=days_back)
    
    logger.info(f"Generating mock Internals data from {start_date} to {end_date}")
    
    # Generate the data
    data = generator.generate_internals_data(start_date, end_date, interval_minutes=5)
    
    # Store in database
    generator.store_internals_data(data)
    
    generator.close()
    
    logger.info("✅ Mock Internals data setup complete!")


if __name__ == '__main__':
    setup_mock_internals_data(days_back=5)
