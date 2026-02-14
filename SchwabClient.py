import schwab
from schwab.auth import easy_client
import sqlite3
import time
from datetime import datetime, timedelta
import pandas as pd
from IndicatorEngine import IndicatorEngine
from JournalEngine import JournalEngine

# Your unique application ID
API_KEY = 'BR0o3XFTyp4HUy5z7dWYX3IzgWGlJrIN'
# Your Consumer Key/Secret (Schwab terminology)
APP_SECRET = 'mgG7N0t2AKGSAIKu'
# The callback URL you registered in the Schwab Developer Portal.
CALLBACK_URL = 'https://127.0.0.1:8182'
# The file path where the script will save your refresh token.
TOKEN_PATH = 'schwab_token.json' 

# This is the encrypted identifier required for all account-specific API calls (e.g., placing orders, checking balances).
ACCOUNT_HASH = 'F2CD8657DA2EE4542D6CFA3F9786570754AACEAAA493E37E7DBC9B3C6F9BFD40'

# The DB path can stay here too
DB_PATH = 'trading_data.db'

def authenticate_client():
    """
    Handles OAuth2 authentication and token persistence.
    If 'schwab_token.json' exists, it refreshes the token.
    If not, it opens a browser for manual login and creates the file.
    """
    print(f"Attempting to create/load client using token file: {TOKEN_PATH}")

    # easy_client handles the initial browser login OR the refresh logic
    try:
        client = easy_client(
            api_key=API_KEY,
            app_secret=APP_SECRET,
            callback_url=CALLBACK_URL,
            token_path=TOKEN_PATH
        )
        print("✅ Schwab Client authenticated successfully.")
        return client
    except Exception as e:
        print(f"❌ Authentication Failed. Check your API Key/Secret and internet connection.")
        print(f"Error details: {e}")
        return None

def fetch_and_store_bars(client, symbol, timeframe_seconds):
    """
    Fetches OHLC bar data and inserts/updates the 'bars' table in SQLite.
    It fetches 1 day of 1-minute data and then resamples it into the target timeframe 
    (e.g., 3m or 5m bars) before loading into the database.
    """
    conn = sqlite3.connect(DB_PATH)
    # Convert seconds to a Pandas frequency string (e.g., 180s -> '3T' or '5T')
    timeframe_str = f'{int(timeframe_seconds / 60)}min'
    print(f"\n--- Fetching 1-minute data for {symbol} ---")

    try:
        # 1. API Extraction (The 'E' in ETL)
        resp = client.get_price_history_every_minute(
            symbol # Pass only the symbol
            # All frequency and period details are assumed by the function name
        )
        
        candles = resp.json().get('candles', [])
        if not candles:
            print(f"⚠️ No candle data returned for {symbol}.")
            return
            
        df = pd.DataFrame(candles)
        
        # 🚨 DEFENSIVE FIX FOR VOLUME OMISSION 🚨
        if 'volume' not in df.columns:
            # If the API response is for an index and volume is missing, create the column with zero.
            df['volume'] = 0
            
        # Ensure all columns are the correct numeric type for calculations
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) 
            # 'errors=coerce' turns non-numeric values ('null' strings, etc.) into NaN, then fillna(0) makes them zero.
        # END OF FIX

        # Data Transformation (Initial 1m setup)
        df['symbol'] = symbol
        df['epoch_time'] = (df['datetime'] / 1000).astype(int) # Convert ms to seconds
        df['timeframe'] = '1m' # Store 1m bars as reference
        df = df[['symbol', 'timeframe', 'epoch_time', 'open', 'high', 'low', 'close', 'volume']]
        
        # Load 1m data (optional, but good for historical checks)
        df.to_sql('bars', conn, if_exists='append', index=False)
        print(f"✅ Successfully ingested {len(df)} 1m bars for {symbol}.")

        # 2. Resampling Transformation (The 'T' in ETL)
        if timeframe_seconds > 60:
            print(f"--- Resampling 1m data to {timeframe_str} bars ---")
            
            # Convert epoch time to datetime objects for resampling
            df['datetime_utc'] = pd.to_datetime(df['epoch_time'], unit='s', utc=True)
            df = df.set_index('datetime_utc')

            # Pandas Resample: This is the core logic that aggregates the data.
            # 'closed='left'' and 'label='left'' ensure the time stamp corresponds to the START of the bar.
            resampled_bars = df.resample(timeframe_str, closed='left', label='left').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            resampled_bars = resampled_bars.dropna().reset_index()
            resampled_bars['symbol'] = symbol
            resampled_bars['timeframe'] = timeframe_str.replace('T', 'm') # Rename '3T' to '3m' for DB consistency
            resampled_bars['epoch_time'] = resampled_bars['datetime_utc'].astype(int) // 10**9 # Convert back to seconds
            
            # 3. Loading the Resampled Data (The 'L' in ETL)
            resampled_bars[['symbol', 'timeframe', 'epoch_time', 'open', 'high', 'low', 'close', 'volume']].to_sql('bars', conn, if_exists='append', index=False)
            print(f"✅ Successfully created and stored {len(resampled_bars)} {timeframe_str.replace('T', 'm')} bars for {symbol}.")
            
    except Exception as e:
        print(f"❌ Error fetching data for {symbol}: {e}")
    finally:
        conn.close()

# --- Main execution block for testing ---
if __name__ == '__main__':
    schwab_client = authenticate_client()

    if schwab_client:
        # 1. ETL (Bars)
        fetch_and_store_bars(schwab_client, '/ES', 180) 
        fetch_and_store_bars(schwab_client, '$SPX', 300) 

        # 2. Indicator Calculation
        engine = IndicatorEngine()
        engine.calculate_indicators_and_regimes('/ES', 3)
        engine.calculate_indicators_and_regimes('$SPX', 5)

        # 3. Journaling (The New Step)
        journal = JournalEngine(client=schwab_client)
        journal.process_and_enrich_fills()