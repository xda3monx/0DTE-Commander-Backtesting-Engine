import pandas as pd
import sqlite3
import pandas_ta as ta

# --- CONFIGURATION ---
DB_PATH = 'trading_data.db' 
# Define the periods your Thinkscript uses
ADX_LENGTH = 14
WR_LENGTH = 12
ADX_THRESHOLD = 25 # Thinkscript standard for trend strength

class IndicatorEngine:
    def __init__(self, db_path=DB_PATH):
        self.conn = sqlite3.connect(db_path)
    
    def calculate_indicators_and_regimes(self, symbol, timeframe):
        tf_str = f'{timeframe}min'
        print(f"\n--- Calculating Indicators for {symbol} ({tf_str}) ---")

        # 1. Extract (Load the Bars)
        query = f"SELECT * FROM bars WHERE symbol = '{symbol}' AND timeframe = '{tf_str}' ORDER BY epoch_time ASC"
        df = pd.read_sql_query(query, self.conn)

        if df.empty:
            print(f"⚠️ No bars found for {symbol} ({tf_str}). Skipping.")
            return

        # 2. Transformation (Thinkscript Logic - Vectorized)

        # Williams %R (Exhaustion Logic) - RAW
        # Note: ta.willr returns values in the -100..0 range
        df.ta.willr(length=WR_LENGTH, append=True)  # WILLR_12 (or WILLR_{WR_LENGTH})
        # Smooth it using EMA(length=3) to create our fast %R signal
        willr_col = f'WILLR_{WR_LENGTH}'
        if willr_col in df.columns:
            df['williams_r_fast'] = df[willr_col].ewm(span=3, adjust=False).mean()

        # ADX (Trend Strength/Direction)
        df.ta.adx(length=ADX_LENGTH, append=True) # Creates ADX_14, DMN_14, DMP_14

        # ATR (Volatility/Stop Size)
        df.ta.atr(length=14, append=True) # Creates ATR_14 or ATRr_14 depending on pandas_ta version
        # Normalize ATR column name if needed
        if 'ATRr_14' in df.columns and 'ATR_14' not in df.columns:
            df['ATR_14'] = df['ATRr_14']

        # 🚨 TEMPORARY DEBUG LINE 🚨
        print(df.columns.tolist())
        # 🚨 END DEBUG LINE 🚨

        # --- VWAP Calculation (intraday, per day) ---
        # Prepare a datetime column for session grouping
        df['datetime_utc'] = pd.to_datetime(df['epoch_time'], unit='s', utc=True)
        # Compute intraday VWAP (cumulative by trading date) using typical price = close
        def compute_vwap(group):
            vol_cum = group['volume'].cumsum()
            tp_v = (group['close'] * group['volume']).cumsum()
            # avoid divide by zero
            return (tp_v / vol_cum).fillna(method='ffill')
        try:
            df['vwap'] = df.groupby(df['datetime_utc'].dt.date).apply(compute_vwap).reset_index(level=0, drop=True)
        except Exception:
            df['vwap'] = df['close']

        # 3. Regime Determination (The Bridge Logic)
        
        # A. TREND/RANGE Regime
        # Thinkscript equivalent of: def trend = if ADX(14) > 25 then 1 else 0;
        df['trend_regime'] = 0 # Default to RANGE (0)
        # Condition: ADX > 25 (Strong Trend)
        df.loc[df[f'ADX_{ADX_LENGTH}'] > ADX_THRESHOLD, 'trend_regime'] = 1 # TREND (1)
        
        # B. EXHAUSTION Signal (Uses Williams %R)
        # Thinkscript equivalent of: plot BuySignal = WildersR(14) crosses below -80;
        df['exhaustion_signal'] = 0 # Default to none (0)
        # %R < -90 (Oversold/Buy Exhaustion)
        df.loc[df[f'WILLR_{WR_LENGTH}'] < -90, 'exhaustion_signal'] = -1 
        # %R > -10 (Overbought/Sell Exhaustion)
        df.loc[df[f'WILLR_{WR_LENGTH}'] > -10, 'exhaustion_signal'] = 1 

        # 4. Load (Save to 'indicators' Table)
        
        # Select and rename columns to match our proposed DB schema
        # Ensure columns exist; also include close and vwap and williams_r_fast
        indicator_df = df[[
            'symbol', 'timeframe', 'epoch_time', 'close',
            f'ADX_{ADX_LENGTH}',
            f'WILLR_{WR_LENGTH}',
            'williams_r_fast',
            'vwap',
            'ATR_14',
            'trend_regime', 'exhaustion_signal'
        ]].copy()
        
        indicator_df.columns = [
            'symbol', 'timeframe', 'epoch_time', 'close',
            'adx_val',                          # Matches ADX_14
            'williams_r',                       # Matches WILLR_12
            'williams_r_fast',
            'vwap',
            'atr_val',                          # Matches ATR_14 (we rename it to atr_val for the DB)
            'trend_regime', 'exhaustion_signal'
        ]
        
        # Insert into the database. If table doesn't exist, it creates it.
        try:
            indicator_df.to_sql('indicators', self.conn, if_exists='append', index=False)
        except Exception as e:
            # if schema changed (new columns), replace the table to match new schema
            print('Warning: could not append indicators (schema mismatch?), replacing table:', e)
            indicator_df.to_sql('indicators', self.conn, if_exists='replace', index=False)
        print(f"✅ Successfully calculated and stored {len(indicator_df)} indicator bars.")
