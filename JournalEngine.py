import pandas as pd
import sqlite3
import numpy as np
import schwab
from datetime import datetime, timedelta
import re

# --- CONFIGURATION ---
DB_PATH = 'trading_data.db' 
# Define a range for history (e.g., last 90 days)
LOOKBACK_DAYS = 90
# Define the transaction types you want to include (e.g., trade executions)
TRADE_TYPES = ['TRADE']
# Schwab's API has many types; you might need to test which one captures the "Fill" event.

class JournalEngine:
    # Now requires the authenticated client object
    def __init__(self, client, db_path=DB_PATH): 
        self.client = client
        self.conn = sqlite3.connect(db_path)
        self.account_hash = 'F2CD8657DA2EE4542D6CFA3F9786570754AACEAAA493E37E7DBC9B3C6F9BFD40' # You must define this globally or pass it in!

    def fetch_and_normalize_trades(self):
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=LOOKBACK_DAYS)
        print(f"\n--- Fetching Trades from Schwab API ({start_date} to {end_date}) ---")

        # 1. API Extraction
        transactions_response = self.client.get_transactions(
            account_hash=self.account_hash,
            start_date=start_date,
            end_date=end_date,
        )

        transactions_data = transactions_response.json()

        # --- CRITICAL FIX: Extract the list of transactions or use an empty list ---
        # The API responses can vary between a direct list of transactions or a dict wrapper.
        # Normalize both possibilities to `transactions_list`.
        if isinstance(transactions_data, list):
            transactions_list = transactions_data
        elif isinstance(transactions_data, dict):
            # Common keys where the transaction list may be nested
            possible_keys = ['transactionCollection', 'transactions', 'data', 'transactionCollection']
            transactions_list = []
            for key in possible_keys:
                if key in transactions_data and isinstance(transactions_data[key], list):
                    transactions_list = transactions_data[key]
                    break
            # Fallback: find the first list value in the dict
            if not transactions_list:
                for v in transactions_data.values():
                    if isinstance(v, list):
                        transactions_list = v
                        break
        else:
            transactions_list = []
        
        # 2. Convert to DataFrame and also persist raw JSON for auditing
        raw_trades_df = pd.DataFrame(transactions_list) # Pass the extracted list
        # Persist raw transactions into a raw_transactions table for debug/audit
        try:
            # Create raw table if it doesn't exist
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS raw_transactions (
                    activityId TEXT,
                    orderId TEXT,
                    positionId TEXT,
                    tradeDate TEXT,
                    raw_json TEXT,
                    inserted_at TEXT
                )
            ''')
            self.conn.commit()
            import json as _json
            for tx in transactions_list:
                act = tx.get('activityId') if isinstance(tx, dict) else None
                oid = tx.get('orderId') if isinstance(tx, dict) else None
                pid = tx.get('positionId') if isinstance(tx, dict) else None
                tdate = tx.get('tradeDate') or tx.get('time') if isinstance(tx, dict) else None
                self.conn.execute('INSERT INTO raw_transactions(activityId, orderId, positionId, tradeDate, raw_json, inserted_at) VALUES (?, ?, ?, ?, ?, ?)',
                                  (act, oid, pid, tdate, _json.dumps(tx), datetime.utcnow().isoformat()))
            self.conn.commit()
        except Exception as e:
            print('Could not persist raw transactions: ', e)
        
        # 3. Handle Empty DataFrame (FIXED)
        if raw_trades_df.empty:
            print("⚠️ Retrieved 0 trades from the API (Empty response or no trades found in range).")
            # Ensure all columns used downstream are present with the correct type
            required_cols = ['symbol', 'transactionDate', 'entry_epoch', 'transactionId', 'type', 'price', 'quantity']
            empty_df = pd.DataFrame(columns=required_cols)

            # CRITICAL: Ensure the epoch column is int64
            empty_df['entry_epoch'] = empty_df['entry_epoch'].astype('int64')

            empty_df['stage'] = 'FILL' 
            return empty_df
        
        # 4. Filtering and Normalization
        
        # Filter to only keep relevant transactions (Fills/Executions)
        filtered_df = raw_trades_df[raw_trades_df['type'].isin(TRADE_TYPES)].copy()
        
        # Normalize/Clean Timestamps
        # Attempt to find a timestamp column from several possible names used by different APIs
        ts_candidates = [
            'transactionDate', 'transactionDateTime', 'transactionTimestamp',
            'tradeDate', 'tradeDateTime', 'trade_timestamp', 'timestamp',
            'date', 'time', 'createdAt', 'settlementDate'
        ]
        ts_col = None
        for c in ts_candidates:
            if c in filtered_df.columns:
                ts_col = c
                break

        if ts_col is None:
            # fallback: inspect any column that looks date/time-like
            possible_date_cols = [c for c in filtered_df.columns if 'date' in c.lower() or 'time' in c.lower()]
            if possible_date_cols:
                ts_col = possible_date_cols[0]

        if ts_col is None:
            print(f"⚠️ No date/time field found in transaction columns: {list(filtered_df.columns)}; setting entry_time to NaT")
            filtered_df['entry_time'] = pd.NaT
            filtered_df['entry_epoch'] = pd.NA
        else:
            print(f"ℹ️ Using timestamp column '{ts_col}' for trade timestamps")
            filtered_df['entry_time'] = pd.to_datetime(filtered_df[ts_col])
            filtered_df['entry_epoch'] = (filtered_df['entry_time'].astype(np.int64) // 10**9)
        
        # IMPORTANT: This initial pull gives you *fills*, not trades (trade = entry + exit).
        # For V1, we will process fills and treat them as the 'event'.
        
        # Normalize each transaction into a canonical dict and collect into a list
        normalized_rows = []
        for idx, tx in filtered_df.iterrows():
            try:
                norm = self._normalize_transaction(tx)
                normalized_rows.append(norm)
            except Exception as e:
                print('Warning: failed to normalize tx', e)

        trades_for_join = pd.DataFrame(normalized_rows)

        # If there is no `symbol` column, try to detect it or extract it from nested structures
        if 'symbol' not in trades_for_join.columns:
            # Possible symbol column names
            sym_candidates = ['instrument', 'security', 'symbolDescription', 'symbolName', 'securitySymbol']
            found_sym = None
            for c in sym_candidates:
                if c in trades_for_join.columns:
                    trades_for_join = trades_for_join.rename(columns={c: 'symbol'})
                    found_sym = 'symbol'
                    break

            # If still not found, look for a dict-like column that contains a 'symbol' key
            if 'symbol' not in trades_for_join.columns:
                for c in trades_for_join.columns:
                    if trades_for_join[c].apply(lambda x: isinstance(x, dict) and 'symbol' in x).any():
                        trades_for_join['symbol'] = trades_for_join[c].apply(lambda x: x.get('symbol') if isinstance(x, dict) else None)
                        found_sym = 'symbol'
                        break

            if 'symbol' not in trades_for_join.columns:
                # Try to extract symbol information from 'transferItems' if present
                if 'transferItems' in trades_for_join.columns:
                    def _extract_from_transfer(x):
                        if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], dict) and 'symbol' in x[0]:
                            return x[0]['symbol']
                        if isinstance(x, dict) and 'symbol' in x:
                            return x['symbol']
                        return None
                    trades_for_join['symbol'] = trades_for_join['transferItems'].apply(_extract_from_transfer)

                # If still not found, try parse the 'description' field for symbol tokens
                if 'symbol' not in trades_for_join.columns and 'description' in trades_for_join.columns:
                    def _extract_from_desc(s):
                        if not isinstance(s, str):
                            return None
                        # Prefers tokens starting with / or $ (e.g., /ES or $SPX), else uppercase symbols
                        candidates = re.findall(r'([/$]?[A-Z0-9]{1,6})', s)
                        if not candidates:
                            return None
                        for c in candidates:
                            if c.startswith('/') or c.startswith('$'):
                                return c
                        return candidates[0]
                    trades_for_join['symbol'] = trades_for_join['description'].apply(_extract_from_desc)

                # Last resort: print warning showing columns and few sample descriptions to aid debugging
                if 'symbol' not in trades_for_join.columns or trades_for_join['symbol'].isna().all():
                    print(f"⚠️ Cannot find a symbol column in transaction data. Available columns: {list(trades_for_join.columns)}")
                    # Show the first few descriptions to help debug
                    if 'description' in trades_for_join.columns:
                        print("Sample descriptions:")
                        print(trades_for_join['description'].head(5).to_list())

        print(f"✅ Retrieved {len(trades_for_join)} trade fills from the API.")
        # Normalize expected transaction ID to 'transactionId' for downstream code
        if 'transactionId' not in trades_for_join.columns:
            if 'activityId' in trades_for_join.columns:
                trades_for_join = trades_for_join.rename(columns={'activityId': 'transactionId'})
            elif 'orderId' in trades_for_join.columns:
                trades_for_join = trades_for_join.rename(columns={'orderId': 'transactionId'})
            elif 'positionId' in trades_for_join.columns:
                trades_for_join['transactionId'] = trades_for_join['positionId']

        return trades_for_join

    def _normalize_transaction(self, tx_row):
        """Normalize a tx (pandas row) into canonical fields.
        Returns a dict with: transactionId, symbol, side, entry_time, entry_epoch, price, quantity, orderId, netAmount, commission, description, type, positionId, option fields, implied_volatility
        """
        # Row is a Pandas Series; convert to dict for easier work
        tx = tx_row.to_dict() if hasattr(tx_row, 'to_dict') else dict(tx_row)
        normalized = {}
        # identifiers
        normalized['transactionId'] = tx.get('activityId') or tx.get('orderId') or tx.get('positionId')
        normalized['orderId'] = tx.get('orderId')
        normalized['positionId'] = tx.get('positionId')
        normalized['accountNumber'] = tx.get('accountNumber')

        # timestamps
        ts = tx.get('tradeDate') or tx.get('time')
        if ts:
            try:
                dt = pd.to_datetime(ts)
                normalized['entry_time'] = dt
                normalized['trade_epoch'] = int(dt.value // 10**9)
            except Exception:
                normalized['entry_time'] = pd.NaT
                normalized['trade_epoch'] = pd.NA
        else:
            normalized['entry_time'] = pd.NaT
            normalized['trade_epoch'] = pd.NA

        normalized['type'] = tx.get('type')
        normalized['status'] = tx.get('status')
        normalized['description'] = tx.get('description')
        normalized['netAmount'] = tx.get('netAmount')

        # Find the primary transfer item (non-currency), often it's the first asset-type that is not 'CURRENCY'
        transfer_items = tx.get('transferItems') or []
        primary_item = None
        fees = []
        for it in transfer_items:
            inst = it.get('instrument') if isinstance(it, dict) else None
            if inst and inst.get('assetType') and inst.get('assetType').upper() != 'CURRENCY' and primary_item is None:
                primary_item = it
            # fee detection based on feeType field
            if it.get('feeType'):
                fees.append(it)

        # Use a fallback: if no primary_item found, pick the first instrument-based item
        if primary_item is None and transfer_items:
            for it in transfer_items:
                if isinstance(it, dict) and it.get('instrument'):
                    primary_item = it
                    break

        # Extract data from primary_item
        normalized['price'] = None
        normalized['quantity'] = None
        normalized['symbol'] = None
        normalized['assetType'] = None
        normalized['option_symbol'] = None
        normalized['strikePrice'] = None
        normalized['expirationDate'] = None
        normalized['putCall'] = None
        normalized['option_multiplier'] = None
        if primary_item:
            inst = primary_item.get('instrument', {})
            normalized['assetType'] = inst.get('assetType')
            # symbol for instrument (e.g., option symbol or underlying like '$SPX')
            normalized['symbol'] = inst.get('underlyingSymbol') or inst.get('symbol') or None
            # For options, preserve the raw option symbol
            if inst.get('assetType') and inst.get('assetType').upper() == 'OPTION':
                normalized['option_symbol'] = inst.get('symbol')
                normalized['strikePrice'] = inst.get('strikePrice')
                normalized['expirationDate'] = inst.get('expirationDate')
                normalized['putCall'] = inst.get('putCall')
                normalized['option_multiplier'] = inst.get('optionPremiumMultiplier') or inst.get('optionPremiumMultiplier')
            # quantities and price
            amount = primary_item.get('amount')
            price = primary_item.get('price')
            normalized['quantity'] = amount
            normalized['price'] = price

        # Commission and fees
        commission_val = 0
        total_fees = 0
        for f in transfer_items:
            if isinstance(f, dict):
                fee_type = f.get('feeType')
                cost = f.get('cost')
                if cost is None:
                    # some fee items may use 'amount' with opposite sign
                    cost = -f.get('amount', 0)
                if fee_type == 'COMMISSION':
                    if cost is not None:
                        commission_val += float(cost)
                if cost is not None:
                    total_fees += float(cost)
        normalized['commission'] = commission_val
        normalized['total_fees'] = total_fees

        # Net amount excluding commission (user said netAmount should not include commission) — net_ex_commission
        try:
            net_amount = float(tx.get('netAmount')) if tx.get('netAmount') is not None else None
            if net_amount is not None:
                # commission_val is usually negative; subtracting it removes its effect
                # e.g., netAmount = -3112.21, commission_val = -6.5, net_ex_commission = -3112.21 - (-6.5) = -3105.71
                normalized['net_ex_commission'] = net_amount - commission_val
        except Exception:
            normalized['net_ex_commission'] = tx.get('netAmount')

        # Side inference: user does only long BUY for calls/puts. We can still infer if unclear
        # We'll infer BUY if netAmount (cash out) is negative OR if positionEffect == 'OPENING' and amount > 0
        normalized['side'] = None
        try:
            if primary_item and primary_item.get('positionEffect'):
                if primary_item.get('positionEffect').upper() == 'OPENING':
                    normalized['side'] = 'BUY'
                elif primary_item.get('positionEffect').upper() == 'CLOSING':
                    normalized['side'] = 'SELL'
            if normalized['side'] is None and 'netAmount' in normalized and normalized['netAmount'] is not None:
                # typically negative netAmount indicates buy
                normalized['side'] = 'BUY' if float(normalized['netAmount']) < 0 else 'SELL'
        except Exception:
            normalized['side'] = None

        # Fill implied volatility if possible (Option only)
        normalized['implied_vol'] = None
        if normalized.get('assetType') and normalized['assetType'].upper() == 'OPTION':
            try:
                # best-effort fetch; we use the API if available
                iv = self._fetch_option_implied_vol(primary_item.get('instrument'), normalized['entry_time'])
                normalized['implied_vol'] = iv
            except Exception:
                normalized['implied_vol'] = None

        # include raw transfer items as small JSON for debugging
        normalized['transferItems'] = transfer_items

        return normalized

    def _fetch_option_implied_vol(self, instrument, timestamp=None):
        """Attempt to fetch implied volatility using the client API given an instrument dict.
        Returns a float or None. We try a few client methods defensively.
        """
        if instrument is None:
            return None
        # instrument may be dict with 'symbol'. We'll first try to find a quote endpoint
        option_sym = instrument.get('symbol') or instrument.get('instrumentId')
        if option_sym is None:
            return None
        # If the client provides an 'get_option_quote' or 'get_quotes', try it.
        try:
            if hasattr(self.client, 'get_option_quote'):
                resp = self.client.get_option_quote(option_symbol=option_sym)
                data = resp.json()
                # try common keys
                iv = None
                if isinstance(data, dict):
                    iv = (data.get('impliedVolatility') or data.get('impliedVol'))
                if iv is not None:
                    return float(iv)
            if hasattr(self.client, 'get_quotes'):
                # Grenade: some client wrappers expect a list or single symbol
                try:
                    resp = self.client.get_quotes(symbols=[option_sym])
                except TypeError:
                    resp = self.client.get_quotes(symbol=option_sym)
                q = resp.json()
                # q might be a dict keyed by symbol
                if isinstance(q, dict):
                    entry = q.get(option_sym) or next(iter(q.values()))
                    if entry:
                        iv = entry.get('impliedVolatility') or entry.get('impliedVol')
                        if iv:
                            return float(iv)
        except Exception as e:
            # This is a best-effort; don't fail the whole pipeline on IV lookup
            print('IV fetch attempt failed: ', e)
            return None
        return None
        
    def process_and_enrich_fills(self):
        # We need to simplify the logic to process individual FILLS instead of ENTRY/EXIT pairs.
        
        # 1. Fetch Transactions via API (Replaces CSV/mock data)
        fill_events = self.fetch_and_normalize_trades()
        
        # Ensure we have the necessary epoch time column, and rename it for joining
        fill_events = fill_events.rename(columns={'entry_epoch': 'trade_epoch'}) 
        fill_events['stage'] = 'FILL'

        # 2. Extract Indicators and Prepare for Join
        indicators_df = pd.read_sql_query("SELECT * FROM indicators ORDER BY epoch_time ASC", self.conn)
        # Normalize indicator column names to match expected final table names
        indicator_renames = {}
        if 'ADX_14' in indicators_df.columns:
            indicator_renames['ADX_14'] = 'adx_val'
        if 'WILLR_14' in indicators_df.columns:
            indicator_renames['WILLR_14'] = 'williams_r'
        if 'ATRr_14' in indicators_df.columns:
            indicator_renames['ATRr_14'] = 'atr_val'
        # If the exact 'trend_regime' and 'exhaustion_signal' columns don't exist, create defaults
        indicators_df = indicators_df.rename(columns=indicator_renames)
        if 'trend_regime' not in indicators_df.columns:
            indicators_df['trend_regime'] = None
        if 'exhaustion_signal' not in indicators_df.columns:
            indicators_df['exhaustion_signal'] = None
        
        # 3. The Confluence Analysis (Merge As Of)
        # Sort by trade_epoch is CRITICAL for merge_asof
        fill_events = fill_events.sort_values('trade_epoch')
        
        # Join each FILL event to the closest preceding indicator bar
        enriched_df = pd.merge_asof(
            fill_events, 
            indicators_df, 
            left_on='trade_epoch', 
            right_on='epoch_time',
            by='symbol',                
            direction='backward'        
        )
        
        # 4. Final Clean and Load
        final_table = enriched_df[[
            'transactionId', 'symbol', 'stage', 'trade_epoch', 'epoch_time', 
            'adx_val', 'williams_r', 'atr_val', 'trend_regime', 'exhaustion_signal'
        ]].rename(columns={'epoch_time': 'linked_bar_epoch', 'transactionId': 'fill_id'})
        
        # Load into the final snapshot table
        final_table.to_sql('trade_context_snapshot', self.conn, if_exists='replace', index=False)
        print(f"✅ Successfully created trade_context_snapshot with {len(final_table)} enriched fills.")
