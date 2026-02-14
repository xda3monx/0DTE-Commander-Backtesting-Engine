import sqlite3
import time
from datetime import datetime, timedelta, timezone
import json
import os
import logging
from SchwabClient import authenticate_client

try:
    from win10toast import ToastNotifier
    TOAST_AVAILABLE = True
except Exception:
    TOAST_AVAILABLE = False

DB_PATH = 'trading_data.db'

DEFAULT_CONFIG = {
    'watchlist': ['/ES', '$SPX'],
    'timeframe': '3min',
    'poll_interval_seconds': 10,
    'exclude_first_minutes': 15,
    'position_size_contracts': 1,
}

logging.basicConfig(level=logging.INFO)


class TradeMonitor:
    def __init__(self, db_path=DB_PATH, config=None):
        self.conn = sqlite3.connect(db_path)
        self.client = authenticate_client()
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        if TOAST_AVAILABLE:
            self.notifier = ToastNotifier()
        self._create_signals_table()

    def _create_signals_table(self):
        # Basic signals table to store entry and exit signals
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                signal_type TEXT,
                side TEXT,
                signal_price REAL,
                trade_epoch INTEGER,
                reason TEXT,
                payload TEXT,
                created_at TEXT
            )
        ''')
        self.conn.commit()

    def notify(self, title, body):
        logging.info(f"NOTIFY: {title} - {body}")
        if TOAST_AVAILABLE:
            try:
                self.notifier.show_toast(title, body, duration=5, threaded=True)
            except Exception:
                pass

    def run(self):
        print('Starting TradeMonitor...')
        last_checked_epoch = None
        while True:
            try:
                if not self._is_allowed_trading_time():
                    time.sleep(self.config['poll_interval_seconds'])
                    continue

                # Get latest epoch for /ES indicators
                es_bar = self._get_latest_bar('/ES', self.config['timeframe'])
                spx_bar = self._get_latest_bar('$SPX', self.config['timeframe'])
                if es_bar is None or spx_bar is None:
                    time.sleep(self.config['poll_interval_seconds'])
                    continue

                # Only act on new bar completion
                cur_epoch = es_bar['epoch_time']
                if last_checked_epoch is not None and cur_epoch == last_checked_epoch:
                    time.sleep(self.config['poll_interval_seconds'])
                    continue

                last_checked_epoch = cur_epoch

                # Get latest indicators (current and previous) for /ES
                es_ix = self._get_indicators('/ES', self.config['timeframe'], limit=2)
                spx_ix = self._get_indicators('$SPX', self.config['timeframe'], limit=2)
                if len(es_ix) < 2 or len(spx_ix) < 2:
                    time.sleep(self.config['poll_interval_seconds'])
                    continue

                # Evaluate signals
                entry_signal, reason = self._evaluate_entry(es_bar, spx_bar, es_ix, spx_ix)
                if entry_signal:
                    # don't issue entry if there is already an open position for this symbol
                    if self._has_open_position('/ES'):
                        logging.info('Skipping ENTRY signal because an open position exists for /ES')
                    else:
                        payload = json.dumps({'es_bar': es_bar, 'spx_bar': spx_bar, 'reason': reason}, default=str)
                        self._insert_signal('/ES', 'ENTRY', entry_signal['side'], entry_signal['price'], cur_epoch, reason, payload)
                        self.notify('Trade Signal', f"{entry_signal['side']} /ES at {entry_signal['price']} - {reason}")

                # Evaluate exit signals for open positions
                exit_signal, reason = self._evaluate_exit(es_bar, spx_bar, es_ix, spx_ix)
                if exit_signal:
                    # Only exit if there is an open position
                    if self._has_open_position('/ES'):
                        payload = json.dumps({'es_bar': es_bar, 'spx_bar': spx_bar, 'reason': reason}, default=str)
                        self._insert_signal('/ES', 'EXIT', exit_signal['side'], exit_signal['price'], cur_epoch, reason, payload)
                        self.notify('Exit Signal', f"{exit_signal['side']} /ES at {exit_signal['price']} - {reason}")
                    else:
                        logging.info('Skipping EXIT signal because no open position exists for /ES')

                time.sleep(self.config['poll_interval_seconds'])

            except KeyboardInterrupt:
                print('TradeMonitor stopped by user')
                return
            except Exception as e:
                logging.exception('Error in monitor loop: %s', e)
                time.sleep(self.config['poll_interval_seconds'])

    def _insert_signal(self, symbol, signal_type, side, price, epoch, reason, payload):
        self.conn.execute('INSERT INTO signals(symbol, signal_type, side, signal_price, trade_epoch, reason, payload, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                          (symbol, signal_type, side, price, epoch, reason, payload, datetime.utcnow().isoformat()))
        self.conn.commit()

    def _has_open_position(self, symbol):
        cur = self.conn.cursor()
        q = "SELECT signal_type FROM signals WHERE symbol = ? ORDER BY created_at DESC LIMIT 1"
        r = cur.execute(q, (symbol,)).fetchone()
        if not r:
            return False
        return r[0] == 'ENTRY'

    def _get_latest_bar(self, symbol, timeframe):
        # Query the bars table for the latest 3min bar for the symbol
        cur = self.conn.cursor()
        query = "SELECT symbol, timeframe, epoch_time, open, high, low, close, volume FROM bars WHERE symbol = ? AND timeframe = ? ORDER BY epoch_time DESC LIMIT 1"
        rows = cur.execute(query, (symbol, timeframe)).fetchall()
        if not rows:
            return None
        row = rows[0]
        return {'symbol': row[0], 'timeframe': row[1], 'epoch_time': row[2], 'open': row[3], 'high': row[4], 'low': row[5], 'close': row[6], 'volume': row[7]}

    def _get_indicators(self, symbol, timeframe, limit=2):
        cur = self.conn.cursor()
        query = "SELECT symbol, timeframe, epoch_time, adx_val, williams_r, williams_r_fast, vwap, atr_val, trend_regime, exhaustion_signal FROM indicators WHERE symbol = ? AND timeframe = ? ORDER BY epoch_time DESC LIMIT ?"
        rows = cur.execute(query, (symbol, timeframe, limit)).fetchall()
        results = []
        for r in rows:
            results.append({'symbol': r[0], 'timeframe': r[1], 'epoch_time': r[2], 'adx_val': r[3], 'williams_r': r[4], 'williams_r_fast': r[5], 'vwap': r[6], 'atr_val': r[7], 'trend_regime': r[8], 'exhaustion_signal': r[9]})
        return results

    def _compute_basis(self, es_close, spx_close, cur_epoch):
        # Basis = es_close - spx_close, compute basis_sma_20 using last 20 epochs with matching times
        cur = self.conn.cursor()
        # Query last 20 matching epochs for both symbols
        query = "SELECT b.epoch_time, b.close as es_close, s.close as spx_close FROM bars as b JOIN bars as s ON b.epoch_time = s.epoch_time WHERE b.symbol = ? AND s.symbol = ? AND b.timeframe = ? ORDER BY b.epoch_time DESC LIMIT 20"
        rows = cur.execute(query, ('/ES', '$SPX', self.config['timeframe'])).fetchall()
        basis_list = []
        for r in rows:
            basis_list.append(r[1] - r[2])
        if not basis_list:
            return 0, 0
        import statistics
        sma20 = statistics.mean(basis_list)
        basis_cur = es_close - spx_close
        return basis_cur, sma20

    def _evaluate_entry(self, es_bar, spx_bar, es_ix, spx_ix):
        # Returns (signal_dict/None, reason)
        prev = es_ix[1]
        cur = es_ix[0]
        # Basis & context
        basis_cur, basis_sma = self._compute_basis(es_bar['close'], spx_bar['close'], es_bar['epoch_time'])
        context_score = basis_cur - basis_sma

        # Long Entry
        # Context positive, williams_r_fast crosses above -80 from <=-80, Close > VWAP
        if context_score > 0:
            if prev['williams_r_fast'] <= -80 and cur['williams_r_fast'] > -80 and es_bar['close'] > cur['vwap']:
                return ({'side': 'BUY', 'price': es_bar['close']}, 'Long entry: Hook Up + Engine (> VWAP) + Context')

        # Short Entry
        if context_score < 0:
            if prev['williams_r_fast'] >= -20 and cur['williams_r_fast'] < -20 and es_bar['close'] < cur['vwap']:
                return ({'side': 'SELL', 'price': es_bar['close']}, 'Short entry: Hook Down + Engine (< VWAP) + Context')

        return (None, None)

    def _evaluate_exit(self, es_bar, spx_bar, es_ix, spx_ix):
        # Returns (signal_dict/None, reason)
        prev = es_ix[1]
        cur = es_ix[0]
        # Profit Exit
        if cur['williams_r_fast'] >= -20:
            return ({'side': 'SELL', 'price': es_bar['close']}, 'Profit exit: williams_r_fast >= -20')
        if cur['williams_r_fast'] <= -80:
            return ({'side': 'BUY', 'price': es_bar['close']}, 'Profit exit: williams_r_fast <= -80')

        # Stop Exit (ATR based) - requires position context; we check using a simple placeholder
        # TODO: implement position-aware stop-loss by tracking open positions
        return (None, None)

    def _is_allowed_trading_time(self):
        # Exclude first 15 minutes of market open (assume 9:30 ET)
        # Convert to US/Eastern local time using zoneinfo
        from datetime import datetime, time
        try:
            from zoneinfo import ZoneInfo
            eastern = ZoneInfo('America/New_York')
        except Exception:
            # fallback to UTC if zoneinfo not available
            from datetime import timezone
            eastern = timezone.utc
        now = datetime.now(eastern)
        # Only allow Monday-Friday
        if now.weekday() >= 5:
            return False
        # Trading hours 9:30 - 16:00 ET
        if now.time() < time(9, 45) or now.time() > time(16, 0):
            return False
        # If within first exclude minutes after open
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        if (now - market_open).total_seconds() < (self.config['exclude_first_minutes'] * 60):
            return False
        return True

if __name__ == '__main__':
    tm = TradeMonitor()
    tm.run()
