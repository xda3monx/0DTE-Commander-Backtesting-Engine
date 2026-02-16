import sqlite3
import os

db_files = ['backtesting_data.db']

for db_file in db_files:
    if os.path.exists(db_file):
        print(f"\nChecking {db_file}:")
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()

            # Check tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            print('Tables:', [t[0] for t in tables])

            if 'historical_data' in [t[0] for t in tables]:
                # Check date range for SPY
                cursor.execute('SELECT MIN(datetime_utc), MAX(datetime_utc) FROM historical_data WHERE symbol = "SPY"')
                date_range = cursor.fetchone()
                print('SPY date range:', date_range)

                # Check count in 2023
                cursor.execute('SELECT COUNT(*) FROM historical_data WHERE symbol = "SPY" AND datetime_utc >= "2023-08-01" AND datetime_utc <= "2023-12-31"')
                count = cursor.fetchone()[0]
                print('SPY rows in 2023:', count)

            conn.close()
        except Exception as e:
            print(f"Error checking {db_file}: {e}")
    else:
        print(f"{db_file} does not exist")