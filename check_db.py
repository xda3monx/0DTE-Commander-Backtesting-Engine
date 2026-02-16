import sqlite3

conn = sqlite3.connect('backtesting_data.db')
cursor = conn.cursor()

# Check tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]
print('Tables:', tables)

# Check indicators table schema
if 'indicators' in tables:
    cursor.execute('PRAGMA table_info(indicators)')
    columns = cursor.fetchall()
    print('Indicators table schema:')
    for col in columns:
        print(f"  {col[1]}: {col[2]}")
else:
    print('Indicators table does not exist')

# Check if there's any data
for table in tables:
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    count = cursor.fetchone()[0]
    print(f"{table}: {count} rows")

conn.close()