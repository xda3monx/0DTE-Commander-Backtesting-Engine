import yfinance as yf

df = yf.download('SPY', period='1mo', interval='5m', progress=False)
print('Columns type:', type(df.columns))
print('Columns list sample:', list(df.columns)[:20])
print('Columns repr:', repr(df.columns))
