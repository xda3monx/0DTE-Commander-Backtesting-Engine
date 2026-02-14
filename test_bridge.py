import sys
import pandas as pd
import schwab

# 1. Verify Python Version
print("--- System Check ---")
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")

# 2. Verify Libraries
print("\n--- Library Check ---")
print(f"Pandas Version: {pd.__version__}")
print(f"Schwab-py Version: {schwab.__version__}")

# 3. Verify Data Capability
print("\n--- Data Test ---")
data = {
    'Symbol': ['SPX', '/ES', 'AAPL'],
    'Price': [5000.00, 5010.50, 175.00],
    'Regime': ['Bull', 'Bull', 'Neutral']
}
df = pd.DataFrame(data)
print("Pandas DataFrame created successfully:")
print(df)

print("\n[SUCCESS] The environment is ready for the Schwab API.")