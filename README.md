# 0DTE Commander Backtesting Engine

A robust, vectorized backtesting framework for high-frequency trading strategies based on **Velocity, Drift, and Volatility Physics**.

## 🎯 Mission Transition: Discretionary Trader → Algorithmic Systems Architect

This project represents a fundamental shift from loop-based OOP thinking to **vectorized Pandas mindset** for processing high-frequency financial data efficiently.

### Key Architectural Changes

| **Old Approach (Loop-based)** | **New Approach (Vectorized)** |
|-------------------------------|-------------------------------|
| Manual for-loops for calculations | Pandas `.ewm()`, `.rolling()`, `.pct_change()` |
| Object-oriented indicator classes | Functional indicator pipelines |
| Real-time data processing | Historical batch processing |
| SQLite for live trading | SQLite for backtesting data |
| Individual trade execution | Strategy-level performance analysis |

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Data Ingestion (6 Months of 5-minute Data)
```bash
# Using Schwab API (if available)
python data_ingestion.py --symbol SPX --months 6 --timeframe 5

# Using CSV fallback (recommended for extensive historical data)
python data_ingestion.py --csv spy_5min_data.csv --symbol SPY
```

### 3. Vectorized EMA Demonstration
```bash
python data_ingestion.py --demo-ema
```

## 📊 Vectorized Indicator Calculations

### The "Drift" Component: EMA Calculations (No Loops!)

**Traditional Loop Approach (What NOT to do):**
```python
# ❌ SLOW: Manual loop-based calculation
ema_values = []
multiplier = 2 / (span + 1)
for i in range(len(prices)):
    if i == 0:
        ema_values.append(prices[i])
    else:
        ema = (prices[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
        ema_values.append(ema)
```

**Vectorized Pandas Approach (What TO do):**
```python
# ✅ FAST: Single line vectorized calculation
df['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
df['ema_24'] = df['close'].ewm(span=24, adjust=False).mean()
```

### Why Vectorized Operations Matter

- **Performance**: NumPy's C-optimized code vs Python loops
- **Memory Efficiency**: No intermediate storage of loop variables
- **Code Clarity**: Mathematical operations read like formulas
- **Scalability**: Handles millions of data points seamlessly

## 🏗️ Project Structure

```
SchwabAlgo/
├── BacktestingEngine.py      # Core backtesting framework
├── data_ingestion.py         # Data fetching and processing script
├── requirements.txt          # Python dependencies
├── README.md                 # This documentation
├── SchwabClient.py           # Original Schwab API client (preserved)
├── IndicatorEngine.py        # Original indicators (preserved)
├── JournalEngine.py          # Original journaling (preserved)
└── backtesting_data.db       # SQLite database for backtesting data
```

## 🔧 Core Components

### 1. Data Pipeline
- **Ingestion**: Schwab API or CSV import
- **Cleaning**: Invalid data removal, deduplication
- **Storage**: SQLite database with proper indexing

### 2. Indicator Engine (Vectorized)
- **Drift**: 8-period and 24-period EMAs
- **Velocity**: ROC, momentum, and combined velocity
- **Volatility**: ATR-based volatility measures

### 3. Regime Classification
- Bull/Bear trending markets
- Ranging and choppy conditions
- Volatility-based regime detection

## 📈 Usage Examples

### Basic Data Loading and Processing
```python
from BacktestingEngine import BacktestingEngine

# Initialize engine
engine = BacktestingEngine()

# Process 6 months of SPY 5-minute data
df = engine.run_data_pipeline('SPY', months=6, timeframe_minutes=5, use_csv=True, csv_path='spy_data.csv')

# Data now contains vectorized indicators
print(df[['close', 'ema_8', 'ema_24', 'velocity', 'volatility', 'regime']].head())
```

### Retrieving Processed Data
```python
# Get data with date range
df = engine.get_historical_data('SPY', '5min', '2023-01-01', '2023-12-31')

# Analyze drift strength
df['drift_strength'] = df['ema_8'] - df['ema_24']
strong_trends = df[abs(df['drift_strength']) > df['close'] * 0.01]
```

## 🎓 Learning the Vectorized Mindset

### Common Transition Patterns

**Pattern 1: Moving Averages**
```python
# Loop → Vectorized
# OLD: for i in range(len(data)): sma[i] = data[i-20:i].mean()
# NEW:
df['sma_20'] = df['close'].rolling(window=20).mean()
```

**Pattern 2: Rate of Change**
```python
# Loop → Vectorized
# OLD: for i in range(len(data)): roc[i] = (data[i] - data[i-5]) / data[i-5]
# NEW:
df['roc_5'] = df['close'].pct_change(periods=5)
```

**Pattern 3: Conditional Logic**
```python
# Loop → Vectorized
# OLD: for i in range(len(data)): if condition: signal[i] = 1
# NEW:
df['signal'] = np.where(df['ema_8'] > df['ema_24'], 1, -1)
```

## 🔄 Migration Path

### Phase 1: Data Ingest ✅ (Current)
- [x] Schwab API integration
- [x] CSV data loading
- [x] Vectorized EMA calculations
- [x] Data cleaning and validation

### Phase 2: Strategy Development (Next)
- [ ] Strategy signal generation
- [ ] Position sizing logic
- [ ] Risk management rules
- [ ] Performance metrics

### Phase 3: Backtesting Framework (Future)
- [ ] Walk-forward optimization
- [ ] Monte Carlo simulation
- [ ] Strategy comparison tools
- [ ] Live trading integration

## 📊 Data Sources

### Primary: Schwab API
- Real-time and historical data
- Direct broker integration
- Account and order management

### Fallback: CSV Import
For extensive historical data, use:
- **Yahoo Finance**: `yfinance` library
- **Alpha Vantage**: Free API with 5min data
- **Polygon.io**: Professional-grade data
- **Broker Exports**: Interactive Brokers, TD Ameritrade

### CSV Format Requirements
```csv
timestamp,open,high,low,close,volume
1693526400,450.25,451.10,449.80,450.95,125000
1693526700,450.95,451.50,450.70,451.25,98000
```

## 🚨 Important Notes

1. **API Limitations**: Schwab API may limit historical data depth
2. **Data Quality**: Always validate OHLCV data integrity
3. **Timezone Handling**: All timestamps stored as UTC
4. **Memory Usage**: Large datasets may require chunked processing
5. **Backtesting Bias**: Be aware of lookahead and survivorship bias

## 🤝 Contributing

This is your personal backtesting framework. Key areas for extension:
- Additional indicators (RSI, MACD, Bollinger Bands)
- Strategy implementations
- Risk management modules
- Performance visualization
- Live trading bridges

---

**Remember**: The transition from loops to vectorized operations isn't just about performance—it's about thinking in data flows rather than sequential steps. Welcome to the world of algorithmic trading! 🚀</content>
<parameter name="filePath">c:\Users\panic\OneDrive\Development\DxOrderDaemon\SchwabAlgo\README.md