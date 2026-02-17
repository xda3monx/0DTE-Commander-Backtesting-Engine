#!/usr/bin/env python3
"""
Comprehensive Backtesting Script for CommanderStrategy
Walks through the complete backtesting workflow
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from BacktestingEngine import BacktestingEngine, CommanderStrategy
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_data(symbol: str = '$SPX', days: int = 5) -> pd.DataFrame:
    """
    Create realistic sample data for testing.
    In production, you'd load real historical data.
    """
    logger.info(f"Creating {days} days of sample data for {symbol}")

    # Create timestamps for 5-minute bars during market hours
    start_date = pd.Timestamp('2024-01-01')
    timestamps = []

    for day in range(days):
        current_date = start_date + timedelta(days=day)
        if current_date.weekday() < 5:  # Monday-Friday
            # Market hours: 9:30 AM - 4:00 PM ET
            start_time = pd.Timestamp(f"{current_date.date()} 09:30:00", tz='America/New_York')
            end_time = pd.Timestamp(f"{current_date.date()} 16:00:00", tz='America/New_York')

            current_time = start_time
            while current_time <= end_time:
                timestamps.append(current_time)
                current_time += timedelta(minutes=5)

    # Generate realistic price action
    np.random.seed(42)
    base_price = 4500.0
    prices = []
    volumes = []

    for i, ts in enumerate(timestamps):
        # Add trend, volatility, and noise
        day_progress = (ts.hour * 60 + ts.minute - 570) / (6.5 * 60)  # 0 to 1 during day
        trend = 50 * np.sin(day_progress * 2 * np.pi)  # Daily cycle
        noise = np.random.normal(0, 8)  # Random noise
        seasonal = 20 * np.sin(i * 0.01)  # Longer-term trend

        price = base_price + trend + noise + seasonal
        price = max(price, 4200)  # Floor price

        prices.append(price)
        volumes.append(np.random.randint(50000, 200000))

    # Create OHLCV data
    data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        if i == 0:
            open_price = close
        else:
            open_price = data[-1]['close']

        high = max(open_price, close) + np.random.uniform(0, 5)
        low = min(open_price, close) - np.random.uniform(0, 5)

        data.append({
            'timestamp': int(ts.timestamp()),
            'datetime_utc': ts,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volumes[i],
            'symbol': symbol,
            'timeframe': '5min'
        })

    df = pd.DataFrame(data)
    logger.info(f"Created {len(df)} bars of sample data")
    return df

def run_complete_backtest(symbol: str = '$SPX', days: int = 5):
    """
    Complete backtesting workflow:
    1. Load/Create data
    2. Process indicators
    3. Generate signals
    4. Simulate trades
    5. Calculate performance
    """
    logger.info("="*60)
    logger.info("STARTING COMMANDER STRATEGY BACKTEST")
    logger.info("="*60)

    # Step 1: Initialize components
    logger.info("Step 1: Initializing backtesting engine and strategy")
    engine = BacktestingEngine()
    strategy = CommanderStrategy(risk_per_trade=0.005, max_daily_loss=0.02)

    # Step 2: Get historical data
    logger.info("Step 2: Loading historical data")
    # Use real data pipeline with VIX integration
    from phase1_data_pipeline import DataPipeline
    from vectorized_indicators import VectorizedIndicators
    
    pipeline = DataPipeline()
    try:
        df = pipeline.run_phase1_pipeline()
        logger.info(f"Loaded {len(df)} bars of real SPY/VIX data")
        
        # Load Internals data (UVOL/DVOL/TRIN/TICK/Mag7)
        logger.info("Loading Internals data...")
        internals_df = pipeline.load_internals_data(lookback_periods=len(df))
        
        if not internals_df.empty:
            # Merge internals data with price data
            df = df.reset_index() if hasattr(df, 'index') and isinstance(df.index, pd.DatetimeIndex) else df
            df = df.merge(internals_df, on='timestamp', how='left')
            # Forward fill internals data to handle gaps
            for col in internals_df.columns:
                if col != 'timestamp' and col in df.columns:
                    df[col] = df[col].ffill()
            logger.info(f"Merged Internals data. Total columns: {len(df.columns)}")
        else:
            logger.warning("No Internals data available, continuing with VIX veto only")
        
        # Step 3: Process data and calculate indicators
        logger.info("Step 3: Calculating indicators with VIX veto conditions")
        
        # Use VectorizedIndicators for comprehensive indicator calculation including VIX veto
        indicators = VectorizedIndicators(df)
        
        # Calculate core indicators (skip complex velocity physics due to data alignment issues)
        indicators.calculate_drift_indicators()
        
        # Add basic ATR calculation needed for regime states
        if 'atr' not in indicators.df.columns:
            high_low = indicators.df['high'] - indicators.df['low']
            high_close = (indicators.df['high'] - indicators.df['close'].shift(1)).abs()
            low_close = (indicators.df['low'] - indicators.df['close'].shift(1)).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            indicators.df['atr'] = tr.rolling(14).mean()
        
        # Calculate velocity physics first (includes ADX and is_chop)
        indicators.calculate_velocity_physics()
        
        indicators.calculate_regime_states()
        indicators.calculate_volume_physics()
        indicators.calculate_veto_conditions()  # This includes VIX-based veto logic
        indicators.calculate_market_bias()
        
        df = indicators.df
        logger.info(f"Calculated {len(df.columns)} indicator columns including VIX veto conditions")

        # Reset index to make datetime_utc a column for BacktestingEngine compatibility
        df = df.reset_index()

        # Step 4: Generate entry signals
        logger.info("Step 4: Generating entry signals")
        df = strategy.generate_entry_signals(df)

        # Step 5: Simulate trades
        logger.info("Step 5: Simulating trades")
        trades = strategy.simulate_trades(df, capital=10000.0)

        # Step 6: Calculate performance metrics
        logger.info("Step 6: Calculating performance metrics")
        metrics = strategy.calculate_performance_metrics(trades)

        # Step 7: Display results
        logger.info("Step 7: Displaying results")
        print_backtest_results(metrics, trades, df)

        return metrics, trades, df
        
    finally:
        pipeline.close()

def print_backtest_results(metrics: dict, trades: list, df: pd.DataFrame):
    """Pretty-print the backtest results"""
    print("\n" + "="*80)
    print("COMMANDER STRATEGY BACKTEST RESULTS")
    print("="*80)

    if 'error' in metrics:
        print("[NO TRADES] No trades executed: {}".format(metrics['error']))
        print("\nPossible reasons:")
        print("- No signals generated (check regime conditions)")
        print("- Data doesn't meet signal criteria")
        print("- Indicators not calculated properly")
        return

    # Summary metrics
    summary = metrics['summary']
    print("📊 SUMMARY METRICS")
    print(f"   Total Trades:     {summary['total_trades']}")
    print(f"   Win Rate:         {summary['win_rate']:.1%}")
    print(f"   Total P&L:        ${summary['total_pnl']:.2f}")
    print(f"   Profit Factor:    {summary['profit_factor']:.2f}")
    print(f"   Max Drawdown:     ${summary['max_drawdown']:.2f}")
    print(f"   Avg Win:          ${summary['avg_win']:.2f}")
    print(f"   Avg Loss:         ${summary['avg_loss']:.2f}")
    print(f"   Payoff Ratio:     {summary['payoff_ratio']:.2f}")

    # Signal performance
    if 'by_signal_type' in metrics and metrics['by_signal_type']:
        print("\n🎯 PERFORMANCE BY SIGNAL TYPE")
        for signal_type, data in metrics['by_signal_type'].items():
            print(f"   {signal_type.upper()}:")
            print(f"      Trades: {data['trades']}, Win Rate: {data['win_rate']:.1%}, P&L: ${data['total_pnl']:.2f}")

    # Exit reason analysis
    if 'by_exit_reason' in metrics and metrics['by_exit_reason']:
        print("\n⏹️  EXIT REASON ANALYSIS")
        for exit_reason, data in metrics['by_exit_reason'].items():
            print(f"   {exit_reason.upper()}:")
            print(f"      Trades: {data['trades']}, Win Rate: {data['win_rate']:.1%}, P&L: ${data['total_pnl']:.2f}")

    # Data quality check
    print("\n📈 DATA QUALITY CHECK")
    print(f"   Total bars processed: {len(df)}")
    print(f"   Signals generated: {df['entry_signal'].abs().sum()}")
    
    # Check for regime column (different names in different systems)
    regime_col = 'regime' if 'regime' in df.columns else 'regime_state' if 'regime_state' in df.columns else 'market_regime'
    if regime_col in df.columns:
        print(f"   Regime distribution: {df[regime_col].value_counts().to_dict()}")
    else:
        print("   Regime distribution: N/A")

    # Sample trades
    if trades:
        print("\n💼 SAMPLE TRADES (First 5)")
        for i, trade in enumerate(trades[:5]):
            print(f"   Trade {i+1}: {trade['signal_type']} {trade['direction']} "
                  f"@ ${trade['entry_price']:.2f}, Exit: ${trade['exit_price']:.2f}, "
                  f"P&L: ${trade['pnl']:.2f} ({trade['exit_reason']})")

    print("\n" + "="*80)

def main():
    """Main execution function"""
    print("Commander Strategy Backtesting Script")
    print("====================================")

    # Run the backtest
    try:
        metrics, trades, df = run_complete_backtest('$SPX', days=5)
        logger.info("Backtest completed successfully!")

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise

if __name__ == '__main__':
    main()