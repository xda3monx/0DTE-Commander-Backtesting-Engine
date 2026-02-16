#!/usr/bin/env python3
"""
Test Commander Strategy with Real VIX Data
==========================================

Tests the Commander strategy using real SPY and VIX data
from the data pipeline with VIX-based veto conditions.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from BacktestingEngine import CommanderStrategy
from phase1_data_pipeline import DataPipeline
from vectorized_indicators import VectorizedIndicators
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_commander_with_vix_data():
    """
    Test Commander strategy with real VIX data integration.
    """
    logger.info("Testing Commander Strategy with Real VIX Data")
    logger.info("="*60)

    # Step 1: Load real data
    logger.info("Step 1: Loading real SPY/VIX data")
    pipeline = DataPipeline()

    try:
        df = pipeline.run_phase1_pipeline()
        logger.info(f"Loaded {len(df)} bars of real data")

        # Step 2: Calculate indicators using VectorizedIndicators
        logger.info("Step 2: Calculating indicators with VIX veto conditions")
        indicators = VectorizedIndicators(df)

        # Calculate all indicators
        indicators.calculate_drift_indicators()
        indicators.calculate_velocity_physics()
        indicators.calculate_regime_states()
        indicators.calculate_volume_physics()
        indicators.calculate_veto_conditions()
        indicators.calculate_market_bias()

        df_with_indicators = indicators.df
        logger.info(f"Calculated {len(df_with_indicators.columns)} indicator columns")

        # Step 3: Initialize Commander strategy
        logger.info("Step 3: Initializing Commander strategy")
        strategy = CommanderStrategy(risk_per_trade=0.005, max_daily_loss=0.02)

        # Step 4: Generate signals
        logger.info("Step 4: Generating Commander signals")
        df_signals = strategy.generate_commander_signals(df_with_indicators)

        # Count signals
        entry_signals = df_signals['entry_signal'].value_counts()
        logger.info(f"Generated signals: {dict(entry_signals)}")

        # Check veto effectiveness
        veto_cols = [col for col in df_signals.columns if 'veto' in col.lower()]
        if veto_cols:
            logger.info("Veto condition summary:")
            for col in veto_cols:
                count = df_signals[col].sum()
                logger.info(f"  {col}: {int(count)} occurrences")

        # Step 5: Simulate trades (sample)
        logger.info("Step 5: Simulating sample trades")
        trades = strategy.simulate_trades(df_signals, capital=10000.0)

        # Step 6: Calculate performance
        logger.info("Step 6: Calculating performance metrics")
        metrics = strategy.calculate_performance_metrics(trades)

        # Step 7: Display results
        logger.info("Step 7: Results Summary")
        print("\n" + "="*60)
        print("COMMANDER STRATEGY TEST WITH REAL VIX DATA")
        print("="*60)
        print(f"Data Period: {df_signals.index.min()} to {df_signals.index.max()}")
        print(f"Total Bars: {len(df_signals)}")
        print(f"Signals Generated: {len(trades)}")
        print(f"Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
        print(f"Total P&L: ${metrics.get('total_pnl', 0):.2f}")
        print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")

        if trades:
            print(f"\nSample Trades:")
            for i, trade in enumerate(trades[:5]):
                print(f"  {i+1}. {trade['direction']} @ {trade['entry_price']:.2f}, "
                      f"Exit: {trade['exit_price']:.2f}, P&L: ${trade['pnl']:.2f}")

        return metrics, trades, df_signals

    finally:
        pipeline.close()

if __name__ == "__main__":
    test_commander_with_vix_data()