"""
Vectorized Indicator Calculations for 0DTE Commander System
==========================================================

This script demonstrates how to translate ThinkScript logic into vectorized
Pandas operations for high-performance backtesting.

Key Concepts:
- Replace loops with vectorized operations
- Use pandas-ta, pandas .ewm(), .rolling() for technical indicators
- Chain operations for complex calculations
- Leverage NumPy for mathematical operations
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorizedIndicators:
    """
    Vectorized implementation of 0DTE Commander indicators.
    Translates ThinkScript logic to Pandas/NumPy operations.
    """

    def __init__(self, df: pd.DataFrame, primary_symbol: str = 'spy'):
        """
        Initialize with OHLCV DataFrame.

        For merged multi-symbol DataFrames, specify primary_symbol to extract its data.
        Expected columns for primary symbol: ['timestamp', 'datetime_utc', f'{primary_symbol}_open', etc.]
        """
        self.df = df.copy()
        self.primary_symbol = primary_symbol
        
        # Handle case where datetime_utc might be the index
        if 'datetime_utc' not in self.df.columns and isinstance(self.df.index, pd.DatetimeIndex):
            self.df['datetime_utc'] = self.df.index
        elif 'datetime_utc' in self.df.columns:
            self.df['datetime_utc'] = pd.to_datetime(self.df['datetime_utc'])
        
        self.df = self.df.set_index('datetime_utc')

        # Check if this is a merged DataFrame with symbol-prefixed columns
        spy_cols = [f'{primary_symbol}_open', f'{primary_symbol}_high', f'{primary_symbol}_low', 
                   f'{primary_symbol}_close', f'{primary_symbol}_volume']
        
        if all(col in self.df.columns for col in spy_cols):
            # Extract primary symbol columns and rename to standard format
            rename_dict = {
                f'{primary_symbol}_open': 'open',
                f'{primary_symbol}_high': 'high',
                f'{primary_symbol}_low': 'low',
                f'{primary_symbol}_close': 'close',
                f'{primary_symbol}_volume': 'volume'
            }
            self.df = self.df.rename(columns=rename_dict)
            logger.info(f"Extracted {primary_symbol.upper()} data from merged DataFrame")
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in self.df.columns for col in required_cols):
            raise ValueError(f"DataFrame missing required columns: {required_cols}")

        logger.info(f"Initialized with {len(self.df)} bars")

    def calculate_drift_indicators(self) -> pd.DataFrame:
        """
        Calculate Drift indicators from 0DTE Commander (Fast/Slow EMAs).

        ThinkScript equivalent:
        def lengthFast = 8;
        def lengthSlow = 24;
        def rawFast = if (Highest(high, lengthFast) - Lowest(low, lengthFast)) != 0
                      then -100 * (Highest(high, lengthFast) - close) / (Highest(high, lengthFast) - Lowest(low, lengthFast))
                      else 0;
        def rawSlow = similar for lengthSlow;
        plot FastLine = ExpAverage(rawFast, 3);
        plot SlowLine = ExpAverage(rawSlow, 2);
        """
        logger.info("Calculating Drift indicators...")

        # Vectorized highest/lowest calculations
        # Rolling max/min over windows
        fast_window = 8
        slow_window = 24

        # Calculate rolling highest and lowest
        self.df['fast_high'] = self.df['high'].rolling(window=fast_window).max()
        self.df['fast_low'] = self.df['low'].rolling(window=fast_window).min()
        self.df['slow_high'] = self.df['high'].rolling(window=slow_window).max()
        self.df['slow_low'] = self.df['low'].rolling(window=slow_window).min()

        # Calculate raw oscillator values (vectorized)
        # Avoid division by zero
        fast_range = self.df['fast_high'] - self.df['fast_low']
        fast_range = fast_range.replace(0, np.nan)  # Avoid div by zero

        slow_range = self.df['slow_high'] - self.df['slow_low']
        slow_range = slow_range.replace(0, np.nan)

        # Raw oscillator: -100 * (highest - close) / range
        self.df['raw_fast'] = np.where(
            fast_range.notna(),
            -100 * (self.df['fast_high'] - self.df['close']) / fast_range,
            0
        )

        self.df['raw_slow'] = np.where(
            slow_range.notna(),
            -100 * (self.df['slow_high'] - self.df['close']) / slow_range,
            0
        )

        # Smooth with exponential moving averages
        self.df['fast_line'] = self.df['raw_fast'].ewm(span=3, adjust=False).mean()
        self.df['slow_line'] = self.df['raw_slow'].ewm(span=2, adjust=False).mean()

        # Drift spread and velocity
        self.df['drift_spread'] = self.df['fast_line'] - self.df['slow_line']
        self.df['drift_velocity'] = self.df['drift_spread'].diff()
        
        # Alias for backward compatibility with BacktestingEngine
        self.df['velocity'] = self.df['drift_velocity']

        logger.info("Drift indicators calculated")
        return self.df

    def calculate_velocity_physics(self) -> pd.DataFrame:
        """
        Calculate Velocity and Theta Physics from Commander.

        Includes ATR normalization, MVV thresholds, Theta Burn detection.
        """
        logger.info("Calculating Velocity Physics...")

        # ATR calculation (manual implementation)
        high_low = self.df['high'] - self.df['low']
        high_close = (self.df['high'] - self.df['close'].shift(1)).abs()
        low_close = (self.df['low'] - self.df['close'].shift(1)).abs()

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df['atr'] = tr.rolling(14).mean()

        # Alias ATR as volatility for BacktestingEngine compatibility
        self.df['volatility'] = self.df['atr']
        self.df['volatility_pct'] = (self.df['volatility'] / self.df['close']) * 100

        # ADX calculation (simplified implementation)
        # Directional Movement
        dm_plus = np.where(
            (self.df['high'] - self.df['high'].shift(1)) > (self.df['low'].shift(1) - self.df['low']),
            np.maximum(self.df['high'] - self.df['high'].shift(1), 0),
            0
        )
        dm_minus = np.where(
            (self.df['low'].shift(1) - self.df['low']) > (self.df['high'] - self.df['high'].shift(1)),
            np.maximum(self.df['low'].shift(1) - self.df['low'], 0),
            0
        )

        # Smoothed DM
        dm_plus_smooth = pd.Series(dm_plus, index=self.df.index).rolling(14).mean()
        dm_minus_smooth = pd.Series(dm_minus, index=self.df.index).rolling(14).mean()

        # DI values
        di_plus = 100 * (dm_plus_smooth / (self.df['atr'] + 0.01))
        di_minus = 100 * (dm_minus_smooth / (self.df['atr'] + 0.01))

        # DX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 0.01)
        self.df['adx'] = dx.rolling(14).mean()

        # IsChop (ADX-based choppiness filter)
        self.df['is_chop'] = self.df['adx'] < 15

        # Normalize velocity by ATR
        self.df['normalized_velocity'] = self.df['drift_velocity'] / (self.df['atr'] + 0.01)

        # Time-based MVV thresholds (vectorized)
        # Extract hour from datetime index
        hour = self.df.index.hour

        self.df['base_mvv'] = np.select(
            [
                hour >= 15,  # 3:00 PM
                hour >= 14,  # 2:00 PM
                hour >= 13,  # 1:00 PM
            ],
            [
                1.25,  # Strong push required late day
                0.75,  # Solid trend mid-afternoon
                0.40,  # Moderate life early afternoon
            ],
            default=0.0  # Morning: no threshold
        )

        # VIX proxy (placeholder - would need VIX data)
        # For now, use ATR as volatility proxy
        self.df['vix_proxy'] = self.df['atr'].rolling(20).mean()

        # Vega Governor (simplified)
        # Slope of VIX proxy
        self.df['vix_slope'] = self.df['vix_proxy'].diff()

        # Raw Vega Factor
        self.df['raw_vega_factor'] = np.where(
            self.df['vix_slope'] < -0.01, 1.5,  # Headwind
            np.where(self.df['vix_slope'] > 0.01, 0.8, 1.0)  # Tailwind
        )

        # Asymmetric smoothing (vectorized)
        # Fast attack for higher hurdles, slow decay for lower
        self.df['vega_factor'] = self.df['raw_vega_factor'].copy()

        # Apply asymmetric smoothing
        for i in range(1, len(self.df)):
            if self.df.iloc[i]['raw_vega_factor'] > self.df.iloc[i-1]['vega_factor']:
                self.df.iloc[i, self.df.columns.get_loc('vega_factor')] = self.df.iloc[i]['raw_vega_factor']
            else:
                # Decay toward normal (20% per bar)
                prev = self.df.iloc[i-1]['vega_factor']
                self.df.iloc[i, self.df.columns.get_loc('vega_factor')] = prev * 0.8 + self.df.iloc[i]['raw_vega_factor'] * 0.2

        # Final MVV threshold
        self.df['mvv_threshold'] = self.df['base_mvv'] * self.df['vega_factor']

        # Theta Burn condition
        self.df['is_theta_burn'] = (
            (self.df.index.hour >= 13) &  # After 1 PM
            (self.df['drift_spread'] > 20) &  # Effective drift min (adaptive)
            (self.df['normalized_velocity'] < self.df['mvv_threshold'])
        )

        logger.info("Velocity Physics calculated")
        return self.df

    def calculate_regime_states(self) -> pd.DataFrame:
        """
        Calculate regime states from Commander logic.

        Includes Launch/Snapback detection, Peg identification with ADX/IsChop filtering.
        """
        logger.info("Calculating Regime States...")

        # Effective drift minimum (adaptive based on volatility)
        # Higher in volatile conditions, lower in calm conditions
        vix_proxy = self.df['vix_proxy'] if 'vix_proxy' in self.df.columns else self.df['atr']
        self.df['effective_drift_min'] = np.where(
            vix_proxy > vix_proxy.quantile(0.8), 25,  # High volatility: lowered from 35
            np.where(vix_proxy < vix_proxy.quantile(0.2), 10, 15)  # Low volatility: lowered from 15/25
        )

        # Shock events (using effective drift min)
        self.df['shock_event'] = self.df['drift_spread'] > self.df['effective_drift_min'] * 1.5

        # Launch/Snapback detection
        launch_velocity = 5
        snapback_velocity = -9

        self.df['is_launch'] = (
            (self.df['drift_spread'].shift(1) < self.df['effective_drift_min']) &  # Was below drift min
            (self.df['drift_velocity'] > launch_velocity)
        )

        self.df['is_snapback'] = (
            (self.df['drift_spread'].shift(1) > self.df['effective_drift_min'] * 1.5) &
            (self.df['drift_velocity'] < snapback_velocity)
        )

        # Peg detection with ADX/IsChop filtering
        self.df['bull_peg'] = (
            (self.df['fast_line'] > -20) &
            (self.df['slow_line'] > -20) &
            (~self.df['is_chop']) &  # Not choppy
            (self.df['drift_spread'] > self.df['effective_drift_min'])
        )

        self.df['bear_peg'] = (
            (self.df['fast_line'] < -80) &
            (self.df['slow_line'] < -80) &
            (~self.df['is_chop']) &  # Not choppy
            (self.df['drift_spread'] > self.df['effective_drift_min'])
        )

        # Regime state mapping (similar to Debug_State in ThinkScript)
        conditions = [
            self.df['is_snapback'],  # 9: Snapback
            self.df['is_launch'],    # 10: Launch
            self.df['bull_peg'],     # 3: Bull Peg
            self.df['bear_peg'],     # 4: Bear Peg
            self.df['drift_spread'] < self.df['effective_drift_min'],  # 2: Noise
        ]

        choices = [9, 10, 3, 4, 2]

        self.df['regime_state'] = np.select(conditions, choices, default=5)  # 5: Trend

        # Map regime_state integer to string description for BacktestingEngine compatibility
        regime_map = {
            9: 'snapback',
            10: 'launch',
            3: 'bull_peg',
            4: 'bear_peg',
            2: 'noise',
            5: 'trend'
        }
        
        # Create 'regime' column using map
        self.df['regime'] = self.df['regime_state'].map(regime_map)
        self.df['regime'] = self.df['regime'].fillna('unknown')

        # Log regime distribution to help debug missing signals
        regime_counts = self.df['regime_state'].value_counts().to_dict()
        logger.info(f"Regime state distribution: {regime_counts}")

        logger.info("Regime states calculated")
        return self.df

    def calculate_volume_physics(self) -> pd.DataFrame:
        """
        Calculate Volume Physics from Unified Internals.

        Includes VWAP, volume ratios, climax detection.
        """
        logger.info("Calculating Volume Physics...")

        # VWAP calculation (vectorized)
        # VWAP = cumulative(price * volume) / cumulative(volume)
        typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        self.df['price_volume'] = typical_price * self.df['volume']

        # Cumulative sums for VWAP
        self.df['cum_volume'] = self.df['volume'].cumsum()
        self.df['cum_price_volume'] = self.df['price_volume'].cumsum()

        self.df['vwap'] = self.df['cum_price_volume'] / self.df['cum_volume']

        # VWAP bands (2 standard deviations)
        # Calculate rolling std of (close - vwap)
        self.df['vwap_diff'] = self.df['close'] - self.df['vwap']
        self.df['vwap_std'] = self.df['vwap_diff'].rolling(20).std()

        self.df['vwap_upper'] = self.df['vwap'] + (2 * self.df['vwap_std'])
        self.df['vwap_lower'] = self.df['vwap'] - (2 * self.df['vwap_std'])

        # Volume ratios (relative to recent average)
        volume_window = 50
        self.df['avg_volume'] = self.df['volume'].rolling(volume_window).mean()
        self.df['volume_ratio'] = self.df['volume'] / self.df['avg_volume']

        # Volume states
        self.df['is_high_vol'] = self.df['volume_ratio'] > 1.1
        self.df['is_climax_vol'] = self.df['volume_ratio'] > 2.5

        # Sniper signals (from Debugger)
        self.df['sniper_buy'] = (
            (self.df['low'] <= self.df['vwap_lower']) &
            (self.df['close'] > self.df['vwap_lower']) &
            self.df['is_high_vol'] &
            (abs(self.df['drift_velocity']) < 8)
        )

        self.df['sniper_sell'] = (
            (self.df['high'] >= self.df['vwap_upper']) &
            (self.df['close'] < self.df['vwap_upper']) &
            self.df['is_high_vol'] &
            (abs(self.df['drift_velocity']) < 8)
        )

        logger.info("Volume Physics calculated")
        return self.df

    def calculate_veto_conditions(self) -> pd.DataFrame:
        """
        Calculate veto conditions using Commander's IV Engine with actual VIX/VIX1D data.

        Direct port of ThinkScript IV Crush logic:
        - VIX Drop Veto: >12% drop from open
        - IV Crush: VIX in bottom 5% of session range + range >1.5pts + before 3PM
        - PM Expansion: After 1PM, VIX >115% of session mean
        - Flash Crash: Before 3PM, VIX <85% of session mean
        """
        logger.info("Calculating Veto Conditions...")

        # Check for VIX data availability
        has_vix = 'vix_close' in self.df.columns
        has_vix1d = 'vix1d_close' in self.df.columns

        if has_vix:
            # Current VIX close
            self.df['vix_current'] = self.df['vix_close']

            # Session-level calculations (grouped by date)
            # First, create a date column for grouping
            self.df['date'] = self.df.index.date
            
            self.df['vix_session_open'] = self.df.groupby('date')['vix_current'].transform('first')
            self.df['vix_session_high'] = self.df.groupby('date')['vix_current'].transform('max')
            self.df['vix_session_low'] = self.df.groupby('date')['vix_current'].transform('min')
            self.df['vix_session_mean'] = self.df.groupby('date')['vix_current'].transform(lambda x: x.expanding().mean())
            self.df['vix_range'] = self.df['vix_session_high'] - self.df['vix_session_low']

            # Intraday percentile within session (0-1 scale)
            self.df['vix_percentile'] = self.df.groupby('date')['vix_current'].transform(
                lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else 0.5
            )

            # VIX Drop Veto: >12% drop from session open (ThinkScript: vixDropPercent > 0.12)
            self.df['vix_drop_percent'] = (self.df['vix_session_open'] - self.df['vix_current']) / self.df['vix_session_open']
            self.df['veto_vix_drop'] = self.df['vix_drop_percent'] > 0.12

            # IV Crush Veto: Bottom 5% of session range + range >1.5pts + before 3PM
            self.df['veto_percentile'] = (
                (self.df['vix_percentile'] < 0.05) &
                (self.df['vix_range'] > 1.5) &
                (self.df.index.hour < 15)
            )

            # PM Expansion Veto: After 1PM, VIX >115% of session mean
            self.df['veto_pm_expansion'] = (
                (self.df.index.hour >= 13) &
                (self.df['vix_current'] > self.df['vix_session_mean'] * 1.15)
            )

            # Flash Crash Veto: Before 3PM, VIX <85% of session mean
            self.df['veto_flash_crash'] = (
                (self.df.index.hour < 15) &
                (self.df['vix_current'] < self.df['vix_session_mean'] * 0.85)
            )

            # Combined IV Crush veto (any of the above conditions)
            self.df['veto_iv_crush'] = (
                self.df['veto_vix_drop'] |
                self.df['veto_percentile'] |
                self.df['veto_pm_expansion'] |
                self.df['veto_flash_crash']
            )

            # Terminal Compression Warning (Chronos): Late day VIX below session mean
            self.df['warn_terminal_compress'] = (
                (self.df.index.hour >= 15) &
                (self.df['vix_current'] < self.df['vix_session_mean'])
            )

        else:
            # Fallback to ATR-based veto (less accurate)
            logger.warning("VIX data not available, using ATR-based veto conditions")
            self.df['veto_iv_crush'] = self.df['atr'] < self.df['atr'].rolling(50).quantile(0.1)
            self.df['warn_terminal_compress'] = False

        # VIX1D comparison (if available) - for additional context
        if has_vix1d:
            # Previous day VIX close (forward fill to align with intraday data)
            self.df['vix_prev_close'] = self.df.groupby(self.df.index.date)['vix1d_close'].transform('first').ffill()

            # VIX vs previous close (additional context for regime detection)
            self.df['vix_vs_prev'] = (self.df['vix_current'] - self.df['vix_prev_close']) / self.df['vix_prev_close']

        # Internals-based veto conditions
        self._calculate_internals_veto()

        logger.info("Veto conditions calculated")
        return self.df
    
    def _calculate_internals_veto(self) -> None:
        """
        Calculate Internals-based veto conditions using UVOL/DVOL/TRIN/TICK/Mag7.
        
        These conditions prevent trading when market internals show unhealthy structure:
        - UVOL/DVOL Imbalance: Extreme volume bias (>80/20 split = hidden weakness)
        - TRIN Extreme: Arms Index out of normal range (>2 or <0.5 = unsustainable)
        - TICK Extreme: Too many/few stocks advancing (extreme breadth)
        - Mag7 Concentration: Excessive tech mega-cap concentration
        """
        # Check for internals data availability (multiple column name variants)
        has_uvol = 'uvol_close' in self.df.columns or '$uvol_close' in self.df.columns
        has_dvol = 'dvol_close' in self.df.columns or '$dvol_close' in self.df.columns
        has_trin = 'trin_close' in self.df.columns or '$trin_close' in self.df.columns
        has_tick = 'tick_close' in self.df.columns or '$tick_close' in self.df.columns
        has_mag7 = 'mag7_close' in self.df.columns
        
        if not has_uvol or not has_dvol:
            logger.warning("UVOL/DVOL data not available, skipping internals veto")
            self.df['veto_internals'] = False
            return
        
        # Normalize column names
        uvol_col = '$uvol_close' if '$uvol_close' in self.df.columns else 'uvol_close'
        dvol_col = '$dvol_close' if '$dvol_close' in self.df.columns else 'dvol_close'
        trin_col = '$trin_close' if '$trin_close' in self.df.columns else 'trin_close'
        tick_col = '$tick_close' if '$tick_close' in self.df.columns else 'tick_close'
        mag7_col = 'mag7_close'
        
        # UVOL/DVOL ratio analysis
        # Ratio > 3 = strong bull, ratio < 0.33 = strong bear
        # Extreme ratios (>4 or <0.25) suggest market fatigue / hidden weakness
        self.df['uvol_dvol_ratio'] = self.df[uvol_col] / (self.df[dvol_col] + 1)  # Avoid div by 0
        
        uvol_dvol_extreme = (
            (self.df['uvol_dvol_ratio'] > 4.0) |  # Excessive bull bias
            (self.df['uvol_dvol_ratio'] < 0.25)   # Excessive bear bias
        )
        
        self.df['veto_uvol_dvol'] = uvol_dvol_extreme
        
        # TRIN analysis (Arms Index)
        # Normal range: 0.5 - 2.0
        # > 2.0: More declines than advances (breadth weakness)
        # < 0.5: Extreme rally (unsustainable)
        if has_trin:
            trin_extreme = (
                (self.df[trin_col] > 2.5) |  # Breadth very weak
                (self.df[trin_col] < 0.4)    # Rally too strong (bubble risk)
            )
            self.df['veto_trin_extreme'] = trin_extreme
        else:
            self.df['veto_trin_extreme'] = False
        
        # TICK analysis (NYSE Advance/Decline)
        # Based on session extremes (percentile relative to session)
        if has_tick:
            session_tick_high = self.df.groupby(self.df.index.date)[tick_col].transform('max')
            session_tick_low = self.df.groupby(self.df.index.date)[tick_col].transform('min')
            
            tick_extreme_low = self.df[tick_col] < session_tick_low * 0.8  # Bottom 20%
            tick_extreme_high = self.df[tick_col] > session_tick_high * 0.8  # Top 20%
            
            self.df['veto_tick_extreme'] = tick_extreme_low | tick_extreme_high
        else:
            self.df['veto_tick_extreme'] = False
        
        # Mag7 concentration risk
        # High concentration (Mag7 outperforming by >2% vs market average) = risk
        if has_mag7:
            # Simplified: If Mag7 volatility exceeds 2x market volatility = concentration risk
            mag7_vol = self.df[mag7_col].pct_change().rolling(20).std()
            market_vol = self.df['close'].pct_change().rolling(20).std()
            
            mag7_concentrated = (mag7_vol > market_vol * 2.0)
            
            self.df['veto_mag7_concentration'] = mag7_concentrated
        else:
            self.df['veto_mag7_concentration'] = False
        
        # Combined internals veto: ANY unhealthy internal condition
        self.df['veto_internals'] = (
            self.df['veto_uvol_dvol'] |
            self.df['veto_trin_extreme'] |
            self.df['veto_tick_extreme'] |
            self.df['veto_mag7_concentration']
        )
        
        # Log statistics
        internals_veto_count = self.df['veto_internals'].sum()
        logger.info(f"Internals veto periods: {internals_veto_count} ({100*internals_veto_count/len(self.df):.1f}%)")


    def calculate_market_bias(self) -> pd.DataFrame:
        """
        Calculate market bias from regime states and tactical alignment.
        """
        logger.info("Calculating Market Bias...")
        
        # Market Bias from regime states
        self.df['market_bias'] = np.select(
            [
                self.df['bull_peg'],
                self.df['bear_peg'],
                self.df['is_launch']
            ],
            [1, -1, 1],  # Bullish, Bearish, Bullish (launch)
            default=0    # Neutral
        )
        
        # Tactical Alignment (example for LONG_CALL context)
        trade_context = "LONG_CALL"  # This could be parameterized
        
        if trade_context == "LONG_CALL":
            self.df['context_status'] = np.where(
                self.df['market_bias'] == 1, 1,  # Aligned
                np.where(self.df['market_bias'] == -1, -1, 0)  # Misaligned
            )
        elif trade_context == "LONG_PUT":
            self.df['context_status'] = np.where(
                self.df['market_bias'] == -1, 1,  # Aligned
                np.where(self.df['market_bias'] == 1, -1, 0)  # Misaligned
            )
        else:  # FLAT
            self.df['context_status'] = 0
        
        # Velocity coloring (from Commander)
        self.df['velocity_color'] = np.select(
            [
                (self.df['context_status'] == -1),
                (self.df['context_status'] == 1) & self.df['is_snapback'],
                (self.df['context_status'] == 1) & self.df['is_theta_burn'],
                (self.df['context_status'] == 1) & (self.df['drift_velocity'] < -5),
                (self.df['context_status'] == 1),
                self.df['is_snapback'],
                self.df['is_theta_burn'],
                self.df['is_launch']
            ],
            [
                "RED",      # Misaligned
                "YELLOW",   # Snapback in alignment
                "ORANGE",   # Theta burn in alignment
                "ORANGE",   # Decel in alignment
                "GREEN",    # Good momentum in alignment
                "MAGENTA",  # Snapback neutral
                "ORANGE",   # Theta burn neutral
                "GREEN"     # Launch neutral
            ],
            default="GRAY"
        )
        
        logger.info("Market bias calculated")
        return self.df

    def calculate_unified_internals(self) -> pd.DataFrame:
        """
        Calculate Unified Internals from SPX Unified Internals script.
        
        Includes breadth, tick, Mag7, volume flow, regime classification.
        """
        logger.info("Calculating Unified Internals...")
        
        # Placeholder for breadth data (would need $UVOLSP, $DVOLSP, etc.)
        # For now, simulate with volume-based metrics
        
        # Volume Flow (from Unified Internals)
        if 'volume' in self.df.columns:
            # Adaptive volume thresholds
            self.df['uvol_raw'] = self.df['volume'] * (self.df['close'] > self.df['open']).astype(int)  # Up volume proxy
            self.df['dvol_raw'] = self.df['volume'] * (self.df['close'] < self.df['open']).astype(int)  # Down volume proxy
            
            self.df['uvol_velocity'] = self.df['uvol_raw'] - self.df['uvol_raw'].shift(1)
            self.df['dvol_velocity'] = self.df['dvol_raw'] - self.df['dvol_raw'].shift(1)
            
            # Adaptive thresholds
            self.df['uvol_std'] = self.df['uvol_velocity'].rolling(20).std()
            self.df['dvol_std'] = self.df['dvol_velocity'].rolling(20).std()
            
            # Flow signals
            self.df['is_aggressive_buy'] = (self.df.index.hour >= 9) & (self.df.index.hour <= 16) & (self.df['uvol_velocity'] > self.df['uvol_std'] * 1.5)
            self.df['is_seller_exhaustion'] = (self.df.index.hour >= 9) & (self.df.index.hour <= 16) & (self.df['dvol_velocity'] < -self.df['dvol_std'] * 1.5)
            self.df['is_aggressive_sell'] = (self.df.index.hour >= 9) & (self.df.index.hour <= 16) & (self.df['dvol_velocity'] > self.df['dvol_std'] * 1.5)
            self.df['is_buyer_exhaustion'] = (self.df.index.hour >= 9) & (self.df.index.hour <= 16) & (self.df['uvol_velocity'] < -self.df['uvol_std'] * 1.5)
            
            # Sustained flow
            self.df['aggressive_buy_streak'] = self.df['is_aggressive_buy'].groupby((~self.df['is_aggressive_buy']).cumsum()).cumsum()
            self.df['aggressive_sell_streak'] = self.df['is_aggressive_sell'].groupby((~self.df['is_aggressive_sell']).cumsum()).cumsum()
            
            self.df['is_sustained_buy'] = self.df['aggressive_buy_streak'] >= 3
            self.df['is_sustained_sell'] = self.df['aggressive_sell_streak'] >= 3
            
            # Price-Volume Divergence
            self.df['price_change'] = self.df['close'] - self.df['close'].shift(1)
            self.df['is_price_up'] = self.df['price_change'] > 0
            self.df['is_price_down'] = self.df['price_change'] < 0
            
            self.df['bullish_absorption'] = self.df['is_price_down'] & self.df['is_seller_exhaustion']
            self.df['bearish_distribution'] = self.df['is_price_up'] & self.df['is_buyer_exhaustion']
        
        # Mag7 Commander (placeholder - would need individual stock data)
        # For now, use SPY as proxy
        
        # Health Score (from Unified Internals)
        self.df['breadth_score'] = np.select(
            [self.df['volume_ratio'] > 1.5, self.df['volume_ratio'] > 1.1, self.df['volume_ratio'] < 0.9, self.df['volume_ratio'] < 0.7],
            [2, 1, -1, -2], default=0
        )
        
        self.df['trend_score'] = np.select(
            [self.df['drift_spread'] > 10, self.df['drift_spread'] > 5, self.df['drift_spread'] < -5, self.df['drift_spread'] < -10],
            [2, 1, -1, -2], default=0
        )
        
        self.df['health_score'] = self.df['breadth_score'] + self.df['trend_score']
        
        # Regime Classification
        self.df['market_regime'] = np.select(
            [self.df['health_score'] >= 3, self.df['health_score'] <= -3],
            [2, -2],  # RISK_OFF, RISK_ON
            default=0  # TRANSITIONAL
        )
        
        logger.info("Unified Internals calculated")
        return self.df

    def run_all_calculations(self) -> pd.DataFrame:
        """
        Run all indicator calculations in sequence.
        """
        logger.info("Running all vectorized calculations...")

        self.calculate_drift_indicators()
        self.calculate_velocity_physics()
        self.calculate_regime_states()
        self.calculate_volume_physics()
        self.calculate_veto_conditions()
        self.calculate_market_bias()
        self.calculate_unified_internals()

        # Final cleanup
        self.df = self.df.round(6)  # Reasonable precision

        logger.info(f"Completed calculations for {len(self.df)} bars")
        logger.info(f"Available indicators: {list(self.df.columns)}")

        return self.df

# Example usage
if __name__ == "__main__":
    # Load sample data - now with merged data including VIX
    import sqlite3
    
    # Load merged data from database
    conn = sqlite3.connect('backtesting_data.db')
    
    # Get SPY data (5m) and resample VIX/SPX to 5m for alignment
    spy_query = "SELECT * FROM historical_data WHERE symbol='SPY' ORDER BY timestamp"
    spy_df = pd.read_sql_query(spy_query, conn)
    spy_df['datetime_utc'] = pd.to_datetime(spy_df['datetime_utc'])
    spy_df = spy_df.set_index('datetime_utc')
    
    # Get VIX data (1h) and resample to 5m
    vix_query = "SELECT * FROM historical_data WHERE symbol='^VIX' ORDER BY timestamp"
    vix_df = pd.read_sql_query(vix_query, conn)
    vix_df['datetime_utc'] = pd.to_datetime(vix_df['datetime_utc'])
    vix_df = vix_df.set_index('datetime_utc')
    
    # Resample VIX to 5m by forward filling
    vix_5m = vix_df.resample('5min').ffill()
    
    # Merge SPY and VIX data
    merged_df = pd.merge(spy_df, vix_5m[['close']], left_index=True, right_index=True, how='left', suffixes=('', '_vix'))
    merged_df = merged_df.rename(columns={'close_vix': 'vix_close'})
    
    # Keep only necessary columns
    cols_to_keep = ['open', 'high', 'low', 'close', 'volume', 'vix_close']
    merged_df = merged_df[cols_to_keep]
    
    conn.close()
    
    # Initialize calculator with merged data
    calculator = VectorizedIndicators(merged_df)

    # Run calculations
    result_df = calculator.run_all_calculations()

    print("Sample of calculated indicators:")
    print(result_df[['close', 'fast_line', 'slow_line', 'drift_spread', 'regime_state', 'veto_iv_crush', 'market_bias']].tail(10).to_string())

    print(f"\nTotal bars processed: {len(result_df)}")
    print(f"Date range: {result_df.index.min()} to {result_df.index.max()}")
    print(f"VIX data available: {'vix_close' in result_df.columns}")