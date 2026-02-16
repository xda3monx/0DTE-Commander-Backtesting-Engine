"""
Data Validation Suite for 0DTE Commander Backtesting Engine
==========================================================

Comprehensive validation script that checks for common data quality issues,
ensures vectorized operations work correctly, and validates regime classification.

Checks performed:
- NaN propagation and data completeness
- Timezone alignment and continuity
- Regime distribution anomalies
- Vectorized calculation accuracy
- VWAP session reset validation
- ADX chop filter enforcement
- Data integrity and consistency
- Performance benchmarks
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataValidator:
    """
    Comprehensive data validation suite for the backtesting engine.
    Validates data quality, vectorized operations, and regime classification.
    """

    def __init__(self, db_path: str = 'backtesting_data.db'):
        """
        Initialize the validator.

        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.validation_results = {}

    def run_full_validation(self, symbol: str = 'SPY', sample_size: int = 1000) -> Dict:
        """
        Run the complete validation suite.

        Args:
            symbol: Trading symbol to validate
            sample_size: Number of bars to sample for intensive checks

        Returns:
            Dictionary with validation results
        """
        logger.info(f"🚀 Starting full validation suite for {symbol}")
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'checks': {}
        }

        # Load sample data
        df = self._load_sample_data(symbol, sample_size)
        if df.empty:
            logger.error("No data available for validation")
            return self.validation_results

        # Filter to only records with indicators
        indicator_cols = ['fast_line', 'slow_line', 'velocity', 'volatility']
        available_cols = [col for col in indicator_cols if col in df.columns]
        if available_cols:
            has_indicators = df[available_cols].notnull().any(axis=1)
            df_with_indicators = df[has_indicators].copy()
            if len(df_with_indicators) < len(df):
                logger.info(f"Filtered out {len(df) - len(df_with_indicators)} records without indicators")
            df = df_with_indicators

        # Run all validation checks
        checks = [
            self._validate_data_completeness,
            self._validate_nan_propagation,
            self._validate_timezone_alignment,
            self._validate_vectorized_calculations,
            self._validate_vwap_session_reset,      # NEW
            self._validate_adx_chop_filter,         # NEW
            self._validate_regime_distribution,
            self._validate_data_continuity,
            self._validate_calculation_accuracy,
            self._validate_performance_benchmarks
        ]

        for check_func in checks:
            check_name = check_func.__name__.replace('_validate_', '')
            logger.info(f"Running {check_name} check...")
            try:
                result = check_func(df)
                self.validation_results['checks'][check_name] = result

                if result.get('status') == 'skip':
                    status = "⏭️ SKIP"
                elif result['status'] == 'pass':
                    status = "✅ PASS"
                elif result['status'] == 'warning':
                    status = "⚠️ WARN"
                else:
                    status = "❌ FAIL"

                logger.info(f"{status}: {check_name}")
                if result['status'] in ['fail', 'warning', 'skip']:
                    if result['status'] == 'skip':
                        logger.info(f"Details: {result.get('details', 'N/A')}")
                    else:
                        logger.warning(f"Details: {result.get('details', 'N/A')}")
            except Exception as e:
                logger.error(f"Error in {check_name}: {e}")
                self.validation_results['checks'][check_name] = {
                    'status': 'error',
                    'details': str(e)
                }

        # Summary (exclude skipped checks)
        non_skip = [r for r in self.validation_results['checks'].values() 
                    if r.get('status') != 'skip']
        passed = sum(1 for r in non_skip if r.get('status') == 'pass')
        total = len(non_skip)

        self.validation_results['summary'] = {
            'total_checks': total,
            'passed': passed,
            'failed': total - passed,
            'success_rate': f"{passed/total*100:.1f}%" if total > 0 else "0%"
        }

        logger.info(f"📊 Validation complete: {passed}/{total} checks passed ({self.validation_results['summary']['success_rate']})")
        return self.validation_results

    def _load_sample_data(self, symbol: str, sample_size: int) -> pd.DataFrame:
        """Load sample data for validation."""
        try:
            query = f"""
            SELECT h.*, i.fast_line, i.slow_line, i.drift_spread, i.drift_velocity,
                   i.velocity, i.volatility, i.vwap, i.adx, i.regime, i.regime_state
            FROM historical_data h
            LEFT JOIN indicators i ON h.symbol = i.symbol
                AND h.timeframe = i.timeframe
                AND h.timestamp = i.timestamp
            WHERE h.symbol = '{symbol}'
            ORDER BY h.timestamp DESC
            LIMIT {sample_size}
            """
            df = pd.read_sql_query(query, self.conn)

            if not df.empty:
                # Convert datetime_utc to timezone-aware datetime
                if 'datetime_utc' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['datetime_utc']):
                    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], utc=True)
                elif 'datetime_utc' not in df.columns:
                    df['datetime_utc'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)

                df = df.sort_values('timestamp').reset_index(drop=True)
            return df
        except Exception as e:
            logger.error(f"Error loading sample data: {e}")
            return pd.DataFrame()

    def _validate_data_completeness(self, df: pd.DataFrame) -> Dict:
        """Check for missing data and completeness."""
        result = {'status': 'pass', 'details': {}}

        # Check for empty dataframe
        if df.empty:
            return {'status': 'fail', 'details': 'DataFrame is empty'}

        # Check required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            result['status'] = 'fail'
            result['details']['missing_columns'] = missing_cols

        # Check for completely null columns
        null_cols = []
        for col in df.columns:
            if df[col].isnull().all():
                null_cols.append(col)

        if null_cols:
            result['status'] = 'warning'
            result['details']['null_columns'] = null_cols

        # Check data size
        if len(df) < 100:
            result['status'] = 'warning'
            result['details']['small_dataset'] = f"Only {len(df)} rows available"

        result['details']['total_rows'] = len(df)
        result['details']['total_columns'] = len(df.columns)
        return result

    def _validate_nan_propagation(self, df: pd.DataFrame) -> Dict:
        """Check for NaN values and propagation patterns."""
        result = {'status': 'pass', 'details': {}}

        # Count NaN values per column
        nan_counts = df.isnull().sum()
        nan_cols = nan_counts[nan_counts > 0]

        if not nan_cols.empty:
            result['details']['nan_columns'] = nan_cols.to_dict()

        # Check for excessive NaN values (>20% of data)
        excessive_nan = nan_cols[nan_cols > len(df) * 0.20]
        if not excessive_nan.empty:
            result['status'] = 'fail'
            result['details']['excessive_nan'] = excessive_nan.to_dict()

        # Check for NaN propagation in calculated columns
        calc_cols = ['fast_line', 'slow_line', 'velocity', 'volatility', 'drift_spread']
        available_calc_cols = [col for col in calc_cols if col in df.columns]

        for col in available_calc_cols:
            # First N values should be NaN (due to lookback), but not excessively
            expected_nan = 50  # Maximum expected for 24-period calculations with buffer
            actual_nan = df[col].isnull().sum()
            if actual_nan > expected_nan:
                result['status'] = 'fail'
                result['details'][f'{col}_nan_anomaly'] = f"Expected ≤{expected_nan} NaN, got {actual_nan}"

        return result

    def _validate_timezone_alignment(self, df: pd.DataFrame) -> Dict:
        """Validate timezone handling and datetime consistency."""
        result = {'status': 'pass', 'details': {}}

        if 'datetime_utc' not in df.columns:
            return {'status': 'fail', 'details': 'datetime_utc column missing'}

        # Check timezone
        if not df['datetime_utc'].dt.tz:
            result['status'] = 'fail'
            result['details']['timezone'] = 'datetime_utc is not timezone-aware'

        # Check timestamp consistency (pandas datetime is in microseconds)
        timestamp_from_dt = (df['datetime_utc'].astype('int64') // 1_000_000).values  # microseconds to seconds
        timestamp_original = df['timestamp'].values

        # Allow 1-second tolerance for rounding
        mismatches = np.sum(np.abs(timestamp_from_dt - timestamp_original) > 1)
        if mismatches > 0:
            result['status'] = 'fail'
            result['details']['timestamp_mismatches'] = int(mismatches)

        # Check for duplicate timestamps
        duplicates = df['timestamp'].duplicated().sum()
        if duplicates > 0:
            result['status'] = 'warning'
            result['details']['duplicate_timestamps'] = int(duplicates)

        return result

    def _validate_vectorized_calculations(self, df: pd.DataFrame) -> Dict:
        """Validate that vectorized calculations work correctly."""
        result = {'status': 'pass', 'details': {}}

        # Test EMA calculations
        if len(df) >= 50:
            # Manual calculation for comparison
            test_data = df['close'].iloc[:50].values

            # Vectorized EMA
            ema_8_vectorized = pd.Series(test_data).ewm(span=8, adjust=False).mean().iloc[20:30]

            # Manual EMA calculation
            ema_8_manual = []
            multiplier = 2 / (8 + 1)
            ema_val = test_data[0]
            for i in range(len(test_data)):
                if i == 0:
                    ema_val = test_data[i]
                else:
                    ema_val = (test_data[i] * multiplier) + (ema_val * (1 - multiplier))
                if i >= 20:
                    ema_8_manual.append(ema_val)
                    if len(ema_8_manual) >= 10:
                        break

            # Compare results
            diff = np.abs(np.array(ema_8_manual) - ema_8_vectorized.values)
            max_diff = np.max(diff)
            if max_diff > 1e-6:  # Reasonable floating point tolerance
                result['status'] = 'fail'
                result['details']['ema_calculation_error'] = f"Max difference: {max_diff}"
            else:
                result['details']['ema_accuracy'] = f"Max difference: {max_diff:.2e}"

        return result

    def _validate_vwap_session_reset(self, df: pd.DataFrame) -> Dict:
        """Validate that VWAP resets at session boundaries (9:30 AM ET)."""
        result = {'status': 'pass', 'details': {}}

        if 'vwap' not in df.columns or len(df) < 50:  # Reduced requirement for testing
            return {'status': 'skip', 'details': 'Not enough VWAP data'}

        # Convert to ET timezone
        df_et = df.copy()
        df_et['datetime_et'] = df_et['datetime_utc'].dt.tz_convert('America/New_York')

        # Find session starts (9:30 AM)
        session_starts = df_et[
            (df_et['datetime_et'].dt.hour == 9) & 
            (df_et['datetime_et'].dt.minute == 30) &
            (df_et['datetime_et'].dt.dayofweek < 5)  # Monday-Friday
        ]

        if len(session_starts) < 2:
            return {'status': 'skip', 'details': 'Need multi-day data to validate session resets'}

        # Check VWAP values at session starts
        vwap_at_start = session_starts['vwap'].values
        close_at_start = session_starts['close'].values

        # VWAP should be close to current price at session start (indicating fresh reset)
        vwap_price_diff = np.abs((vwap_at_start - close_at_start) / close_at_start)

        # VWAP should be within 0.5% of price at session open
        max_drift = vwap_price_diff.max()
        if max_drift > 0.005:
            result['status'] = 'fail'
            result['details']['vwap_not_resetting'] = f"Max drift at session start: {max_drift:.2%}"
        else:
            result['details']['max_drift_at_reset'] = f"{max_drift:.3%}"

        result['details']['sessions_checked'] = len(session_starts)
        return result

    def _validate_adx_chop_filter(self, df: pd.DataFrame) -> Dict:
        """Validate that PEG signals respect ADX chop filter (ADX > 20)."""
        result = {'status': 'pass', 'details': {}}

        if 'regime' not in df.columns or 'adx' not in df.columns:
            return {'status': 'skip', 'details': 'Missing regime or ADX data'}

        # Find PEG signals
        peg_signals = df[df['regime'].isin(['bull_peg', 'bear_peg'])]

        if len(peg_signals) == 0:
            return {'status': 'skip', 'details': 'No PEG signals found in sample'}

        # Check that all PEG signals have ADX > 20
        choppy_pegs = peg_signals[peg_signals['adx'] < 20]

        if len(choppy_pegs) > 0:
            result['status'] = 'fail'
            result['details']['choppy_peg_signals'] = len(choppy_pegs)
            result['details']['min_adx_on_peg'] = float(peg_signals['adx'].min())
        else:
            result['details']['peg_signals_checked'] = len(peg_signals)
            result['details']['min_adx_on_peg'] = float(peg_signals['adx'].min())

        return result

    def _validate_regime_distribution(self, df: pd.DataFrame) -> Dict:
        """Check regime distribution for anomalies."""
        result = {'status': 'pass', 'details': {}}

        if 'regime' not in df.columns:
            return {'status': 'fail', 'details': 'regime column missing'}

        # Count regime distribution
        regime_counts = df['regime'].value_counts()
        total_regimes = len(df)
        result['details']['regime_distribution'] = regime_counts.to_dict()

        # Check for single regime dominance (>90%)
        if not regime_counts.empty:
            max_regime_pct = regime_counts.max() / total_regimes * 100
            if max_regime_pct > 90:
                result['status'] = 'warning'
                result['details']['regime_dominance'] = f"One regime dominates: {max_regime_pct:.1f}%"

        # Check for unknown regimes (FIXED: complete list matching BacktestingEngine.py)
        valid_regimes = [
            'halt_veto',        # 0
            'squeeze',          # 1
            'band_touch',       # 2
            'bull_peg',         # 3
            'bear_peg',         # 4
            'neutral',          # 5
            'bull_divergence',  # 7
            'bear_divergence',  # 8
            'bull_snapback',    # 9
            'launch',           # 10
            'bear_snapback',    # 11
            'sniper_buy',       # 13
            'sniper_sell',      # 14
            'unknown'           # Legacy/fallback
        ]

        unknown_regimes = [r for r in regime_counts.index if r not in valid_regimes and pd.notna(r)]
        if unknown_regimes:
            result['status'] = 'fail'
            result['details']['unknown_regimes'] = unknown_regimes

        return result

    def _validate_data_continuity(self, df: pd.DataFrame) -> Dict:
        """Check for data gaps and continuity issues."""
        result = {'status': 'pass', 'details': {}}

        if len(df) < 2:
            return result

        # Check for time gaps (assuming 5-minute data)
        time_diffs = df['timestamp'].diff().dropna()
        expected_diff = 300  # 5 minutes in seconds

        # Find gaps larger than expected (allow 50% tolerance)
        gaps = time_diffs[time_diffs > expected_diff * 1.5]
        if not gaps.empty:
            result['status'] = 'warning'
            result['details']['data_gaps'] = len(gaps)
            result['details']['largest_gap_minutes'] = int(gaps.max() / 60)

        # Check for price continuity (no extreme jumps)
        price_changes = df['close'].pct_change().abs()
        extreme_changes = price_changes[price_changes > 0.20]  # >20% change
        if not extreme_changes.empty:
            result['status'] = 'warning'
            result['details']['extreme_price_changes'] = len(extreme_changes)

        return result

    def _validate_calculation_accuracy(self, df: pd.DataFrame) -> Dict:
        """Validate calculation accuracy and consistency."""
        result = {'status': 'pass', 'details': {}}

        # Check Williams %R bounds (FIXED: use correct scale)
        if 'fast_line' in df.columns and 'slow_line' in df.columns and len(df) > 50:
            # Williams %R should be between -100 and 0
            fast_out_of_bounds = ((df['fast_line'] < -100) | (df['fast_line'] > 0)).sum()
            slow_out_of_bounds = ((df['slow_line'] < -100) | (df['slow_line'] > 0)).sum()

            if fast_out_of_bounds > 0 or slow_out_of_bounds > 0:
                result['status'] = 'fail'
                result['details']['williams_r_out_of_bounds'] = {
                    'fast_line': int(fast_out_of_bounds),
                    'slow_line': int(slow_out_of_bounds)
                }

            # Check drift spread is reasonable (max 100 for Williams %R)
            drift_strength = (df['fast_line'] - df['slow_line']).abs()
            unreasonable_drift = drift_strength > 100
            if unreasonable_drift.any():
                result['status'] = 'fail'
                result['details']['drift_out_of_bounds'] = int(unreasonable_drift.sum())

        # Check velocity bounds
        if 'velocity' in df.columns:
            velocity_extremes = df['velocity'].abs() > 1000  # Reasonable upper bound
            if velocity_extremes.any():
                result['status'] = 'warning'
                result['details']['extreme_velocity'] = int(velocity_extremes.sum())

        return result

    def _validate_performance_benchmarks(self, df: pd.DataFrame) -> Dict:
        """Benchmark performance of vectorized operations."""
        result = {'status': 'pass', 'details': {}}

        import time

        # Benchmark EMA calculation
        test_data = df['close'].values
        start_time = time.time()
        ema_8 = pd.Series(test_data).ewm(span=8, adjust=False).mean()
        ema_24 = pd.Series(test_data).ewm(span=24, adjust=False).mean()
        vectorized_time = time.time() - start_time

        result['details']['vectorized_time_ms'] = round(vectorized_time * 1000, 2)
        result['details']['data_points'] = len(test_data)
        result['details']['throughput'] = f"{len(test_data)/vectorized_time:.0f} points/sec"

        # Performance should be very fast (< 100ms for reasonable datasets)
        if vectorized_time > 0.1:
            result['status'] = 'warning'
            result['details']['slow_performance'] = f"{vectorized_time:.3f}s"

        return result

    def generate_report(self) -> str:
        """Generate a human-readable validation report."""
        if not self.validation_results:
            return "No validation results available. Run run_full_validation() first."

        report = []
        report.append("=" * 60)
        report.append("📊 0DTE COMMANDER DATA VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Symbol: {self.validation_results.get('symbol', 'N/A')}")
        report.append(f"Timestamp: {self.validation_results.get('timestamp', 'N/A')}")
        report.append("")

        # Summary
        summary = self.validation_results.get('summary', {})
        report.append("📈 SUMMARY:")
        report.append(f"   Total Checks: {summary.get('total_checks', 0)}")
        report.append(f"   Passed: {summary.get('passed', 0)}")
        report.append(f"   Failed: {summary.get('failed', 0)}")
        report.append(f"   Success Rate: {summary.get('success_rate', 'N/A')}")
        report.append("")

        # Detailed results
        report.append("🔍 DETAILED RESULTS:")
        report.append("-" * 60)
        for check_name, check_result in self.validation_results.get('checks', {}).items():
            status = check_result.get('status', 'unknown')
            emoji_map = {
                'pass': '✅',
                'fail': '❌',
                'warning': '⚠️',
                'error': '💥',
                'skip': '⏭️'
            }
            emoji = emoji_map.get(status, '❓')

            report.append(f"{emoji} {check_name.upper()}: {status.upper()}")

            details = check_result.get('details', {})
            if details:
                if isinstance(details, dict):
                    for key, value in details.items():
                        report.append(f"   • {key}: {value}")
                else:
                    # Handle string details
                    report.append(f"   • {details}")

        report.append("")
        report.append("=" * 60)
        return "\n".join(report)

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


# Standalone validation functions for quick checks
def quick_nan_check(df: pd.DataFrame) -> bool:
    """Quick check for NaN values in critical columns."""
    critical_cols = ['close', 'fast_line', 'slow_line']
    available_cols = [col for col in critical_cols if col in df.columns]
    for col in available_cols:
        if df[col].isnull().any():
            return False
    return True


def quick_regime_check(df: pd.DataFrame) -> bool:
    """Quick check for regime distribution."""
    if 'regime' not in df.columns:
        return False
    regime_counts = df['regime'].value_counts()
    # Should have some distribution, not all same regime
    return len(regime_counts) > 1 and regime_counts.max() / len(df) < 0.95


# Example usage and testing
if __name__ == '__main__':
    # Initialize validator
    validator = DataValidator()

    try:
        # Run full validation
        results = validator.run_full_validation('SPY', sample_size=1000)

        # Generate and print report
        report = validator.generate_report()
        print(report)

        # Save results to file
        with open('validation_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)

        print("\n📄 Report saved to: validation_report.txt")

    finally:
        validator.close()