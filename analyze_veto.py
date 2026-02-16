import pandas as pd
from phase1_data_pipeline import DataPipeline
from vectorized_indicators import VectorizedIndicators

# Load the data
pipeline = DataPipeline()
df = pipeline.run_phase1_pipeline()

# Calculate indicators
indicators = VectorizedIndicators(df)
indicators.calculate_drift_indicators()
indicators.calculate_velocity_physics()
indicators.calculate_regime_states()
indicators.calculate_volume_physics()
indicators.calculate_veto_conditions()

# Check veto conditions
veto_cols = [col for col in indicators.df.columns if 'veto' in col.lower() or 'crush' in col.lower()]
print('Veto condition columns:', veto_cols)

# Count veto periods
for col in veto_cols:
    count = indicators.df[col].sum()
    print(f'{col}: {count} periods')

# Check VIX levels
print(f'VIX mean: {indicators.df["vix_close"].mean():.2f}')
print(f'VIX max: {indicators.df["vix_close"].max():.2f}')
print(f'VIX periods > 20: {(indicators.df["vix_close"] > 20).sum()}')
print(f'VIX periods > 25: {(indicators.df["vix_close"] > 25).sum()}')

# Check overall veto
total_veto = indicators.df['veto_iv_crush'].sum()
print(f'Total veto_iv_crush periods: {total_veto}')
print(f'Veto percentage: {total_veto/len(indicators.df)*100:.1f}%')

# Check regime states that would generate signals
regime_counts = indicators.df['regime_state'].value_counts().sort_index()
print(f'\nRegime state distribution:')
for state, count in regime_counts.items():
    print(f'  State {state}: {count} periods')

# Check launch signals (regime_state == 10)
launch_signals = (indicators.df['regime_state'] == 10).sum()
print(f'\nLaunch signals (regime_state=10): {launch_signals}')

# Check how many launch signals were vetoed
launch_and_veto = ((indicators.df['regime_state'] == 10) & indicators.df['veto_iv_crush']).sum()
print(f'Launch signals vetoed: {launch_and_veto}')
print(f'Launch signals allowed: {launch_signals - launch_and_veto}')