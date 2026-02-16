import itertools
import pandas as pd
from BacktestingEngine import BacktestingEngine, CommanderStrategy

# Parameter grid
volume_multipliers = [1.5, 2.0, 2.5, 3.0]
adx_cutoffs = [10, 15, 20]
mvv_scales = [0.5, 0.75, 1.0, 1.5]

results = []

engine = BacktestingEngine()
df = engine.run_data_pipeline('SPY', use_csv=True, csv_path='spy_1mo_5m.csv')
engine.close()

for vol_mult, adx_cut, mvv_scale in itertools.product(volume_multipliers, adx_cutoffs, mvv_scales):
    strategy = CommanderStrategy()
    # generate signals with parameters
    df_signals = strategy.generate_entry_signals(df.copy(), vix_data=None,
                                                volume_multiplier=vol_mult,
                                                adx_cutoff=adx_cut,
                                                mvv_scale=mvv_scale)
    trades = strategy.simulate_trades(df_signals, capital=10000.0)
    metrics = strategy.calculate_performance_metrics(trades)
    summary = metrics.get('summary', {})

    results.append({
        'volume_multiplier': vol_mult,
        'adx_cutoff': adx_cut,
        'mvv_scale': mvv_scale,
        'total_trades': summary.get('total_trades', 0),
        'win_rate': summary.get('win_rate', 0),
        'total_pnl': summary.get('total_pnl', 0),
        'profit_factor': summary.get('profit_factor', 0)
    })

# Save results
res_df = pd.DataFrame(results)
res_df.to_csv('sweep_veto_results.csv', index=False)
print('Wrote sweep_veto_results.csv')
