"""
Visualization Script for Commander Strategy
===========================================

Plots the price action, executed trades, and veto zones from the backtest.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from run_backtest import run_complete_backtest
import logging

# Disable info logging for cleaner output
logging.getLogger().setLevel(logging.WARNING)

def plot_backtest_results():
    print("Running backtest to generate data for plotting...")
    metrics, trades, df = run_complete_backtest('$SPX', days=5)
    
    if df.empty:
        print("No data to plot.")
        return

    # Ensure datetime is index for plotting
    if 'datetime_utc' in df.columns:
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
        df.set_index('datetime_utc', inplace=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # 1. Plot Price
    ax.plot(df.index, df['close'], label='Price', color='black', alpha=0.6, linewidth=1)
    
    # 2. Plot VWAP bands if available
    if 'vwap_upper' in df.columns and 'vwap_lower' in df.columns:
        ax.plot(df.index, df['vwap_upper'], color='gray', linestyle='--', alpha=0.3, label='VWAP Bands')
        ax.plot(df.index, df['vwap_lower'], color='gray', linestyle='--', alpha=0.3)
        
    # 3. Plot Trades
    for trade in trades:
        entry_time = pd.to_datetime(trade['entry_time'])
        exit_time = pd.to_datetime(trade['exit_time'])
        
        # Entry marker
        if trade['direction'] == 'long':
            ax.scatter(entry_time, trade['entry_price'], color='green', marker='^', s=120, zorder=5, 
                       label='Long Entry' if 'Long Entry' not in ax.get_legend_handles_labels()[1] else "")
        else:
            ax.scatter(entry_time, trade['entry_price'], color='red', marker='v', s=120, zorder=5, 
                       label='Short Entry' if 'Short Entry' not in ax.get_legend_handles_labels()[1] else "")
            
        # Exit marker
        pnl_color = 'green' if trade['pnl'] > 0 else 'red'
        ax.scatter(exit_time, trade['exit_price'], color=pnl_color, marker='x', s=100, zorder=5, 
                   label='Exit' if 'Exit' not in ax.get_legend_handles_labels()[1] else "")
        
        # Connect entry and exit
        ax.plot([entry_time, exit_time], [trade['entry_price'], trade['exit_price']], 
                color=pnl_color, linestyle='-', linewidth=1.5, alpha=0.7)

    # 4. Shade Veto Zones (Red Background)
    # Combine veto flags
    veto_mask = pd.Series(False, index=df.index)
    if 'veto_iv_crush' in df.columns:
        veto_mask |= df['veto_iv_crush']
    if 'veto_internals' in df.columns:
        veto_mask |= df['veto_internals']
    
    # Use fill_between to shade areas where veto is True
    # We use a transform to shade the full height of the plot
    import matplotlib.transforms as mtransforms
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    
    # Find contiguous regions of veto=True
    # This is a bit tricky with time gaps, so we'll just fill where True
    # A simple way is to fill between y-limits
    ymin, ymax = df['close'].min() * 0.99, df['close'].max() * 1.01
    
    ax.fill_between(df.index, ymin, ymax, where=veto_mask, 
                    facecolor='red', alpha=0.1, transform=ax.get_xaxis_transform(), label='Veto Active')

    # Formatting
    ax.set_title('Commander Strategy: Trades & Veto Zones', fontsize=14)
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Price')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Save
    output_file = 'backtest_chart.png'
    plt.tight_layout()
    plt.savefig(output_file)
    
    abs_path = os.path.abspath(output_file)
    print(f"\n✅ Chart saved to: {abs_path}")
    
    if os.path.exists(output_file):
        print(f"File size: {os.path.getsize(output_file) / 1024:.1f} KB")
    else:
        print("❌ Error: File was not created!")
        
    print("Open this file to see your trades and the veto zones.")
    plt.show()

if __name__ == "__main__":
    plot_backtest_results()
