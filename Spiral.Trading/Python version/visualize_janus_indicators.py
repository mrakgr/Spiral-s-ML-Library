"""
Visualize Janus Factor indicators
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from janus_factor_indicators import JanusFactorIndicators

def plot_janus_indicators(indicators_df: pd.DataFrame, output_file: str = "janus_indicators.html"):
    """Create interactive chart of Janus Factor indicators."""
    
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Direction of Momentum (DOM)', 'Performance Spread (%)', 'Trading Signals'),
        row_heights=[0.4, 0.4, 0.2]
    )
    
    # DOM chart
    fig.add_trace(
        go.Scatter(
            x=indicators_df['date'],
            y=indicators_df['dom'],
            name='DOM',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Add zero line for DOM
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    
    # Performance Spread chart
    fig.add_trace(
        go.Scatter(
            x=indicators_df['date'],
            y=indicators_df['performance_spread'],
            name='Performance Spread',
            line=dict(color='green', width=2)
        ),
        row=2, col=1
    )
    
    # Add zero line for Performance Spread
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
    
    # Trading signals
    signal_colors = {'LONG': 'green', 'SHORT': 'red', 'NEUTRAL': 'gray'}
    signal_values = {'LONG': 1, 'SHORT': -1, 'NEUTRAL': 0}
    
    indicators_df['signal_value'] = indicators_df['signal'].map(signal_values)
    
    for signal in ['LONG', 'SHORT', 'NEUTRAL']:
        signal_data = indicators_df[indicators_df['signal'] == signal]
        fig.add_trace(
            go.Scatter(
                x=signal_data['date'],
                y=signal_data['signal_value'],
                name=signal,
                mode='markers',
                marker=dict(color=signal_colors[signal], size=8, symbol='square')
            ),
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(
        title='Janus Factor Market Timing Indicators',
        height=900,
        showlegend=True,
        hovermode='x unified'
    )
    
    # Update axes
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="DOM", row=1, col=1)
    fig.update_yaxes(title_text="Performance Spread (%)", row=2, col=1)
    fig.update_yaxes(title_text="Signal", row=3, col=1, tickvals=[-1, 0, 1], ticktext=['SHORT', 'NEUTRAL', 'LONG'])
    
    # Save
    fig.write_html(output_file)
    print(f"\n✓ Chart saved to {output_file}")


def main():
    """Generate visualization."""
    
    print("=" * 80)
    print("VISUALIZING JANUS FACTOR INDICATORS")
    print("=" * 80)
    
    db_path = Path("data/trading.db")
    
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return
    
    # Calculate indicators
    with JanusFactorIndicators(db_path) as jf:
        indicators_df = jf.calculate_indicators()
    
    # Create visualization
    print("\n[Creating chart]")
    plot_janus_indicators(indicators_df, output_file="janus_indicators.html")
    
    # Show statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    
    print(f"\nDate range: {indicators_df['date'].min()} to {indicators_df['date'].max()}")
    print(f"Total trading days: {len(indicators_df)}")
    
    print("\nDOM Statistics:")
    print(f"  Current: {indicators_df['dom'].iloc[-1]:.2f}")
    print(f"  Min: {indicators_df['dom'].min():.2f}")
    print(f"  Max: {indicators_df['dom'].max():.2f}")
    print(f"  Mean: {indicators_df['dom'].mean():.2f}")
    
    print("\nPerformance Spread Statistics:")
    print(f"  Current: {indicators_df['performance_spread'].iloc[-1]:.2f}%")
    print(f"  Min: {indicators_df['performance_spread'].min():.2f}%")
    print(f"  Max: {indicators_df['performance_spread'].max():.2f}%")
    print(f"  Mean: {indicators_df['performance_spread'].mean():.2f}%")
    
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
